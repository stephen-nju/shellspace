# Ray Multi-Node 启动设计

- **日期**：2026-07-07
- **作者**：stephen-nju（经 brainstorming 流程）
- **范围**：`miles` 仓库 bash 启动链路（`run.sh` → `rltrain.sh` → `lib/*.sh`）
- **目标**：在保持单机调用方式 100% 不变的前提下，支持 head + N-1 worker 的多机部署
- **非目标**：不引入 KubeRay Operator / Slurm / `ray up cluster.yaml` 的自动编排（仅支持"调用方在每台机器上 ssh 设置环境变量后跑同一份 run.sh"）

---

## 1. 背景与现状

### 1.1 当前链路

```
run.sh → rltrain.sh → lib/{ray,config,env,args,...}.sh → ray_submit_job(train.py)
```

### 1.2 已存在的多机语义

- `NNODES` 变量在 `args.sh:40` 与 `rl/defaults.sh:35` 定义，默认 1
- `MASTER_ADDR` 在 `config.sh:22` 默认从 `RAY_HEAD_IP` 取
- `NCCL_SOCKET_IFNAME` 在 `env.sh:7` 与 `utils.sh:299` 留空（NCCL 自动选）
- `RAY_NODE_IP` / `RAY_HEAD_IP` 在 `config.sh:18` 默认 `127.0.0.1`

### 1.3 现有缺陷

1. `lib/ray.sh:ray_start()` 总是执行 `ray start --head --node-ip-address ${RAY_NODE_IP:-127.0.0.1}`，**没有任何 head/worker 分支**——多机时每个节点都会尝试拉起独立的 head，集群连不起来。
2. `RANK` 变量从未被读取，worker 节点没有任何"阻塞等待 head 训练结束"的早返机制。
3. `NCCL_SOCKET_IFNAME` 留空——多机时 NCCL 选错接口的概率高。
4. `lib/ray.sh:43` 注释明确写着"多机时移除 --address，改由外部启动脚本指定"——意图有了但代码没实现。

---

## 2. 设计原则

| 原则 | 体现 |
|------|------|
| **完全向后兼容** | 单机调用 `bash run.sh` 0 改动，行为与改动前等价 |
| **最小改动面** | 4 个文件、~30 行代码 |
| **不绑死编排器** | 脚本不感知 K8s/Slurm/裸金属；调用方传 env 即可 |
| **与现有模块化风格一致** | 改动集中在 `lib/ray.sh`；`run.sh` 只加注释，不动逻辑 |
| **可降级** | 若 RANK 未设，自动按单机 head 处理 |

---

## 3. 架构

### 3.1 拓扑

```
┌─────────── Head Node (RANK=0) ───────────┐
│  ray start --head --node-ip-address=HEAD_IP
│       │                                  │
│       ▼                                  │
│  ray_submit_job (train_cmd) ──► GCS ────┼────► Dashboard :8265
│  miles training driver process           │
└──────────────────────────────────────────┘
                  ▲
                  │ NCCL / Ray internal
                  │ (gcs :6379, node-mgr :8076)
                  │
┌─────────── Worker Node (RANK=1..N-1) ─────┐
│  ray start --address=HEAD_IP:6379        │
│       --node-ip-address=WORKER_IP        │
│  （阻塞等待；不启动训练）                  │
└──────────────────────────────────────────┘
```

### 3.2 改动文件清单

| 文件 | 改动 |
|------|------|
| `lib/ray.sh:ray_start()` | 增加 `RANK` 分支；worker 走 `ray start --address=...` |
| `lib/config.sh:apply_config_defaults()` | 新增 `RANK` / `NNODES` 默认值；`MASTER_ADDR` 与 `RAY_HEAD_IP` 一致性断言 |
| `rltrain.sh:main()` | 在 `ray_start` 之后、配置打印之前，对 `RANK>0` 早返 `wait` |
| `lib/base/env.sh:setup_env()` | 注入 `NCCL_SOCKET_IFNAME`（仅多机时）；worker 上设置 `no_proxy` 包含 `RAY_HEAD_IP` |
| `run.sh` | **只加注释**，不加逻辑 |

`args.sh` / `rl/defaults.sh` / `cmd.sh` / `model_config.sh` 不动。

---

## 4. 数据流

### 4.1 启动流程

```
[head_node] bash
  ssh user@head "cd /opt/.../miles && \
    export RANK=0 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10 NNODES=2 && \
    bash run.sh ..."
  run.sh → rltrain.sh
  main() →
    apply_config_defaults()      [RANK=0, MASTER_ADDR=10.0.1.10]
    setup_signal_handlers()
    setup_env()                   [NCCL_SOCKET_IFNAME: 用户传入]
    detect_hardware()
    ray_start()                   [RANK=0 → ray start --head ...]
    build_runtime_env_json()      [含 MASTER_ADDR=10.0.1.10]
    ray_submit_job(train_cmd)     [向 head 的 GCS 提交]
    ray job logs --follow         [阻塞流日志]

[worker_node] bash
  ssh user@worker "cd /opt/.../miles && \
    export RANK=1 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 NNODES=2 && \
    bash run.sh ..."
  run.sh → rltrain.sh
  main() →
    apply_config_defaults()      [RANK=1, MASTER_ADDR=10.0.1.10]
    setup_signal_handlers()
    setup_env()
    detect_hardware()
    ray_start()                   [RANK=1 → ray start --address=10.0.1.10:6379 --node-ip-address=10.0.2.20]
    [worker 早返] wait            [阻塞；不提交 job]

[head_node] ray job 内部
  miles driver process
    torch.distributed init (MASTER_ADDR=10.0.1.10, RANK=0)
    megatron/actor 在 head 起
    rollout engine 由 SGLANG_ROUTER_IP 控制
```

### 4.2 关键变量

| 变量 | 含义 | 单机默认 | 多机注入（按节点） |
|------|------|----------|--------------------|
| `RANK` | 节点 rank（0=head） | 0（空视为 0） | head:0, worker:1..N-1 |
| `NNODES` | 节点总数 | 1 | 所有节点都填 N |
| `RAY_NODE_IP` | **本机**真实网卡 IP | 127.0.0.1 | 各节点填自己 IP |
| `RAY_HEAD_IP` | head 节点 IP | 127.0.0.1 | 所有节点填同一个 head IP |
| `RAY_PORT` | GCS 端口 | 6379 | 不变 |
| `MASTER_ADDR` | torch distributed master | `${RAY_HEAD_IP:-127.0.0.1}` 展开值 | `${RAY_HEAD_IP}`（与 RAY_HEAD_IP 必须一致；多机时通过断言校验） |
| `NCCL_SOCKET_IFNAME` | NCCL 通信网卡 | 空（不动） | 每节点传自己网卡名（如 eth0/ib0/ens5）；不传则依赖 NCCL 自动选 |
| `NCCL_IB_HCA` | IB HCA 名 | 空 | 多机 IB 时按需传（如 mlx5_0） |

<!-- review patch: F8 — 关键变量表补 MASTER_ADDR 默认值是 bash 变量展开结果，避免"等于 RAY_HEAD_IP"的同义反复 -->

---

## 5. 关键改动细节

### 5.1 `lib/ray.sh:ray_start()`

```bash
ray_start() {
    # ... 现有清理逻辑（pkill ray/sglang）...

    log_info "Starting Ray cluster..."
    log_info "  RANK: ${RANK:-0}"
    log_info "  RAY_NODE_IP: ${RAY_NODE_IP}"
    log_info "  RAY_HEAD_IP: ${RAY_HEAD_IP}"

    local ray_gpus="${RAY_NUM_GPUS:-}"
    if [ -z "$ray_gpus" ]; then
        ray_gpus="$((${ACTOR_NUM_GPUS_PER_NODE:-0} + ${ROLLOUT_NUM_GPUS:-0}))"
    fi

    local node_ip="${RAY_NODE_IP:-127.0.0.1}"
    local head_ip="${RAY_HEAD_IP:-127.0.0.1}"
    local port="${RAY_PORT:-6379}"

    if [ "${RANK:-0}" = "0" ]; then
        # head 节点
        ray start --head \
            --node-ip-address "${node_ip}" \
            --port "${port}" \
            --num-gpus "${ray_gpus}" \
            --disable-usage-stats \
            --dashboard-host="${RAY_DASHBOARD_HOST}" \
            --dashboard-port="${RAY_DASHBOARD_PORT}"
    else
        # worker 节点
        ray start \
            --node-ip-address "${node_ip}" \
            --address "${head_ip}:${port}" \
            --num-gpus "${ray_gpus}" \
            --block
    fi

    # ... 现有 is_ray_running 等待逻辑保留 ...

    # 多机 worker 模式额外验证 GCS 可达 (F9):
    # is_ray_running 只检查 ray:: 进程在不在; 进程能起来但 GCS 连不通时仍会
    # 被认为成功, 然后进入无限 wait. 这里用 ray status 显式 ping head GCS.
    if [ "${RANK:-0}" != "0" ]; then
        if ! ray status --address "${head_ip}:${port}" >/dev/null 2>&1; then
            log_error "Worker cannot reach GCS at ${head_ip}:${port}"
            return 1
        fi
    fi
}
```

**`--block` 行为**：Ray 文档说明 worker `ray start` 加 `--block` 会让进程前台阻塞直到 head 断开；这正是我们需要的"worker 早返后阻塞等待"语义。SIGINT 在 raylet 内部被 handle（exit 130），SIGTERM 在 Ray 2.5+ 也会 graceful shutdown（exit 143），但建议优先用 `ray stop` 而非原始信号。

<!-- review patch: F9 — worker 模式额外用 ray status 验证 GCS 连通，避免 pgrep 假阳性 -->

### 5.2 `lib/config.sh:apply_config_defaults()`

```bash
# 在 RAY 段尾追加：
RANK="${RANK:-0}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-${RAY_HEAD_IP}}"

# 多机一致性断言（仅 NNODES>1 时）
if [ "${NNODES}" -gt 1 ]; then
    if [ "${MASTER_ADDR}" != "${RAY_HEAD_IP}" ]; then
        log_error "MASTER_ADDR (${MASTER_ADDR}) != RAY_HEAD_IP (${RAY_HEAD_IP}); multi-node requires they match"
        return 1
    fi
    # 防御性兜底 (F1): NNODES>1 但 RANK 未显式给出, 默认 0 意味着 worker
    # 漏 export RANK 时会假装自己是 head 然后抢占 6379. 用 warn 提示而非报错
    # 是因为单机调用方式 "bash run.sh" 的 RANK 也不显式给 (默认 0), 该路径必须保持不变.
    if [ -z "${RANK+x}" ]; then
        log_warn "NNODES>1 but RANK unset; defaulting to RANK=0 (head). If this is a worker, set RANK=1..N-1 explicitly."
    fi
fi
```

<!-- review patch: F1 — NNODES>1 + RANK 未显式给时加 warn，避免 worker 漏 export 静默抢 head 端口 -->

### 5.3 `rltrain.sh:main()`

```bash
main() {
    # ... 现有 1-6 步保持 ...
    # apply_config_defaults
    # apply_rl_defaults / apply_sft_defaults
    # apply_megatron_overrides
    # load_model_args_from_file
    # run_validation
    # setup_signal_handlers
    # setup_env
    # detect_hardware
    # polar 相关
    ray_start || { log_error "Ray 启动失败"; exit 1; }

    # ── 新增：worker 早返 ──────────────────────────────
    if [ "${RANK:-0}" != "0" ]; then
        log_info "Worker node (RANK=${RANK}); blocking until head stops Ray."
        # --block 让 ray start 进程前台阻塞; head 端 ray stop --force 时
        # raylet 检测到 GCS 断开会退出, 子进程结束 -> wait 返回 -> exit trap -> cleanup.
        # 注意: 此时 cleanup 会再调一次 `ray stop --force`, 因 worker 上已无 ray 进程
        # 故无害 (ray stop 对无 ray 集群是 no-op).
        wait
        exit 0
    fi
    # ── worker 早返结束 ─────────────────────────────────

    build_runtime_env_json
    # ... 后续打印配置 / 启动训练 ...
}
```

**`wait` 行为**：`wait`（无参数）会等待**所有子进程**结束；当 head 通过 `ray stop` 关闭 cluster 时，worker 的 `ray start --block` 会退出，`wait` 随之返回，`rltrain.sh` 退出码 0。Ray 2.5+ 对 SIGTERM 也有 graceful shutdown（PR #35201，`RAY_GRACEFUL_SHUTDOWN_TIMEOUT` 默认 30s）；SIGINT 走 raylet 内部 handler（exit 130）。优先用 `ray stop` 而非原始信号。

<!-- review patch: F2 — 显式说明 worker wait 退出链：ray --block 退出 → wait 返回 → EXIT trap → cleanup；cleanup 里再调 ray stop 在 worker 上是 no-op 故无害 -->
<!-- review patch: F2-extra — 补充 Ray 2.5+ SIGTERM graceful shutdown 与 SIGINT exit 130 的真值，便于读者设 terminationGracePeriodSeconds / TimeoutStopSec ≥ 30s -->

### 5.4 `lib/base/env.sh:setup_env()`

```bash
setup_env() {
    # ... 现有 NCCL_* 变量 ...
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}

    # 多机时 (F4): 既未设 NCCL_SOCKET_IFNAME 也未设 NCCL_IB_HCA 时,
    # 不要默认 eth0 (在 IB / RoCE / 云上会错). 改为 warn + 不注入,
    # 让 NCCL 自己选. Ray driver 日志会显示实际选择的接口, 错了再改 env 即可.
    if [ "${NNODES:-1}" -gt 1 ] && [ -z "${NCCL_SOCKET_IFNAME:-}" ] && [ -z "${NCCL_IB_HCA:-}" ]; then
        log_warn "NNODES>1 but neither NCCL_SOCKET_IFNAME nor NCCL_IB_HCA set."
        log_warn "NCCL will auto-select interface; if it picks the wrong one, set explicitly (e.g. ib0 / eth0 / ens5)."
    fi

    # ... 其余现有逻辑 ...
}
```

<!-- review patch: F4 — 多机时去掉 eth0 硬默认；IB 场景默认 eth0 会强制走 TCP 反倒降级，云/容器上 eth0 也常不存在 -->
<!-- review patch: F5 — ray_gpus 当前公式是"每节点 = actor + rollout"的简化假设；若 actor 只在 head 跑（--actor-num-nodes 1），worker 上应只算 rollout，需用户手动 export RAY_NUM_GPUS=$ROLLOUT_NUM_GPUS on workers -->
**`--num-gpus` 的语义**（F5）：
- 当前实现：`RAY_NUM_GPUS` 优先，否则 = `ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS`
- 这是"每节点 GPU 总数"的简化假设；假设每个节点都跑 actor + rollout
- 如果 actor 只在 head 跑（例如 `--actor-num-nodes 1`），worker 上 `--num-gpus` 应只算 rollout（用户需手动在 worker 端 export `RAY_NUM_GPUS=$ROLLOUT_NUM_GPUS`）
- 在多机文档里需要说明这一限制

### 5.5 `run.sh`（仅注释，不改逻辑）

在 `SGLANG_HOST_IP=127.0.0.1` 后追加 12 行注释块，说明多机用法：

```bash
# ── 多机部署 (optional) ───────────────────────────────────────
# Head 节点:
#   RANK=0 RAY_NODE_IP=<head_ip> RAY_HEAD_IP=<head_ip> NNODES=N ./run.sh ...
# Worker 节点 (RANK=1..N-1):
#   RANK=<i> RAY_NODE_IP=<worker_ip> RAY_HEAD_IP=<head_ip> NNODES=N ./run.sh ...
# 必传:
#   RAY_NODE_IP   本机真实网卡 IP (不要用 127.0.0.1)
#   RAY_HEAD_IP   head 节点 IP (所有节点必须一致)
#   NCCL_SOCKET_IFNAME  本机通信网卡名 (默认 eth0，仅多机时)
# 可选:
#   NCCL_IB_HCA   IB HCA 名 (如 mlx5_0)
#   NCCL_DEBUG=INFO  打开 NCCL 调试日志
# ──────────────────────────────────────────────────────────────
```

---

## 6. 错误处理

| 失败场景 | 检测点 | 行为 |
|----------|--------|------|
| worker `ray start --address` 连不上 head | `is_ray_running` 30s 内 false | log_error + return 1；cleanup 触发 `ray stop` |
| head 端口 6379/8265 被占 | `ray start --head` 退出非 0 | log_error + return 1 |
| `RAY_NODE_IP` 不可路由 | `is_ray_running` 超时 | 同上；日志建议检查 `RAY_NODE_IP` |
| `MASTER_ADDR` ≠ `RAY_HEAD_IP`（多机时） | `apply_config_defaults` 断言 | log_error + return 1 |
| worker 收到 SIGTERM/SIGINT | `setup_signal_handlers` | cleanup 钩子 `ray stop` 后 exit |
| NCCL 跨机通信失败 | 训练 driver 自己报错 | 不在本脚本范围；保留 `NVTE_DEBUG=1` 默认 |
| head 退出后 worker 仍未退出 | 依赖 `ray start --block` 自动退出 | 若 60s 未退出，cleanup SIGTERM 兜底 |

---

## 7. 测试

### 7.1 单元级（dry-run）

| 场景 | 命令 | 期望 |
|------|------|------|
| 单机 dry-run | `DRY_RUN=true ./rltrain.sh --train-mode rl ...` | 打印 `ray start --head --node-ip-address 127.0.0.1 ...` |
| head dry-run | `DRY_RUN=true RANK=0 RAY_NODE_IP=10.0.1.10 ./rltrain.sh ...` | 打印 `ray start --head --node-ip-address 10.0.1.10 ...` |
| worker dry-run | `DRY_RUN=true RANK=1 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 ./rltrain.sh ...` | 打印 `ray start --address=10.0.1.10:6379 --node-ip-address 10.0.2.20 ...` |
| RANK 未设 | `./rltrain.sh ...` | 行为与改动前完全一致 |
| 多机 head | `DRY_RUN=true NNODES=2 RANK=0 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10 ./rltrain.sh ...` | 打印 `ray start --head --node-ip-address 10.0.1.10 ...`；不触发 MASTER_ADDR 断言 |
| 多机 worker | `DRY_RUN=true NNODES=2 RANK=1 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 ./rltrain.sh ...` | 打印 `ray start --address=10.0.1.10:6379 ...`；不进入 `wait` 阻塞（dry-run 路径直接返回） |
| 不一致断言 | `DRY_RUN=true NNODES=2 RANK=0 MASTER_ADDR=10.0.0.99 RAY_HEAD_IP=10.0.1.10 ./rltrain.sh ...` | log_error `MASTER_ADDR (10.0.0.99) != RAY_HEAD_IP (10.0.1.10)`；非 0 退出 |

<!-- review patch: F6 — 补充 3 个 dry-run 用例：多机 head、多机 worker、MASTER_ADDR 不一致断言回归保护 -->

### 7.2 集成级（两台测试机）

```bash
# head
RANK=0 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10 NNODES=2 \
  NCCL_SOCKET_IFNAME=eth0 \
  ./run.sh --train-mode rl -n test --num-rollout 4 ...

# worker (另一个终端)
RANK=1 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 NNODES=2 \
  NCCL_SOCKET_IFNAME=eth0 \
  ./run.sh --train-mode rl -n test --num-rollout 4 ...
```

验证：
- `ray status` 在 head 上看到 2 nodes
- worker 上 `nvidia-smi` 显示被 actor/rollout 占用
- 训练日志中 `actor_num_nodes=2`

### 7.3 回归

- 单机 `bash run.sh` 与改动前**完全等价**——重点回归：head 拉起、job 提交、训练启动、cleanup 退出码。
- `MILES_SCRIPT_EXTERNAL_RAY=true` 路径不回归（`ray_start` 仍执行但不破坏外部 ray cluster）。

---

## 8. 向后兼容矩阵

| 调用方 | 改动前 | 改动后 |
|--------|--------|--------|
| `bash run.sh` 单机 | ✅ 正常 | ✅ 行为完全一致 |
| `rltrain.sh --train-mode rl` | ✅ 正常 | ✅ 行为完全一致 |
| `MILES_SCRIPT_EXTERNAL_RAY=true` | 走外部 ray | ✅ 行为完全一致（注：当前 `lib/ray.sh` 未检查此 env；如启用，请同时在 `ray_start` 头部加 `if [ "${MILES_SCRIPT_EXTERNAL_RAY:-false}" = "true" ]; then return 0; fi`；本 spec 不强制） |
| `--dry-run` 打印 | 仅 head | head 时打印 head，worker 时打印 worker |
| `rltrain.sh` 退出码 | 0/1 | 0/1；worker 早返路径固定 0 |

<!-- review patch: F7 — §8 MILES_SCRIPT_EXTERNAL_RAY=true 行注明"当前 lib/ray.sh 未检查此 env"，避免读者误以为已实现 -->

**唯一新增行为**：`RANK>0` 时 `rltrain.sh` 在 `ray_start` 后**早返**——这是新增能力，不是回归。

---

## 9. 部署文档

在 `docs/multinode.md` 新增（与本 spec 同目录，但属于用户文档）：

- 硬件前置：节点间互通 6379/8265/8076/10001-19999
- 共享存储：HF_HOME、SAVE_DIR 必须在所有节点可访问
- 启动顺序：先 head 后 worker（head 端 `rltrain.sh` 会在 `ray_start` 后立即进入 job 提交；worker 端建议在 head 的 `ray_start` 日志出现后再启动，避免 30s 连通性超时误报；Ray worker 支持后启动自动 reconnect，但当前脚本的 30s 等待窗口不允许这个间隔）
- 关停顺序：先 `Ctrl+C` head（触发 `ray stop --force` + cleanup），worker 因 `ray start --block` 检测到 GCS 断开自动退出
- 故障排查：NCCL_SOCKET_IFNAME 错配、`MASTER_ADDR` 不一致、`RAY_NODE_IP` 不可达

本 spec 不规定 `docs/multinode.md` 的具体内容；属于实现阶段产物。

---

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| worker `ray start --block` 不会响应 Ctrl+C | cleanup 钩子用 `pkill -INT -f "ray::"` 兜底 |
| `wait` 在某些 shell 行为差异 | 用 `bash`（脚本 shebang 已是 bash）；不依赖具体版本 |
| NCCL_SOCKET_IFNAME 默认 eth0 在某些云上不对 | 多机时设默认值但加 `log_warn` 提示用户可覆盖 |
| `MILES_SCRIPT_EXTERNAL_RAY=true` + 多机 + `RANK=0` 会导致重复 head | 文档化约束：外部 ray 模式下不调 `rltrain.sh`，或 `RANK>0` 时跳过 `ray_start` |
| head 端 job 失败后 worker 仍阻塞 | 文档化"失败时手动 `ray stop` 关 head，worker 自动退出" |

---

## 11. 后续（非本 spec 范围）

- KubeRay Operator 适配（CRD + 注入 `RAY_HEAD_SERVICE_HOST` / `RAY_HEAD_SERVICE_PORT`，或 `RAY_ADDRESS` 二选一；以集群实际 Service 名为准）
- Slurm 适配（PMI/PMIx 注入 `RANK` / `NNODES` / `MASTER_ADDR`）
- 自动化 worker 启动（从 head ssh 到各 worker）
- `cluster.yaml` 路径（`ray up` 模式）

<!-- review patch: F10 — §11 修正 KubeRay 引用：实际惯例是 `RAY_HEAD_SERVICE_HOST/PORT`，`RAY_ADDRESS` 不是 KubeRay 标准；改为二选一并注实际集群 Service 名为准 -->
<!-- review patch: F11 — §9 启动顺序段补"30s 硬超时窗口不允许 head/worker 间隔太久"的设计取舍说明 -->

这些属于独立 spec。
