# Ray Multi-Node 启动实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 4 个 bash 文件 + 1 个入口注释中实现 `miles` 多机启动支持（head + N-1 worker），同时保证单机调用 `bash run.sh` 行为完全不变。

**Architecture:** 以 `RANK` 环境变量为驱动：head（`RANK=0`）走 `ray start --head`，worker（`RANK>=1`）走 `ray start --address=HEAD_IP:6379 --block`，worker 进程前台阻塞至 head 关闭；新增 `apply_config_defaults` 一致性断言防御多机误配。

**Tech Stack:** bash 5.x、Ray CLI（`ray start` / `ray status` / `ray job submit`）、`getopt`、GNU coreutils（`pgrep`、`grep`、`sed`、`awk`、`tr`）。

**Source Spec:** `docs/superpowers/specs/2026-07-07-ray-multinode-design.md` (HEAD = 4aa79e8, after 11 review patches).

## Global Constraints

| 项 | 值 | 出处 |
|---|---|---|
| Shell | bash 5.x，`set -euo pipefail` 已在所有脚本顶部 | lib/*.sh 全局 |
| 日志接口 | `log_info`/`log_warn`/`log_error`（来自 `lib/base/log.sh`），所有提示必须走这些函数 | lib/base/log.sh:8-24 |
| 默认端口 | GCS :6379、Dashboard :8265、node-mgr :8076、client :10001 | spec §3.1 (F3 patch) |
| 单机默认 | `RAY_NODE_IP=127.0.0.1`、`RAY_HEAD_IP=127.0.0.1`、`MASTER_ADDR=$RAY_HEAD_IP`、`NNODES=1`、`RANK=0` | lib/config.sh:18-22, spec §4.2 |
| Ray 信号行为 | SIGINT exit 130、SIGTERM exit 143（Ray 2.5+ graceful，`RAY_GRACEFUL_SHUTDOWN_TIMEOUT` 默认 30s） | spec §5.3 F2 patch |
| 改动文件数 | 5（`lib/ray.sh`、`lib/config.sh`、`rltrain.sh`、`lib/base/env.sh`、`run.sh`） | spec §3.2 |
| 不改文件 | `lib/args.sh`、`lib/rl/defaults.sh`、`lib/cmd.sh`、`lib/model_config.sh`、所有 `lib/sft/*`、`lib/polar.sh` | spec §3.2 |
| 测试方法 | `DRY_RUN=true` + bash 文本捕获（无测试框架；引入 bats 超 YAGNI） | spec §7.1 |
| commit 粒度 | 每个任务独立 commit | writing-plans 约定 |

---

## File Structure

| 文件 | 改动函数 | 改动行数估计 |
|---|---|---|
| `lib/config.sh` | `apply_config_defaults()` | +10 行（RAY 段尾追加） |
| `lib/ray.sh` | `ray_start()` | 重构：~25 行（替换原 `ray start --head` 段，加 GCS 探测） |
| `rltrain.sh` | `main()` | +7 行（worker 早返段插在 ray_start 之后） |
| `lib/base/env.sh` | `setup_env()` | 替换第 7 行单行 + 加多机 warn 段，~8 行净增 |
| `run.sh` | （无逻辑） | +12 行注释块 |

接口契约（其他任务可见）：
- `apply_config_defaults` 必须保证调用后 `$RANK` / `$NNODES` / `$MASTER_ADDR` 三个变量都有值，且 NNODES>1 时三者之间的一致性已被断言过
- `ray_start` 调用前：变量已由 `apply_config_defaults` 填好；调用后：head 模式返回 0 表示 ray head 已起，worker 模式返回 0 表示 GCS 可达
- `main` 在 `ray_start` 之后才决定是否早返；早返路径只走 worker

---

## Task 1: `lib/config.sh` 多机默认值与一致性断言

**Files:**
- Modify: `lib/config.sh:6-51`（`apply_config_defaults` 函数体）
- Test: `tests/config_defaults_dryrun.sh`（新建）

**Interfaces:**
- Consumes: 环境变量 `RANK`、`NNODES`、`MASTER_ADDR`、`RAY_HEAD_IP`（可能未设）
- Produces: 调用后 `$RANK` / `$NNODES` / `$MASTER_ADDR` 必有值；NNODES>1 时三者一致性已校验

### Steps

- [ ] **Step 1.1: 写失败的 dry-run 测试**

创建 `tests/config_defaults_dryrun.sh`：

```bash
#!/bin/bash
# tests/config_defaults_dryrun.sh
# 验证 apply_config_defaults 在不同 RANK/NNODES 组合下的行为
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/lib/base/colors.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/lib/base/log.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

# 单机默认: RANK 0, NNODES 1, MASTER_ADDR=127.0.0.1
(
    unset RANK NNODES MASTER_ADDR RAY_HEAD_IP
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/lib/config.sh"
    [ "${RANK:-unset}" = "0" ]   || fail "单机 RANK 默认应为 0, 实际 ${RANK:-unset}"
    [ "${NNODES:-unset}" = "1" ] || fail "单机 NNODES 默认应为 1, 实际 ${NNODES:-unset}"
    [ "${MASTER_ADDR:-unset}" = "127.0.0.1" ] || fail "单机 MASTER_ADDR 默认应为 127.0.0.1, 实际 ${MASTER_ADDR:-unset}"
    pass "单机默认值"
) || exit 1

# 多机 head 一致: RANK=0, NNODES=2, RAY_HEAD_IP=10.0.1.10, MASTER_ADDR 显式一致
(
    export RANK=0 NNODES=2 RAY_HEAD_IP=10.0.1.10 MASTER_ADDR=10.0.1.10
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/lib/config.sh"
    [ "${RANK}" = "0" ]            || fail "head RANK 应保持 0"
    [ "${MASTER_ADDR}" = "10.0.1.10" ] || fail "MASTER_ADDR 应保持 10.0.1.10"
    pass "多机 head 一致"
) || exit 1

# 多机一致断言: NNODES=2 但 MASTER_ADDR != RAY_HEAD_IP 应 return 1
(
    export RANK=0 NNODES=2 RAY_HEAD_IP=10.0.1.10 MASTER_ADDR=10.0.0.99
    # shellcheck disable=SC1090
    set +e
    source "${REPO_ROOT}/lib/config.sh"
    rc=$?
    set -e
    [ "${rc}" = "1" ] || fail "不一致断言应 return 1, 实际 ${rc}"
    pass "MASTER_ADDR 不一致触发断言"
) || exit 1

# NNODES>1 但 RANK 未设 -> 默认 0 + log_warn（不应报错，仅 warn）
(
    unset RANK
    export NNODES=2 RAY_HEAD_IP=10.0.1.10 MASTER_ADDR=10.0.1.10
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/lib/config.sh"
    [ "${RANK}" = "0" ] || fail "NNODES>1 + RANK 未设, 默认应为 0"
    pass "F1 兜底: NNODES>1 + RANK 未设 -> 默认 0"
) || exit 1

echo "ALL PASS"
```

- [ ] **Step 1.2: 跑测试，确认失败**

```bash
chmod +x tests/config_defaults_dryrun.sh
bash tests/config_defaults_dryrun.sh
```

期望：`FAIL: 单机 RANK 默认应为 0, 实际 unset` 之类的错误（当前 `apply_config_defaults` 不读 `RANK`）。

- [ ] **Step 1.3: 修改 `lib/config.sh:apply_config_defaults`**

编辑 `lib/config.sh`，在 `MASTER_ADDR="${MASTER_ADDR:-${RAY_HEAD_IP}}"` 之后追加多机段。**关键：F1 warn 必须在 `RANK="${RANK:-0}"` 覆盖之前**，否则 `${RANK+x}` 永远为非空：

```bash
	# ── 多机语义 (spec §4.2, §5.2) ─────────────────────────────
	# F1 兜底: 在覆盖默认值之前先检测 RANK 是否显式给过
	if [ -z "${RANK+x}" ] && [ "${NNODES:-1}" -gt 1 ]; then
		log_warn "NNODES>1 but RANK unset; defaulting to RANK=0 (head). If this is a worker, set RANK=1..N-1 explicitly."
	fi
	RANK="${RANK:-0}"
	NNODES="${NNODES:-1}"

	# 多机一致性断言
	if [ "${NNODES}" -gt 1 ]; then
		if [ "${MASTER_ADDR}" != "${RAY_HEAD_IP}" ]; then
			log_error "MASTER_ADDR (${MASTER_ADDR}) != RAY_HEAD_IP (${RAY_HEAD_IP}); multi-node requires they match"
			return 1
		fi
	fi
```

- [ ] **Step 1.4: 跑测试，确认通过**

```bash
bash tests/config_defaults_dryrun.sh
```

期望：`ALL PASS` 输出，每个 case 各有 `PASS: ...` 一行。

- [ ] **Step 1.5: 跑回归（单机 0 改动）**

```bash
DRY_RUN=true bash rltrain.sh --train-mode rl -n test -m /tmp -d /tmp/data.jsonl 2>&1 | head -20
```

期望：dry-run 输出 `ray start --head --node-ip-address 127.0.0.1 ...`，无 `MASTER_ADDR != RAY_HEAD_IP` 报错。

- [ ] **Step 1.6: Commit**

```bash
git add lib/config.sh tests/config_defaults_dryrun.sh
git commit -m "feat(config): multi-node defaults + RANK/MNODES/MASTER_ADDR consistency assertion

- RANK defaults to 0 when unset (single-node fallback)
- NNODES defaults to 1
- Assert MASTER_ADDR == RAY_HEAD_IP when NNODES>1 (returns 1 on mismatch)
- F1: warn when NNODES>1 but RANK unset before defaulting to 0
- Tests: tests/config_defaults_dryrun.sh covers 4 cases"
```

---

## Task 2: `lib/ray.sh` `ray_start()` RANK 分支 + GCS 探测

**Files:**
- Modify: `lib/ray.sh:18-65`（`ray_start` 函数体）
- Test: `tests/ray_start_dryrun.sh`（新建）

**Interfaces:**
- Consumes: `$RANK`（已由 Task 1 填好）、`$RAY_NODE_IP`、`$RAY_HEAD_IP`、`$RAY_PORT`、`$RAY_NUM_GPUS`、`$ACTOR_NUM_GPUS_PER_NODE`、`$ROLLOUT_NUM_GPUS`、`$RAY_DASHBOARD_HOST`、`$RAY_DASHBOARD_PORT`
- Produces: head 模式返回 0 = ray head 进程已起；worker 模式返回 0 = GCS 可达（`ray status` 通过）

### Steps

- [ ] **Step 2.1: 写失败的 dry-run 测试**

创建 `tests/ray_start_dryrun.sh`：

```bash
#!/bin/bash
# tests/ray_start_dryrun.sh
# 验证 ray_start 在不同 RANK 组合下生成的 ray 命令
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
LOG_FILE="/tmp/ray_start_dryrun_$$.log"
mkdir -p /tmp

# 用一个 shim 让 ray_submit_job 等不存在的函数被记录, 并替换 ray 为 echo
shim_dir="$(mktemp -d)"
cat > "${shim_dir}/ray" <<'EOF'
#!/bin/bash
echo "RAY_CALL: $*" >&2
EOF
chmod +x "${shim_dir}/ray"

PATH="${shim_dir}:${PATH}" \
LOG_FILE="${LOG_FILE}" \
    "${REPO_ROOT}/lib/ray.sh"

# 提取每次调用 ray 的参数
extract_ray_args() {
    grep -oE 'RAY_CALL: [^&]+' /tmp/ray_start_dryrun_*.log | sed 's/RAY_CALL: //' | tr -s ' ' '\n' | grep '^--' | sort -u
}

# 1) 单机 head
(
    unset RANK
    export RAY_NODE_IP=127.0.0.1 RAY_HEAD_IP=127.0.0.1 RAY_PORT=6379
    export ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
    export LOG_FILE=/tmp/single_$$.log
    ray_start 2>/dev/null || true
    out="$(extract_ray_args)"
    echo "$out" | grep -q "^--head$"        || { echo "FAIL: 单机应包含 --head"; exit 1; }
    echo "$out" | grep -q "^--node-ip-address 127.0.0.1$" || { echo "FAIL: 单机 node-ip 应为 127.0.0.1"; exit 1; }
    echo "$out" | grep -q "^--num-gpus 8$"  || { echo "FAIL: 单机 num-gpus 应为 8 (4+4)"; exit 1; }
    echo "$out" | grep -q "^--dashboard-port 8265$" || true  # 可能
    echo "PASS: 单机 head dry-run"
)

# 2) 多机 head
(
    export RANK=0 NNODES=2 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10 RAY_PORT=6379
    export ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
    export LOG_FILE=/tmp/head_$$.log
    ray_start 2>/dev/null || true
    out="$(extract_ray_args)"
    echo "$out" | grep -q "^--head$"          || { echo "FAIL: 多机 head 应包含 --head"; exit 1; }
    echo "$out" | grep -q "^--node-ip-address 10.0.1.10$" || { echo "FAIL: head node-ip 应为 10.0.1.10"; exit 1; }
    echo "PASS: 多机 head dry-run"
)

# 3) 多机 worker
(
    export RANK=1 NNODES=2 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 RAY_PORT=6379
    export ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
    export LOG_FILE=/tmp/worker_$$.log
    # worker 模式会调 ray status, 这在 shim 里不会成功 -> ray_start 返回 1
    # 我们只验证命令格式, 接受退出码非 0
    ray_start 2>/dev/null
    rc=$?
    [ "${rc}" = "1" ] || { echo "FAIL: worker 在 GCS 不可达时返回 1, 实际 ${rc}"; exit 1; }
    out="$(extract_ray_args)"
    echo "$out" | grep -q "^--node-ip-address 10.0.2.20$" || { echo "FAIL: worker node-ip 应为 10.0.2.20"; exit 1; }
    echo "$out" | grep -q "^--address 10.0.1.10:6379$"   || { echo "FAIL: worker 应连 10.0.1.10:6379"; exit 1; }
    echo "$out" | grep -q "^--block$"                    || { echo "FAIL: worker 应带 --block"; exit 1; }
    echo "PASS: 多机 worker dry-run"
)

echo "ALL PASS"
```

- [ ] **Step 2.2: 跑测试，确认失败**

```bash
chmod +x tests/ray_start_dryrun.sh
bash tests/ray_start_dryrun.sh
```

期望：单机 case 失败（当前 `ray_start` 总是走 `--head` 且不区分 RANK）。注意 worker case 的 shim 方法较粗糙，可能要先简化跑一次看真实失败信息。

- [ ] **Step 2.3: 重构 `lib/ray.sh:ray_start`**

把 `ray_start()` 整个函数替换为：

```bash
ray_start() {
	# ── 启动前清理 (参考 retool 脚本: 防止旧进程干扰) ─────────────────────
	log_info "Cleaning up residual processes before launch..."
	pkill -9 sglang 2>/dev/null || true
	ray stop --force 2>/dev/null || true
	sleep 2
	pkill -9 -f "ray::" 2>/dev/null || true
	pkill -9 -f "train_async" 2>/dev/null || true
	sleep 2
	# 二次确认
	pkill -9 -f "ray::" 2>/dev/null || true
	sleep 1

	log_info "Starting Ray cluster..."
	log_info "  RANK: ${RANK:-0}"
	log_info "  RAY_NODE_IP: ${RAY_NODE_IP}"
	log_info "  RAY_HEAD_IP: ${RAY_HEAD_IP}"

	# GPU 数: RAY_NUM_GPUS 显式设置优先；默认 = actor + rollout (RL/SFT 通用)
	# 注 (spec F5): 简化假设每节点都跑 actor+rollout; 若 actor 只在 head,
	# 用户需在 worker 端 export RAY_NUM_GPUS=$ROLLOUT_NUM_GPUS
	local ray_gpus="${RAY_NUM_GPUS:-}"
	if [ -z "$ray_gpus" ]; then
		ray_gpus="$((${ACTOR_NUM_GPUS_PER_NODE:-0} + ${ROLLOUT_NUM_GPUS:-0}))"
	fi

	local node_ip="${RAY_NODE_IP:-127.0.0.1}"
	local head_ip="${RAY_HEAD_IP:-127.0.0.1}"
	local port="${RAY_PORT:-6379}"

	if [ "${RANK:-0}" = "0" ]; then
		# head 节点 (spec §5.1)
		ray start --head \
			--node-ip-address "${node_ip}" \
			--port "${port}" \
			--num-gpus "${ray_gpus}" \
			--disable-usage-stats \
			--dashboard-host="${RAY_DASHBOARD_HOST}" \
			--dashboard-port="${RAY_DASHBOARD_PORT}"
	else
		# worker 节点 (spec §5.1): --block 让前台阻塞直到 head 关闭
		ray start \
			--node-ip-address "${node_ip}" \
			--address "${head_ip}:${port}" \
			--num-gpus "${ray_gpus}" \
			--block
	fi

	# 等待启动
	local retries=30
	while [ $retries -gt 0 ]; do
		if is_ray_running; then
			log_info "Ray cluster started successfully"

			# F9: worker 模式额外验证 GCS 可达 (pgrep 只看 ray:: 进程存在,
			# 进程能起但 GCS 连不通时仍会被认为成功 -> 后续无限 wait)
			if [ "${RANK:-0}" != "0" ]; then
				if ! ray status --address "${head_ip}:${port}" >/dev/null 2>&1; then
					log_error "Worker cannot reach GCS at ${head_ip}:${port}"
					return 1
				fi
				log_info "Worker GCS reachable at ${head_ip}:${port}"
			fi
			return 0
		fi
		sleep 1
		retries=$((retries - 1))
	done

	log_error "Failed to start Ray cluster"
	return 1
}
```

- [ ] **Step 2.4: 跑测试，确认通过**

```bash
bash tests/ray_start_dryrun.sh
```

期望：`ALL PASS`，含 3 个 `PASS: ...` 行。

- [ ] **Step 2.5: 单机回归**

```bash
DRY_RUN=true bash rltrain.sh --train-mode rl -n test -m /tmp -d /tmp/data.jsonl 2>&1 | grep -E "ray start" | head -3
```

期望：输出含 `ray start --head --node-ip-address 127.0.0.1 ...`。

- [ ] **Step 2.6: Commit**

```bash
git add lib/ray.sh tests/ray_start_dryrun.sh
git commit -m "feat(ray): RANK-based head/worker branch in ray_start

- Head (RANK=0): ray start --head with --node-ip-address/--port/--num-gpus
- Worker (RANK>=1): ray start --address=HEAD_IP:PORT --block (foreground
  until head stops Ray)
- F9: worker mode probes ray status after pgrep, since pgrep can give
  false positive when GCS is unreachable
- F5 note: --num-gpus is per-node sum (actor+rollout); users with actor
  only on head must export RAY_NUM_GPUS=\$ROLLOUT_NUM_GPUS on workers
- Tests: tests/ray_start_dryrun.sh covers single/multi-head/multi-worker"
```

---

## Task 3: `rltrain.sh:main()` worker 早返

**Files:**
- Modify: `rltrain.sh:130-131`（在 `ray_start` 之后插入早返段）
- Test: `tests/worker_early_return.sh`（新建）

**Interfaces:**
- Consumes: `$RANK`（Task 1 已填）、`ray_start`（Task 2 已实现）
- Produces: head 路径行为不变；worker 路径在 `ray_start` 返回后立即 `wait` 然后 `exit 0`

### Steps

- [ ] **Step 3.1: 写失败的 worker 早返测试**

创建 `tests/worker_early_return.sh`：

```bash
#!/bin/bash
# tests/worker_early_return.sh
# 验证 worker (RANK=1) 在 ray_start 之后进入 wait 而不是 build_runtime_env_json
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# shim 掉 ray: head 模式模拟成功 (sleep 0.5 后退出), worker 模式前台阻塞
shim_dir="$(mktemp -d)"
cat > "${shim_dir}/ray" <<'EOF'
#!/bin/bash
# 简化: head 命令不出现 --address, 直接退出 0; 出现 --address (worker) 则 trap + sleep
if echo "$*" | grep -q -- "--address"; then
    # worker 模式: trap SIGTERM 退出
    trap 'exit 0' SIGTERM SIGINT
    while true; do sleep 1; done
else
    # head 模式: 立即成功
    exit 0
fi
EOF
chmod +x "${shim_dir}/ray"
# shim 掉 python3: 不需要
cat > "${shim_dir}/python3" <<'EOF'
#!/bin/bash
exit 0
EOF
chmod +x "${shim_dir}/python3"

export PATH="${shim_dir}:${PATH}"
export LOG_FILE="/tmp/worker_test_$$.log"

# 跑 main, 给 worker 路径一个有限的"等待 - 关闭"窗口:
# 1) 启动 rltrain.sh with RANK=1, 后台运行
# 2) 等 2 秒让 ray_start 调通
# 3) 杀掉进程, 期望 rltrain.sh 在 ray --block 收到 SIGTERM 时立即退出 0
(
    export RANK=1 NNODES=2 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10
    export RAY_PORT=6379 ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
    export DRY_RUN=false
    export TRAIN_MODE=rl
    # 不要真的调 ray_submit_job; 我们在 main 里更早早返
    timeout 5 bash "${REPO_ROOT}/rltrain.sh" --train-mode rl -n wk_test -m /tmp -d /tmp/data.jsonl >/tmp/worker_out_$$.log 2>&1 &
    pid=$!
    sleep 2
    # 发 SIGTERM, 模拟 head 关闭 -> ray --block 退出 -> wait 返回 -> main exit 0
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rc=$?
    # 期望在 0..5 之间 (timeout 杀掉也算正常, 因为 SIGTERM 后清理可能要时间)
    [ "${rc}" -le 5 ] || { echo "FAIL: worker 退出码 ${rc} 异常"; cat /tmp/worker_out_$$.log; exit 1; }
    # 日志里应有 "Worker node (RANK=1); blocking until head stops Ray."
    grep -q "Worker node" /tmp/worker_out_$$.log \
        || { echo "FAIL: 未找到 worker 早返日志"; cat /tmp/worker_out_$$.log; exit 1; }
    # 日志不应出现 "Submitting Ray job" (worker 路径不该到这步)
    if grep -q "Submitting Ray job" /tmp/worker_out_$$.log; then
        echo "FAIL: worker 不应执行 ray_submit_job"
        cat /tmp/worker_out_$$.log
        exit 1
    fi
    echo "PASS: worker 早返到 wait, 不执行 job 提交"
)

rm -rf "${shim_dir}"
echo "ALL PASS"
```

- [ ] **Step 3.2: 跑测试，确认失败**

```bash
chmod +x tests/worker_early_return.sh
bash tests/worker_early_return.sh
```

期望：FAIL，当前 `main` 不区分 RANK，会跑到 `ray_submit_job` 或 `build_runtime_env_json`。

- [ ] **Step 3.3: 修改 `rltrain.sh:main()`**

定位 `rltrain.sh:130-131`：

```bash
	ray_start || { log_error "Ray 启动失败"; exit 1; }
	build_runtime_env_json
```

替换为：

```bash
	ray_start || { log_error "Ray 启动失败"; exit 1; }

	# ── Worker 早返 (spec §5.3) ────────────────────────────────
	# head 端通过 ray stop 关闭 cluster 时, worker 的 ray start --block
	# 会退出 -> wait 返回 -> EXIT trap -> cleanup. cleanup 会再调一次
	# ray stop --force, 因 worker 上已无 ray 进程故无害 (no-op).
	# 信号: SIGINT exit 130, SIGTERM exit 143 (Ray 2.5+ graceful).
	if [ "${RANK:-0}" != "0" ]; then
		log_info "Worker node (RANK=${RANK}); blocking until head stops Ray."
		wait
		exit 0
	fi
	# ── worker 早返结束 ─────────────────────────────────

	build_runtime_env_json
```

- [ ] **Step 3.4: 跑测试，确认通过**

```bash
bash tests/worker_early_return.sh
```

期望：`ALL PASS`。

- [ ] **Step 3.5: 单机回归（head 路径不变）**

```bash
DRY_RUN=true bash rltrain.sh --train-mode rl -n regression -m /tmp -d /tmp/data.jsonl 2>&1 | tail -10
```

期望：打印 `ray job submit --address="http://127.0.0.1:8265" ...`，无 `Worker node` 日志。

- [ ] **Step 3.6: Commit**

```bash
git add rltrain.sh tests/worker_early_return.sh
git commit -m "feat(rltrain): worker early-return after ray_start

- Worker (RANK>=1) waits after ray_start, then exits 0
- Head path unchanged
- Cleanup hook still runs via EXIT trap; ray stop --force on worker
  is no-op (no ray processes to stop)
- Tests: tests/worker_early_return.sh verifies worker blocks and
  doesn't execute ray_submit_job"
```

---

## Task 4: `lib/base/env.sh:setup_env()` 多机 NCCL 警告

**Files:**
- Modify: `lib/base/env.sh:6-31`（`setup_env` 函数体）
- Test: `tests/env_warn_dryrun.sh`（新建）

**Interfaces:**
- Consumes: `$NNODES`（Task 1 已填）、`$NCCL_SOCKET_IFNAME`、`$NCCL_IB_HCA`（用户可能传入）
- Produces: NNODES>1 + NCCL_SOCKET_IFNAME 与 NCCL_IB_HCA 都未设时输出 warn，不注入 eth0 默认

### Steps

- [ ] **Step 4.1: 写失败的 dry-run 测试**

创建 `tests/env_warn_dryrun.sh`：

```bash
#!/bin/bash
# tests/env_warn_dryrun.sh
# 验证 setup_env 在多机 + 未设 NCCL_SOCKET_IFNAME/IB_HCA 时 warn,
# 且不再注入 eth0 默认
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
LOG_FILE="/tmp/env_warn_$$.log"
export LOG_FILE

# Case 1: 单机 NCCL_SOCKET_IFNAME 应保持空 (不注入任何默认)
out=$(bash -c "
    export LOG_FILE=/tmp/env_single_\$\$.log
    unset NNODES NCCL_SOCKET_IFNAME NCCL_IB_HCA
    source '${REPO_ROOT}/lib/base/env.sh' 2>&1
    echo \"NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-<unset>}\"
")
echo "$out" | grep -q "NCCL_SOCKET_IFNAME=<unset>" \
    || { echo "FAIL: 单机 NCCL_SOCKET_IFNAME 应保持 unset, 实际 $out"; exit 1; }
echo "PASS: 单机 NCCL_SOCKET_IFNAME 不注入"

# Case 2: 多机 + 用户传入 NCCL_SOCKET_IFNAME=ib0 -> 保持 ib0
out=$(bash -c "
    export LOG_FILE=/tmp/env_user_ib0_\$\$.log
    export NNODES=2 NCCL_SOCKET_IFNAME=ib0
    source '${REPO_ROOT}/lib/base/env.sh' 2>&1
    echo \"NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-<unset>}\"
")
echo "$out" | grep -q "NCCL_SOCKET_IFNAME=ib0" \
    || { echo "FAIL: 用户传入 ib0 应被保留, 实际 $out"; exit 1; }
echo "PASS: 多机 + 用户传入 ib0 保留"

# Case 3: 多机 + 未设 -> 应 warn + NCCL_SOCKET_IFNAME 保持空 (不注入 eth0)
out=$(bash -c "
    export LOG_FILE=/tmp/env_auto_ib_\$\$.log
    export NNODES=2
    unset NCCL_SOCKET_IFNAME NCCL_IB_HCA
    source '${REPO_ROOT}/lib/base/env.sh' 2>&1
    echo \"NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-<unset>}\"
")
echo "$out" | grep -q "NCCL_SOCKET_IFNAME=<unset>" \
    || { echo "FAIL: 多机未设时不应注入 eth0, 实际 $out"; exit 1; }
echo "$out" | grep -q "NCCL will auto-select" \
    || { echo "FAIL: 多机未设时应有 auto-select warn, 实际 $out"; exit 1; }
echo "PASS: 多机未设时 warn + 不注入 eth0"

# Case 4: 多机 + NCCL_IB_HCA 已设 -> 不 warn (用户已在 IB 上明确)
out=$(bash -c "
    export LOG_FILE=/tmp/env_ib_only_\$\$.log
    export NNODES=2 NCCL_IB_HCA=mlx5_0
    unset NCCL_SOCKET_IFNAME
    source '${REPO_ROOT}/lib/base/env.sh' 2>&1
")
if echo "$out" | grep -q "NCCL will auto-select"; then
    echo "FAIL: 已设 NCCL_IB_HCA 时不应 warn auto-select, 实际 $out"
    exit 1
fi
echo "PASS: 多机 + NCCL_IB_HCA 已设 -> 不 warn"

echo "ALL PASS"
```

- [ ] **Step 4.2: 跑测试，确认失败**

```bash
chmod +x tests/env_warn_dryrun.sh
bash tests/env_warn_dryrun.sh
```

期望：case 3 失败（当前 `env.sh` 在多机未设时注入 `eth0`）。

- [ ] **Step 4.3: 修改 `lib/base/env.sh:setup_env()`**

定位 `lib/base/env.sh:7`。把：

```bash
	export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
```

替换为：

```bash
	export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}

	# F4 (spec §5.4): 多机 + 既未设 NCCL_SOCKET_IFNAME 也未设 NCCL_IB_HCA 时,
	# 不要默认 eth0 (IB / RoCE / 云上会错). 改为 warn + 不注入, 让 NCCL 自己选.
	if [ "${NNODES:-1}" -gt 1 ] && [ -z "${NCCL_SOCKET_IFNAME:-}" ] && [ -z "${NCCL_IB_HCA:-}" ]; then
		log_warn "NNODES>1 but neither NCCL_SOCKET_IFNAME nor NCCL_IB_HCA set."
		log_warn "NCCL will auto-select interface; if it picks the wrong one, set explicitly (e.g. ib0 / eth0 / ens5)."
	fi
```

- [ ] **Step 4.4: 跑测试，确认通过**

```bash
bash tests/env_warn_dryrun.sh
```

期望：`ALL PASS`，含 4 个 `PASS: ...` 行。

- [ ] **Step 4.5: 单机回归**

```bash
unset NCCL_SOCKET_IFNAME NCCL_IB_HCA NNODES
bash -c "source lib/base/env.sh && echo NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-<unset>}"
```

期望：`NCCL_SOCKET_IFNAME=<unset>`（保持空，不注入）。

- [ ] **Step 4.6: Commit**

```bash
git add lib/base/env.sh tests/env_warn_dryrun.sh
git commit -m "feat(env): multi-node NCCL auto-select warning

- F4: drop eth0 hard default; IB/RoCE/cloud setups get auto-select +
  explicit warn instead
- Single-node unchanged (NCCL_SOCKET_IFNAME stays unset)
- Tests: tests/env_warn_dryrun.sh covers 4 cases (single / user-set
  ib0 / multi-auto / multi-IB-only)"
```

---

## Task 5: `run.sh` 多机用法注释块

**Files:**
- Modify: `run.sh:66-67`（在 `SGLANG_HOST_IP=127.0.0.1` 之后插入注释段）

**Interfaces:**
- Consumes: 无（仅注释）
- Produces: `run.sh` 顶部多一段 12 行注释，说明多机 env 变量

### Steps

- [ ] **Step 5.1: 修改 `run.sh`**

定位 `run.sh:66-67`（`export SGLANG_HOST_IP=127.0.0.1` 段）。在该行之后插入：

```bash
# ── 多机部署 (optional) ───────────────────────────────────────
# Head 节点:
#   RANK=0 RAY_NODE_IP=<head_ip> RAY_HEAD_IP=<head_ip> NNODES=N ./run.sh ...
# Worker 节点 (RANK=1..N-1):
#   RANK=<i> RAY_NODE_IP=<worker_ip> RAY_HEAD_IP=<head_ip> NNODES=N ./run.sh ...
# 必传:
#   RAY_NODE_IP   本机真实网卡 IP (不要用 127.0.0.1)
#   RAY_HEAD_IP   head 节点 IP (所有节点必须一致)
#   NCCL_SOCKET_IFNAME  本机通信网卡名 (多机时建议显式设, 否则 NCCL 自动选)
# 可选:
#   NCCL_IB_HCA   IB HCA 名 (如 mlx5_0)
#   NCCL_DEBUG=INFO  打开 NCCL 调试日志
# ──────────────────────────────────────────────────────────────
```

- [ ] **Step 5.2: 单机回归**

```bash
DRY_RUN=true bash run.sh --train-mode rl -n test -m /tmp -d /tmp/data.jsonl 2>&1 | grep -E "ray start|ray job submit" | head -3
```

期望：行为不变（`ray start --head --node-ip-address 127.0.0.1 ...`）。

- [ ] **Step 5.3: dry-run 多机用例**

```bash
DRY_RUN=true RANK=0 NNODES=2 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10 \
    bash rltrain.sh --train-mode rl -n multihead -m /tmp -d /tmp/data.jsonl 2>&1 \
    | grep -E "ray start --head" | head -1

DRY_RUN=true RANK=1 NNODES=2 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 \
    bash rltrain.sh --train-mode rl -n multiworker -m /tmp -d /tmp/data.jsonl 2>&1 \
    | grep -E "ray start --address" | head -1
```

期望：head 输出 `--node-ip-address 10.0.1.10`；worker 输出 `--address 10.0.1.10:6379 --node-ip-address 10.0.2.20 --block`。

- [ ] **Step 5.4: Commit**

```bash
git add run.sh
git commit -m "docs(run): multi-node usage comment block

- Documents RANK / RAY_NODE_IP / RAY_HEAD_IP / NNODES / NCCL_*
- No logic change; comment-only per spec §5.5"
```

---

## Task 6: 集成回归与多机一致性断言 dry-run

**Files:**
- Test: `tests/multinode_integration.sh`（新建）

**Interfaces:**
- Consumes: Tasks 1-5 的所有改动
- Produces: 5 个 dry-run case 的最终回归证据

### Steps

- [ ] **Step 6.1: 写集成测试**

创建 `tests/multinode_integration.sh`：

```bash
#!/bin/bash
# tests/multinode_integration.sh
# 5 case: 单机 / 多机 head / 多机 worker / MASTER_ADDR 不一致 / RANK 未设单机 fallback
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

run_dry() {
    # 把 ray / python 都 shim 成 echo, 避免真启
    local shim
    shim="$(mktemp -d)"
    cat > "${shim}/ray" <<'EOF'
#!/bin/bash
echo "RAY_CALL: $*"
EOF
    cat > "${shim}/python3" <<'EOF'
#!/bin/bash
exit 0
EOF
    chmod +x "${shim}/ray" "${shim}/python3"

    local rc=0
    PATH="${shim}:${PATH}" DRY_RUN=true \
        bash "${REPO_ROOT}/rltrain.sh" "$@" >/tmp/integration_$$.log 2>&1 || rc=$?
    cat /tmp/integration_$$.log
    rm -rf "${shim}"
    return "${rc}"
}

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

# Case 1: 单机 fallback (RANK 未设)
echo "── Case 1: 单机 fallback ──"
out=$(run_dry --train-mode rl -n c1 -m /tmp -d /tmp/data.jsonl) || true
echo "$out" | grep -q "ray start --head"            || fail "Case 1 缺 --head"
echo "$out" | grep -q "127.0.0.1"                     || fail "Case 1 应绑定 127.0.0.1"
if echo "$out" | grep -q "MASTER_ADDR.*RAY_HEAD_IP"; then
    fail "Case 1 不应触发不一致断言"
fi
pass "Case 1"

# Case 2: 多机 head (RANK=0, NNODES=2, 一致)
echo "── Case 2: 多机 head ──"
export RANK=0 NNODES=2 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10
out=$(run_dry --train-mode rl -n c2 -m /tmp -d /tmp/data.jsonl) || true
unset RANK NNODES RAY_NODE_IP RAY_HEAD_IP
echo "$out" | grep -q "ray start --head"             || fail "Case 2 缺 --head"
echo "$out" | grep -q "10.0.1.10"                     || fail "Case 2 应绑定 head IP"
pass "Case 2"

# Case 3: 多机 worker (RANK=1, NNODES=2, 连接到 head)
echo "── Case 3: 多机 worker ──"
export RANK=1 NNODES=2 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10
out=$(run_dry --train-mode rl -n c3 -m /tmp -d /tmp/data.jsonl) || true
unset RANK NNODES RAY_NODE_IP RAY_HEAD_IP
echo "$out" | grep -q "ray start --address=10.0.1.10:6379"  || fail "Case 3 应连 head:6379"
echo "$out" | grep -q "10.0.2.20"                            || fail "Case 3 应绑定 worker IP"
echo "$out" | grep -q "\-\-block"                            || fail "Case 3 应带 --block"
pass "Case 3"

# Case 4: MASTER_ADDR 不一致 (NNODES=2 但 MASTER_ADDR 错)
echo "── Case 4: MASTER_ADDR 不一致 ──"
export NNODES=2 RAY_HEAD_IP=10.0.1.10 MASTER_ADDR=10.0.0.99 RANK=0
out=$(run_dry --train-mode rl -n c4 -m /tmp -d /tmp/data.jsonl) || true
unset NNODES RAY_HEAD_IP MASTER_ADDR RANK
echo "$out" | grep -q "MASTER_ADDR (10.0.0.99) != RAY_HEAD_IP (10.0.1.10)" \
    || fail "Case 4 应触发不一致断言"
pass "Case 4"

# Case 5: 多机 head dry-run 路径不输出 Worker node 日志
echo "── Case 5: 多机 head dry-run 不含 Worker 日志 ──"
export RANK=0 NNODES=2 RAY_HEAD_IP=10.0.1.10 RAY_NODE_IP=10.0.1.10
out=$(run_dry --train-mode rl -n c5 -m /tmp -d /tmp/data.jsonl) || true
unset RANK NNODES RAY_HEAD_IP RAY_NODE_IP
echo "$out" | grep -q "ray job submit"               || fail "Case 5 head 应提交 job"
if echo "$out" | grep -q "Worker node"; then
    fail "Case 5 head 不应输出 Worker node 日志"
fi
pass "Case 5"

echo "ALL PASS"
```

- [ ] **Step 6.2: 跑测试**

```bash
chmod +x tests/multinode_integration.sh
bash tests/multinode_integration.sh
```

期望：5 个 case 全部 `PASS`，结尾 `ALL PASS`。

- [ ] **Step 6.3: 全量回归（单机真实路径）**

```bash
DRY_RUN=true bash run.sh --train-mode rl -n full -m /tmp -d /tmp/data.jsonl 2>&1 \
    | grep -E "ray start|ray job submit" | head -5
```

期望：与改动前完全一致：`ray start --head --node-ip-address 127.0.0.1 ...` + `ray job submit --address="http://127.0.0.1:8265" ...`。

- [ ] **Step 6.4: Commit**

```bash
git add tests/multinode_integration.sh
git commit -m "test(multinode): 5-case integration dry-run regression

- Case 1: single-node fallback (RANK unset)
- Case 2: multi-node head (RANK=0, NNODES=2, RAY_HEAD_IP set)
- Case 3: multi-node worker (RANK=1, --address, --block)
- Case 4: MASTER_ADDR != RAY_HEAD_IP triggers assertion
- Case 5: head dry-run path doesn't emit Worker node log"
```

---

## Self-Review Checklist

| Spec 章节 / 要求 | 落点 |
|---|---|
| §1.3 缺陷 1: ray_start 写死 --head | Task 2 |
| §1.3 缺陷 2: RANK 未读 + worker 无早返 | Task 2 + Task 3 |
| §1.3 缺陷 3: NCCL_SOCKET_IFNAME 默认不安全 | Task 4 |
| §1.3 缺陷 4: 注释提到多机但代码未实现 | Task 2 |
| §5.1 ray_start RANK 分支 | Task 2 |
| §5.2 apply_config_defaults RANK/NNODES/MASTER_ADDR 一致性断言 | Task 1 |
| §5.3 worker 早返 | Task 3 |
| §5.4 setup_env 多机 NCCL 处理 | Task 4 |
| §5.5 run.sh 多机注释 | Task 5 |
| §6 错误处理: worker 连不上 head 30s 超时 + 断言 + SIGTERM | Task 1 (断言) + Task 2 (GCS 探测) + Task 3 (信号行为注) |
| §7.1 dry-run 单元测试矩阵 | Task 1 + Task 2 + Task 6 |
| §7.2 集成测试 (两台测试机) | 不在自动化范围；由用户执行（spec §7.2 本就是手测） |
| §7.3 单机回归 | Task 2.5 / 3.5 / 4.5 / 5.2 / 6.3 各自覆盖 |
| §8 向后兼容矩阵 | 全部 task 都验证 DRY_RUN=true 单机行为不变 |
| F1 (RANK 兜底 warn) | Task 1.3 |
| F2 (worker wait 退出链 + Ray 2.5+ graceful) | Task 3.3 注释 |
| F3 (worker 端口拓扑) | Task 5 注释覆盖 |
| F4 (NCCL auto-select 替代 eth0) | Task 4 |
| F5 (per-node GPU 简化假设) | Task 2.3 注释 |
| F6 (3 个新增 dry-run case) | Task 6 case 4 + 5 |
| F7 (MILES_SCRIPT_EXTERNAL_RAY 说明) | 不在本计划范围（spec 注释已修正，实现由用户后续） |
| F8 (MASTER_ADDR 默认值描述) | spec 已修；Task 1 测试覆盖 |
| F9 (worker ray status GCS 探测) | Task 2.3 |
| F10 (KubeRay env 名) | spec 已修；不影响实施 |
| F11 (30s 硬超时窗口说明) | spec 已修；不影响实施 |