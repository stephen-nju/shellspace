#!/bin/bash
# tests/ray_start_dryrun.sh
# 验证 ray_start 在不同 RANK 组合下生成的 ray 命令
set -eo pipefail
# 不开 -u: ray.sh 内部引用若干可选变量

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# shim: 替换 ray CLI 为 echo shim, 替换 pkill 为 true shim
shim_dir="$(mktemp -d)"
cat > "${shim_dir}/ray" <<'EOF'
#!/bin/bash
echo "RAY_CALL: $*" >&2
# status 模拟失败以便 worker case 走到 GCS unreachable 分支
if echo "$*" | grep -q "status"; then
    exit 1
fi
EOF
cat > "${shim_dir}/pkill" <<'EOF'
#!/bin/bash
exit 0
EOF
# shim pgrep: 总是返回 0 (模拟 ray:: 进程存在) 让 is_ray_running 通过
cat > "${shim_dir}/pgrep" <<'EOF'
#!/bin/bash
exit 0
EOF
chmod +x "${shim_dir}/ray" "${shim_dir}/pkill" "${shim_dir}/pgrep"
export PATH="${shim_dir}:${PATH}"

extract_ray_args() {
    # 抓整条 RAY_CALL 行 (单行), 提取 --flag value 对
    local prefix="$1"
    local line
    line=$(grep "RAY_CALL" "${prefix}.err" 2>/dev/null | tail -1)
    # 去掉 echo 本身的换行处理: tr 把空格换成换行, 再把 --xxx 后的非 -- 行合并
    echo "$line" \
        | sed 's/.*RAY_CALL: //' \
        | awk '
            {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^--/) {
                        if ($(i+1) !~ /^--/ && $(i+1) != "") {
                            print $i " " $(i+1)
                            i++
                        } else {
                            print $i
                        }
                    }
                }
            }
        '
}

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

# 把 log 文件路径传给 ray_start; 它内部用 ${LOG_FILE}
export LOG_FILE_PREFIX="/tmp/ray_start_dryrun_$$"

# source lib/ray.sh 让 ray_start 函数可调用
# shellcheck disable=SC1090
source "${REPO_ROOT}/lib/ray.sh"

# helper: 在 subshell 里 source 完整依赖
setup_shell() {
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/lib/base/colors.sh"
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/lib/base/log.sh"
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/lib/ray.sh"
}

# 1) 单机 head
(
    unset RANK
    export RAY_NODE_IP=127.0.0.1 RAY_HEAD_IP=127.0.0.1 RAY_PORT=6379
    export ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
    setup_shell
    prefix="${LOG_FILE_PREFIX}_single"
    set +e
    LOG_FILE="${prefix}.log" ray_start >/dev/null 2>"${prefix}.err" || true
    set -e
    out=$(extract_ray_args "${prefix}")
    echo "$out" | grep -q "^--head$"        || fail "单机应包含 --head, 实际: $out"
    echo "$out" | grep -q "^--node-ip-address 127.0.0.1$" || fail "单机 node-ip 应为 127.0.0.1"
    echo "$out" | grep -q "^--num-gpus 8$"  || fail "单机 num-gpus 应为 8 (4+4)"
    pass "单机 head dry-run"
)

# 2) 多机 head
(
    export RANK=0 NNODES=2 RAY_NODE_IP=10.0.1.10 RAY_HEAD_IP=10.0.1.10 RAY_PORT=6379
    export ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
    setup_shell
    prefix="${LOG_FILE_PREFIX}_head"
    set +e
    LOG_FILE="${prefix}.log" ray_start >/dev/null 2>"${prefix}.err" || true
    set -e
    out=$(extract_ray_args "${prefix}")
    echo "$out" | grep -q "^--head$"          || fail "多机 head 应包含 --head"
    echo "$out" | grep -q "^--node-ip-address 10.0.1.10$" || fail "head node-ip 应为 10.0.1.10"
    pass "多机 head dry-run"
)

# 3) 多机 worker (GCS 不可达 -> 期望 return 1)
export RANK=1 NNODES=2 RAY_NODE_IP=10.0.2.20 RAY_HEAD_IP=10.0.1.10 RAY_PORT=6379
export ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=4
setup_shell
prefix="${LOG_FILE_PREFIX}_worker"
# set +e 让 ray_start 的 return 1 不终止测试
set +e
LOG_FILE="${prefix}.log" ray_start >/dev/null 2>"${prefix}.err"
rc=$?
set -e
[ "${rc}" = "1" ] || fail "worker 在 GCS 不可达时返回 1, 实际 ${rc}"
out=$(extract_ray_args "${prefix}")
echo "$out" | grep -q "^--node-ip-address 10.0.2.20$" || fail "worker node-ip 应为 10.0.2.20"
echo "$out" | grep -q "^--address 10.0.1.10:6379$"   || fail "worker 应连 10.0.1.10:6379"
echo "$out" | grep -q "^--block$"                    || fail "worker 应带 --block"
pass "多机 worker dry-run"

echo "ALL PASS"