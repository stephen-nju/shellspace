#!/bin/bash
# tests/config_defaults_dryrun.sh
# 验证 apply_config_defaults 在不同 RANK/NNODES 组合下的行为
set -eo pipefail
# 不开 -u: apply_config_defaults 引用了若干可选变量 (DATASET, REF_LOAD 等)

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
    apply_config_defaults
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
    apply_config_defaults
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
    apply_config_defaults
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
    apply_config_defaults
    [ "${RANK}" = "0" ] || fail "NNODES>1 + RANK 未设, 默认应为 0"
    pass "F1 兜底: NNODES>1 + RANK 未设 -> 默认 0"
) || exit 1

echo "ALL PASS"