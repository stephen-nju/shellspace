#!/bin/bash
#
# 统一清理 + hook 注册机制
#

source "${BASH_SOURCE[0]%/*}/log.sh"

CLEANUP_IN_PROGRESS=0
ALL_PIDS=()
RAY_JOB_ID=""
CLEANUP_HOOKS=""

cleanup_register_pid() { ALL_PIDS+=("$1"); }
cleanup_register_ray_job() { RAY_JOB_ID="$1"; }
cleanup_add_hook() { CLEANUP_HOOKS="${CLEANUP_HOOKS:+${CLEANUP_HOOKS} } $1"; }

cleanup() {
	if [ $CLEANUP_IN_PROGRESS -eq 1 ]; then
		log_warn "Cleanup already in progress, waiting..."; sleep 5; return
	fi
	CLEANUP_IN_PROGRESS=1
	log_warn "Cleaning up background services..."

	if [ -n "$RAY_JOB_ID" ]; then
		log_info "Stopping Ray job: $RAY_JOB_ID"
		ray job stop "$RAY_JOB_ID" --address="http://${RAY_HEAD_IP:-127.0.0.1}:8265" 2>/dev/null || true
	fi

	for hook in ${CLEANUP_HOOKS:-}; do
		if declare -F "$hook" >/dev/null 2>&1; then
			log_info "Running cleanup hook: $hook"; "$hook" 2>/dev/null || true
		fi
	done

	for pid in "${ALL_PIDS[@]}"; do
		kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
	done
	sleep 2
	for pid in "${ALL_PIDS[@]}"; do
		kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
	done

	log_info "Stopping Ray cluster..."; ray stop --force 2>/dev/null || true
	log_info "Cleaning up residual processes..."
	pkill -9 -f "[t]rain_async.py" 2>/dev/null || true
	pkill -9 -f "[s]glang" 2>/dev/null || true
	pkill -9 -f "[r]ay::" 2>/dev/null || true
	pkill -9 -f "[p]ython.*miles.*train" 2>/dev/null || true
	log_info "Cleanup completed."
}

setup_signal_handlers() { trap cleanup SIGINT SIGTERM SIGHUP EXIT; }
