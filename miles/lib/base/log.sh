#!/bin/bash
#
# 日志基础设施
#

source "${BASH_SOURCE[0]%/*}/colors.sh"

_log() {
	local level="$1"; local message="$2"; local color=""; local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
	case "$level" in
	DEBUG) color="$COLOR_BLUE" ;;
	INFO)  color="$COLOR_GREEN" ;;
	WARN)  color="$COLOR_YELLOW" ;;
	ERROR) color="$COLOR_RED" ;;
	*)     color="$COLOR_NC" ;;
	esac
	echo -e "${color}[${timestamp}] [${level}] ${message}${COLOR_NC}"
	[ -n "$LOG_FILE" ] && echo "[${timestamp}] [${level}] ${message}" >>"$LOG_FILE"
}

log_debug() { [ "${LOG_LEVEL:-INFO}" = "DEBUG" ] && _log "DEBUG" "$*" || true; }
log_info()  { _log "INFO" "$*"; }
log_warn()  { _log "WARN" "$*"; }
log_error() { _log "ERROR" "$*"; }
