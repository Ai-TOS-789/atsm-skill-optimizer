#!/usr/bin/env bash
# =============================================================================
# ATSM Boot Script — Single entry point for the entire ATSM system
# =============================================================================
#
# Usage:
#   ./atsm.sh start      Start all services (gateway, self-evolving, proactive)
#   ./atsm.sh stop       Stop all services
#   ./atsm.sh restart    Restart all services
#   ./atsm.sh status     Show all service status
#   ./atsm.sh dashboard  Open web dashboard in browser
#   ./atsm.sh health     Full system health check
#   ./atsm.sh logs       Tail all service logs
#
# The ATSM Gateway runs on http://localhost:8766
# The dashboard is at http://localhost:8766/ui
# =============================================================================

set -euo pipefail

# --- Paths ---
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SKILL_DIR/scripts"
DATA_DIR="$SKILL_DIR/data"

# Service scripts
GATEWAY="$SCRIPTS_DIR/atsm_gateway.py"
PROACTIVE="$SCRIPTS_DIR/proactive_agent.py"
STREAM="$SCRIPTS_DIR/atsm_stream.py"
LEARN="$SCRIPTS_DIR/atsm_learn.py"
UPDATE="$SCRIPTS_DIR/atsm_update.py"

# PID files
GATEWAY_PID="$DATA_DIR/atsm_gateway.pid"
PROACTIVE_PID="$DATA_DIR/proactive_agent.pid"
STREAM_PID="$DATA_DIR/atsm_stream.pid"

# Log files
GATEWAY_LOG="$DATA_DIR/atsm_gateway.log"
PROACTIVE_LOG="$DATA_DIR/proactive.log"
STREAM_LOG="$DATA_DIR/atsm_stream.log"

# Ports
GATEWAY_PORT=8766
STREAM_PORT=8765

# Dashboard URL
DASHBOARD_URL="http://localhost:${GATEWAY_PORT}/ui"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Helpers ---
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_header() { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }

# Check if a process is running by PID file
check_pid() {
    local pid_file="$1"
    local service_name="$2"

    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}running${NC} (pid $pid)"
            return 0
        else
            echo -e "${RED}stopped${NC} (stale PID file)"
            return 1
        fi
    else
        echo -e "${RED}stopped${NC}"
        return 1
    fi
}

# Check if a port is in use
check_port() {
    local port="$1"
    if command -v lsof &>/dev/null; then
        lsof -i ":$port" -sTCP:LISTEN -t &>/dev/null
    elif command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":$port "
    else
        return 1
    fi
}

# Check if a process is running by name pattern
check_process() {
    local pattern="$1"
    pgrep -f "$pattern" &>/dev/null
}

# --- Service Management ---

start_gateway() {
    log_info "Starting ATSM Gateway..."
    if check_port "$GATEWAY_PORT"; then
        log_warn "Gateway port $GATEWAY_PORT already in use, skipping"
        return 0
    fi
    if [[ -f "$GATEWAY_PID" ]] && kill -0 "$(cat "$GATEWAY_PID")" 2>/dev/null; then
        log_warn "Gateway already running (pid $(cat "$GATEWAY_PID"))"
        return 0
    fi

    # Start gateway in background
    nohup python3 "$GATEWAY" start > /dev/null 2>&1 &
    local gw_start_pid=$!

    # Wait for it to be ready
    local attempts=0
    while [[ $attempts -lt 30 ]]; do
        if check_port "$GATEWAY_PORT"; then
            log_ok "ATSM Gateway started on port $GATEWAY_PORT"
            return 0
        fi
        sleep 0.3
        attempts=$((attempts + 1))
    done

    log_err "Gateway failed to start within 9 seconds"
    return 1
}

start_proactive() {
    log_info "Starting Proactive Agent..."
    if [[ -f "$PROACTIVE_PID" ]] && kill -0 "$(cat "$PROACTIVE_PID")" 2>/dev/null; then
        log_warn "Proactive Agent already running (pid $(cat "$PROACTIVE_PID"))"
        return 0
    fi

    nohup python3 "$PROACTIVE" start > /dev/null 2>&1 &
    sleep 1

    if [[ -f "$PROACTIVE_PID" ]] && kill -0 "$(cat "$PROACTIVE_PID")" 2>/dev/null; then
        log_ok "Proactive Agent started (pid $(cat "$PROACTIVE_PID"))"
        return 0
    else
        log_warn "Proactive Agent may have started (daemon mode)"
        return 0
    fi
}

start_stream() {
    log_info "Starting ATSM Stream..."
    if check_port "$STREAM_PORT"; then
        log_warn "Stream port $STREAM_PORT already in use, skipping"
        return 0
    fi
    if [[ -f "$STREAM_PID" ]] && kill -0 "$(cat "$STREAM_PID")" 2>/dev/null; then
        log_warn "Stream already running (pid $(cat "$STREAM_PID"))"
        return 0
    fi

    nohup python3 "$STREAM" start > "$STREAM_LOG" 2>&1 &
    sleep 1

    if check_port "$STREAM_PORT"; then
        log_ok "ATSM Stream started on port $STREAM_PORT"
        return 0
    else
        log_warn "Stream may still be starting (check $STREAM_LOG)"
        return 0
    fi
}

start_self_evolving() {
    log_info "Running self-evolution cycle (learn + update)..."
    local learn_output
    learn_output=$(python3 "$LEARN" analyze 2>/dev/null | head -5) || true
    if [[ -n "$learn_output" ]]; then
        log_ok "Self-learning analysis complete"
    else
        log_warn "Self-learning produced no output (may be first run)"
    fi
}

stop_gateway() {
    local killed=false

    # Try PID file first
    if [[ -f "$GATEWAY_PID" ]]; then
        local pid
        pid=$(cat "$GATEWAY_PID" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            # Wait for process to die
            local attempts=0
            while kill -0 "$pid" 2>/dev/null && [[ $attempts -lt 10 ]]; do
                sleep 0.3
                attempts=$((attempts + 1))
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            log_ok "ATSM Gateway stopped (pid $pid)"
            killed=true
        else
            log_warn "Gateway not running (stale PID file)"
        fi
        rm -f "$GATEWAY_PID"
    fi

    # Try to find and kill by port
    if [[ "$killed" == false ]] && command -v lsof &>/dev/null; then
        local pids
        pids=$(lsof -i ":$GATEWAY_PORT" -sTCP:LISTEN -t 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill $pids 2>/dev/null || true
            sleep 0.5
            log_ok "ATSM Gateway stopped (via port $GATEWAY_PORT)"
            killed=true
        fi
    fi

    # Try process pattern
    if [[ "$killed" == false ]]; then
        local pids
        pids=$(pgrep -f "atsm_gateway.py" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill $pids 2>/dev/null || true
            sleep 0.5
            log_ok "ATSM Gateway stopped (via process match)"
            killed=true
        fi
    fi

    if [[ "$killed" == false ]]; then
        log_warn "Gateway not running"
    fi
}

stop_proactive() {
    local killed=false

    if [[ -f "$PROACTIVE_PID" ]]; then
        local pid
        pid=$(cat "$PROACTIVE_PID" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 0.5
            log_ok "Proactive Agent stopped (pid $pid)"
            killed=true
        else
            log_warn "Proactive Agent not running (stale PID file)"
        fi
        rm -f "$PROACTIVE_PID"
    fi

    # Try to find by process pattern
    if [[ "$killed" == false ]]; then
        local pids
        pids=$(pgrep -f "proactive_agent.py" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill $pids 2>/dev/null || true
            sleep 0.5
            log_ok "Proactive Agent stopped (via process match)"
            killed=true
        fi
    fi

    if [[ "$killed" == false ]]; then
        log_warn "Proactive Agent not running"
    fi
}

stop_stream() {
    local killed=false

    if [[ -f "$STREAM_PID" ]]; then
        local pid
        pid=$(cat "$STREAM_PID" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            log_ok "ATSM Stream stopped (pid $pid)"
            killed=true
        else
            log_warn "Stream not running (stale PID file)"
        fi
        rm -f "$STREAM_PID"
    fi

    # Try to find by port
    if [[ "$killed" == false ]] && command -v lsof &>/dev/null; then
        local pids
        pids=$(lsof -i ":$STREAM_PORT" -sTCP:LISTEN -t 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill $pids 2>/dev/null || true
            sleep 0.5
            log_ok "ATSM Stream stopped (via port $STREAM_PORT)"
            killed=true
        fi
    fi

    # Try process pattern
    if [[ "$killed" == false ]]; then
        local pids
        pids=$(pgrep -f "atsm_stream.py" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill $pids 2>/dev/null || true
            sleep 0.5
            log_ok "ATSM Stream stopped (via process match)"
            killed=true
        fi
    fi

    if [[ "$killed" == false ]]; then
        log_warn "Stream not running"
    fi
}

# --- Commands ---

cmd_start() {
    log_header "ATSM System Start"
    echo ""

    mkdir -p "$DATA_DIR"

    start_gateway
    start_proactive
    start_stream
    start_self_evolving

    echo ""
    log_ok "ATSM system started successfully"
    echo ""
    echo -e "  Dashboard: ${CYAN}$DASHBOARD_URL${NC}"
    echo -e "  API:       ${CYAN}http://localhost:${GATEWAY_PORT}/api${NC}"
}

cmd_stop() {
    log_header "ATSM System Stop"
    echo ""

    stop_gateway
    stop_proactive
    stop_stream

    echo ""
    log_ok "ATSM system stopped"
}

cmd_restart() {
    log_header "ATSM System Restart"
    echo ""
    cmd_stop
    echo ""
    sleep 1
    cmd_start
}

cmd_status() {
    log_header "ATSM System Status"
    echo ""

    local gw_status
    gw_status=$(check_pid "$GATEWAY_PID" "gateway" 2>&1) || true
    echo -e "  Gateway:        $gw_status"

    local pa_status
    pa_status=$(check_pid "$PROACTIVE_PID" "proactive" 2>&1) || true
    echo -e "  Proactive Agent: $pa_status"

    local st_status
    st_status=$(check_pid "$STREAM_PID" "stream" 2>&1) || true
    echo -e "  Stream:         $st_status"

    echo ""
    log_header "Ports"
    echo ""
    if check_port "$GATEWAY_PORT"; then
        echo -e "  Port $GATEWAY_PORT (Gateway):  ${GREEN}LISTENING${NC}"
    else
        echo -e "  Port $GATEWAY_PORT (Gateway):  ${RED}CLOSED${NC}"
    fi
    if check_port "$STREAM_PORT"; then
        echo -e "  Port $STREAM_PORT (Stream):    ${GREEN}LISTENING${NC}"
    else
        echo -e "  Port $STREAM_PORT (Stream):    ${RED}CLOSED${NC}"
    fi

    echo ""
    log_header "Processes"
    echo ""
    local proc_count
    proc_count=$(pgrep -f "atsm_gateway.py|proactive_agent.py|atsm_stream.py" 2>/dev/null | wc -l)
    if [[ $proc_count -gt 0 ]]; then
        echo -e "  ${GREEN}$proc_count ATSM process(es) running${NC}"
        pgrep -af "atsm_gateway.py|proactive_agent.py|atsm_stream.py" 2>/dev/null | while read -r line; do
            echo -e "    $line"
        done
    else
        echo -e "  ${RED}No ATSM processes found${NC}"
    fi

    echo ""
    log_header "Data"
    echo ""
    local db_records=0
    if [[ -f "$DATA_DIR/atsm_db.jsonl" ]]; then
        db_records=$(wc -l < "$DATA_DIR/atsm_db.jsonl")
    fi
    echo -e "  DB records:     $db_records"
    echo -e "  PID files:      $DATA_DIR/*.pid"
    echo -e "  Log files:      $DATA_DIR/*.log"
}

cmd_dashboard() {
    log_header "ATSM Dashboard"
    echo ""

    # Check if gateway is running
    if ! check_port "$GATEWAY_PORT"; then
        log_warn "Gateway not running on port $GATEWAY_PORT"
        log_info "Starting gateway first..."
        echo ""
        start_gateway
        echo ""
    fi

    log_info "Opening dashboard: $DASHBOARD_URL"

    # Try to open browser
    local opened=false

    # Linux
    if command -v xdg-open &>/dev/null; then
        xdg-open "$DASHBOARD_URL" &>/dev/null && opened=true
    fi

    # macOS
    if [[ "$opened" == false ]] && command -v open &>/dev/null; then
        open "$DASHBOARD_URL" 2>/dev/null && opened=true
    fi

    # WSL
    if [[ "$opened" == false ]] && command -v cmd.exe &>/dev/null; then
        cmd.exe /c start "$DASHBOARD_URL" 2>/dev/null && opened=true
    fi

    # Brave (common on this system)
    if [[ "$opened" == false ]] && command -v brave-browser &>/dev/null; then
        brave-browser "$DASHBOARD_URL" &>/dev/null && opened=true
    fi
    if [[ "$opened" == false ]] && command -v brave-browser-stable &>/dev/null; then
        brave-browser-stable "$DASHBOARD_URL" &>/dev/null && opened=true
    fi

    # Firefox fallback
    if [[ "$opened" == false ]] && command -v firefox &>/dev/null; then
        firefox "$DASHBOARD_URL" &>/dev/null && opened=true
    fi

    if [[ "$opened" == true ]]; then
        log_ok "Dashboard opened in browser"
    else
        log_warn "Could not auto-open browser"
        echo -e "  Please open manually: ${CYAN}$DASHBOARD_URL${NC}"
    fi
}

cmd_health() {
    log_header "ATSM Health Check"
    echo ""

    local healthy=true

    # 1. Gateway health
    echo -e "  ${BOLD}Gateway API${NC}"
    if check_port "$GATEWAY_PORT"; then
        local health_resp
        health_resp=$(curl -s --max-time 5 "http://localhost:${GATEWAY_PORT}/api/health" 2>/dev/null || echo "")
        if [[ -n "$health_resp" ]]; then
            echo -e "    Status: ${GREEN}REACHABLE${NC}"
            echo -e "    Response: $health_resp"
        else
            echo -e "    Status: ${YELLOW}NO RESPONSE${NC}"
            healthy=false
        fi
    else
        echo -e "    Status: ${RED}UNREACHABLE${NC}"
        healthy=false
    fi

    # 2. Gateway status endpoint
    echo ""
    echo -e "  ${BOLD}Gateway Status${NC}"
    if check_port "$GATEWAY_PORT"; then
        local status_resp
        status_resp=$(curl -s --max-time 5 "http://localhost:${GATEWAY_PORT}/api/status" 2>/dev/null || echo "")
        if [[ -n "$status_resp" ]]; then
            echo -e "    ${GREEN}OK${NC}"
            echo "$status_resp" | python3 -m json.tool 2>/dev/null | head -15 | sed 's/^/    /' || echo "    $status_resp"
        else
            echo -e "    ${YELLOW}No status response${NC}"
        fi
    fi

    # 3. Proactive agent
    echo ""
    echo -e "  ${BOLD}Proactive Agent${NC}"
    if [[ -f "$PROACTIVE_PID" ]] && kill -0 "$(cat "$PROACTIVE_PID")" 2>/dev/null; then
        echo -e "    Status: ${GREEN}RUNNING${NC} (pid $(cat "$PROACTIVE_PID"))"
    else
        echo -e "    Status: ${RED}NOT RUNNING${NC}"
        healthy=false
    fi

    # 4. Stream server
    echo ""
    echo -e "  ${BOLD}Stream Server${NC}"
    if check_port "$STREAM_PORT"; then
        echo -e "    Status: ${GREEN}LISTENING${NC} on port $STREAM_PORT"
    else
        echo -e "    Status: ${YELLOW}NOT RUNNING${NC}"
    fi

    # 5. Data integrity
    echo ""
    echo -e "  ${BOLD}Data Integrity${NC}"
    if [[ -f "$DATA_DIR/atsm_db.jsonl" ]]; then
        local db_count
        db_count=$(wc -l < "$DATA_DIR/atsm_db.jsonl")
        echo -e "    DB records: ${GREEN}$db_count${NC}"
    else
        echo -e "    DB records: ${YELLOW}0 (no DB file)${NC}"
    fi

    if [[ -f "$DATA_DIR/atsm_priors.json" ]]; then
        echo -e "    Priors file: ${GREEN}EXISTS${NC}"
    else
        echo -e "    Priors file: ${YELLOW}MISSING${NC}"
    fi

    # 6. Disk space
    echo ""
    echo -e "  ${BOLD}Disk Space${NC}"
    local data_size
    data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "    Data directory: $data_size"

    # 7. Python dependencies
    echo ""
    echo -e "  ${BOLD}Dependencies${NC}"
    if python3 -c "import json, math, http.server" 2>/dev/null; then
        echo -e "    Python stdlib: ${GREEN}OK${NC}"
    else
        echo -e "    Python stdlib: ${RED}MISSING MODULES${NC}"
        healthy=false
    fi

    # Summary
    echo ""
    if [[ "$healthy" == true ]]; then
        log_ok "All health checks passed"
    else
        log_warn "Some health checks failed — review output above"
    fi
}

cmd_logs() {
    log_header "ATSM Service Logs"
    echo ""

    local log_files=()
    [[ -f "$GATEWAY_LOG" ]] && log_files+=("$GATEWAY_LOG")
    [[ -f "$PROACTIVE_LOG" ]] && log_files+=("$PROACTIVE_LOG")
    [[ -f "$STREAM_LOG" ]] && log_files+=("$STREAM_LOG")

    if [[ ${#log_files[@]} -eq 0 ]]; then
        log_warn "No log files found in $DATA_DIR"
        return
    fi

    echo -e "  Tailing ${#log_files[@]} log file(s). Press Ctrl+C to stop."
    echo ""

    tail -f "${log_files[@]}" 2>/dev/null &
    local tail_pid=$!

    # Wait for Ctrl+C
    trap 'kill $tail_pid 2>/dev/null; exit 0' INT TERM
    wait $tail_pid 2>/dev/null || true
}

# --- Main ---

show_help() {
    cat <<EOF
ATSM Boot Script — Single entry point for the ATSM system

Usage: $(basename "$0") <command>

Commands:
  start      Start all services (gateway, proactive, stream, self-evolving)
  stop       Stop all services
  restart    Restart all services
  status     Show all service status with colored output
  dashboard  Open the web dashboard in default browser
  health     Run full system health check
  logs       Tail all service logs (Ctrl+C to stop)

Services:
  Gateway         HTTP API + Dashboard on port $GATEWAY_PORT
  Proactive Agent Background daemon for periodic checks
  Stream          SSE real-time event stream on port $STREAM_PORT
  Self-Evolving   Learning analysis and prior updates

Dashboard: $DASHBOARD_URL
EOF
}

main() {
    local cmd="${1:-help}"

    case "$cmd" in
        start)
            cmd_start
            ;;
        stop)
            cmd_stop
            ;;
        restart)
            cmd_restart
            ;;
        status)
            cmd_status
            ;;
        dashboard)
            cmd_dashboard
            ;;
        health)
            cmd_health
            ;;
        logs)
            cmd_logs
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_err "Unknown command: $cmd"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
