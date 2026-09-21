#!/usr/bin/env bash
# =============================================================================
# ATSM Boot — Single executable that boots everything and shows the dashboard
# =============================================================================
#
# Usage:
#   ./atsm_boot.sh            Boot all services + open dashboard in browser
#   ./atsm_boot.sh --status   Show service status without opening browser
#   ./atsm_boot.sh --no-browser  Boot all services but don't open browser
#
# What it does:
#   1. Starts ATSM Gateway (REST API on port 8766)
#   2. Starts Proactive Agent (background daemon)
#   3. Starts ATSM Stream (SSE on port 8765)
#   4. Runs self-evolution cycle (learn + update)
#   5. Waits for all services to be ready
#   6. Opens the unified ATSM Dashboard in the default browser
#
# Dashboard: http://localhost:8766/ui
# =============================================================================

set -euo pipefail

# --- Paths ---
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SKILL_DIR/scripts"
DATA_DIR="$SKILL_DIR/data"
UI_DIR="$SKILL_DIR/ui"

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
NC='\033[0m'

# --- Helpers ---
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_header() { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }

check_pid() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

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

wait_for_port() {
    local port="$1"
    local timeout="${2:-15}"
    local attempts=0
    while [[ $attempts -lt $((timeout * 2)) ]]; do
        if check_port "$port"; then
            return 0
        fi
        sleep 0.5
        attempts=$((attempts + 1))
    done
    return 1
}

# --- Service Start Functions ---

start_gateway() {
    log_info "Starting ATSM Gateway..."
    if check_port "$GATEWAY_PORT"; then
        log_warn "Gateway port $GATEWAY_PORT already in use"
        return 0
    fi
    if check_pid "$GATEWAY_PID"; then
        log_warn "Gateway already running (pid $(cat "$GATEWAY_PID"))"
        return 0
    fi

    nohup python3 "$GATEWAY" start > /dev/null 2>&1 &

    if wait_for_port "$GATEWAY_PORT" 15; then
        log_ok "ATSM Gateway started on port $GATEWAY_PORT"
        return 0
    else
        log_err "Gateway failed to start within 15 seconds"
        return 1
    fi
}

start_proactive() {
    log_info "Starting Proactive Agent..."
    if check_pid "$PROACTIVE_PID"; then
        log_warn "Proactive Agent already running (pid $(cat "$PROACTIVE_PID"))"
        return 0
    fi

    nohup python3 "$PROACTIVE" start > /dev/null 2>&1 &
    sleep 1

    if check_pid "$PROACTIVE_PID"; then
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
        log_warn "Stream port $STREAM_PORT already in use"
        return 0
    fi
    if check_pid "$STREAM_PID"; then
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
    log_info "Running self-evolution cycle..."
    local learn_output
    learn_output=$(python3 "$LEARN" analyze 2>/dev/null | head -5) || true
    if [[ -n "$learn_output" ]]; then
        log_ok "Self-learning analysis complete"
    else
        log_warn "Self-learning produced no output (may be first run)"
    fi
}

# --- Dashboard ---

open_dashboard() {
    log_header "ATSM Dashboard"
    echo ""

    # Verify gateway is ready
    if ! check_port "$GATEWAY_PORT"; then
        log_err "Gateway not running on port $GATEWAY_PORT"
        log_info "Cannot open dashboard — gateway is down"
        return 1
    fi

    # Verify dashboard HTML exists
    if [[ ! -f "$UI_DIR/atsm-dashboard.html" ]]; then
        log_err "Dashboard HTML not found: $UI_DIR/atsm-dashboard.html"
        return 1
    fi

    log_info "Opening dashboard: $DASHBOARD_URL"

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

    # Brave
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

# --- Status ---

show_status() {
    log_header "ATSM Service Status"
    echo ""

    printf "  %-20s" "Gateway:"
    if check_pid "$GATEWAY_PID"; then
        echo -e "${GREEN}running${NC} (pid $(cat "$GATEWAY_PID"))"
    elif check_port "$GATEWAY_PORT"; then
        echo -e "${GREEN}running${NC} (port $GATEWAY_PORT)"
    else
        echo -e "${RED}stopped${NC}"
    fi

    printf "  %-20s" "Proactive Agent:"
    if check_pid "$PROACTIVE_PID"; then
        echo -e "${GREEN}running${NC} (pid $(cat "$PROACTIVE_PID"))"
    else
        echo -e "${RED}stopped${NC}"
    fi

    printf "  %-20s" "Stream:"
    if check_pid "$STREAM_PID"; then
        echo -e "${GREEN}running${NC} (pid $(cat "$STREAM_PID"))"
    elif check_port "$STREAM_PORT"; then
        echo -e "${GREEN}running${NC} (port $STREAM_PORT)"
    else
        echo -e "${RED}stopped${NC}"
    fi

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
    log_header "Dashboard"
    echo ""
    echo -e "  URL: ${CYAN}$DASHBOARD_URL${NC}"
    if [[ -f "$UI_DIR/atsm-dashboard.html" ]]; then
        echo -e "  HTML: ${GREEN}EXISTS${NC}"
    else
        echo -e "  HTML: ${RED}MISSING${NC}"
    fi

    echo ""
    log_header "Data"
    echo ""
    local db_records=0
    if [[ -f "$DATA_DIR/atsm_db.jsonl" ]]; then
        db_records=$(wc -l < "$DATA_DIR/atsm_db.jsonl")
    fi
    echo -e "  DB records:     $db_records"
    echo -e "  Log directory:  $DATA_DIR"
}

# --- Main Boot Sequence ---

do_boot() {
    local no_browser=false
    [[ "${1:-}" == "--no-browser" ]] && no_browser=true

    log_header "ATSM Boot — Starting All Services"
    echo ""

    mkdir -p "$DATA_DIR"

    # Start all services
    start_gateway
    start_proactive
    start_stream
    start_self_evolving

    echo ""
    log_ok "ATSM system started successfully"

    # Show quick status
    echo ""
    echo -e "  Dashboard: ${CYAN}$DASHBOARD_URL${NC}"
    echo -e "  API:       ${CYAN}http://localhost:${GATEWAY_PORT}/api${NC}"

    # Open dashboard
    if [[ "$no_browser" == false ]]; then
        echo ""
        open_dashboard
    fi

    echo ""
    log_header "ATSM Boot Complete"
    echo ""
}

# --- Entry Point ---

main() {
    local cmd="${1:-boot}"

    case "$cmd" in
        ""|boot)
            do_boot
            ;;
        --no-browser)
            do_boot --no-browser
            ;;
        --status|-s)
            show_status
            ;;
        --help|-h)
            echo "ATSM Boot — Single executable to boot all services + dashboard"
            echo ""
            echo "Usage: $(basename "$0") [command]"
            echo ""
            echo "Commands:"
            echo "  (none)        Boot all services and open dashboard (default)"
            echo "  --no-browser  Boot all services without opening browser"
            echo "  --status      Show service status"
            echo "  --help        Show this help"
            echo ""
            echo "Dashboard: $DASHBOARD_URL"
            ;;
        *)
            log_err "Unknown command: $cmd"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"
