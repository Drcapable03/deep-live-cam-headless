#!/bin/bash
# ============================================================================
# Deep-Live-Cam Headless Streaming — Deploy & Run Script
# ============================================================================
# Starts the headless pipeline inside a tmux session on a cloud GPU box.
# Idempotent and handles graceful shutdown.
#
# Usage:
#   ./scripts/deploy_and_run.sh [--help] [--stop] [--status] [--run-command "..."]
#
# Default run command (production-ready):
#   python run_headless.py -s face.jpg --input-source video.mp4 \
#     --stream-output rtmp://localhost/live/main \
#     --frame-processor face_swapper face_enhancer_gpen256 \
#     --mouth-mask --eyes-mask --eyebrows-mask \
#     --stream-encoder h264_nvenc --execution-provider cuda
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/deep-live-cam-headless"
VENV_DIR="${WORKSPACE}/.venv"
LOGS_DIR="${WORKSPACE}/logs"
DLC_LOG="${LOGS_DIR}/dlc.log"
TMUX_SESSION="dlc"
MEDIAMTX_BIN="/usr/local/bin/mediamtx"
NGROK_BIN="/usr/local/bin/ngrok"

# Default production run command
DEFAULT_RUN_CMD=(
    python3 run_headless.py
    -s face.jpg
    --input-source video.mp4
    --stream-output rtmp://localhost/live/main
    --frame-processor face_swapper face_enhancer_gpen256
    --mouth-mask --eyes-mask --eyebrows-mask
    --stream-encoder h264_nvenc
    --execution-provider cuda
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log()  { echo "[DEPLOY] $*"; }
info() { echo "       $*"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --help          Show this help message
  --stop          Stop the running DLC session (graceful shutdown)
  --status        Check if DLC is running and show process info
  --run-command   Custom run command (quoted, e.g., 'python run_headless.py ...')
  --tunnel        Start ngrok TCP tunnel for port 1935 (MediaMTX)
  --no-tunnel     Don't start ngrok tunnel (default)
  --verbose       Enable verbose logging

Examples:
  # Start with defaults:
  $(basename "$0")

  # Start with custom command:
  $(basename "$0") --run-command "python run_headless.py -s face.jpg --input-source webcam --stream-output rtmp://localhost/live/main --execution-provider cuda"

  # Stop running session:
  $(basename "$0") --stop

  # Check status:
  $(basename "$0") --status

  # Start with tunnel:
  $(basename "$0") --tunnel
EOF
}

# Check if DLC session is running
is_running() {
    tmux has-session -t "${TMUX_SESSION}" 2>/dev/null
}

# Check if MediaMTX is listening
mediamtx_running() {
    systemctl is-active --quiet mediamtx 2>/dev/null || \
    ss -tlnp 2>/dev/null | grep -q ':1935' || \
    netstat -tlnp 2>/dev/null | grep -q ':1935'
}

# Verify setup prerequisites
verify_setup() {
    local missing=0

    if ! cmd_exists python3; then
        log "ERROR: python3 not found — run scripts/setup_cloud.sh first"
        missing=1
    fi

    if [ ! -d "${VENV_DIR}" ]; then
        log "ERROR: Virtualenv not found at ${VENV_DIR} — run scripts/setup_cloud.sh first"
        missing=1
    fi

    if [ ! -f "${PROJECT_DIR}/run_headless.py" ]; then
        log "ERROR: Project not found at ${PROJECT_DIR} — run scripts/setup_cloud.sh first"
        missing=1
    fi

    if [ ! -f "${PROJECT_DIR}/models/inswapper_128.onnx" ]; then
        log "WARNING: inswapper_128.onnx not found in ${PROJECT_DIR}/models/"
        info "  Expected at: ${PROJECT_DIR}/models/inswapper_128.onnx"
    fi

    if [ ! -f "${PROJECT_DIR}/face.jpg" ]; then
        log "WARNING: Source face image not found at ${PROJECT_DIR}/face.jpg"
        info "  Place a face image there or use -s <path>"
    fi

    return ${missing}
}

# Start MediaMTX if not already running
ensure_mediamtx() {
    if mediamtx_running; then
        log "MediaMTX is already running"
        return 0
    fi

    log "Starting MediaMTX..."
    if cmd_exists mediamtx; then
        systemctl start mediamtx 2>/dev/null && return 0
        ${MEDIAMTX_BIN} --config /etc/mediamtx.yml &
    elif [ -x "${MEDIAMTX_BIN}" ]; then
        ${MEDIAMTX_BIN} --config /etc/mediamtx.yml &
    else
        log "ERROR: MediaMTX not found. Install it or run scripts/setup_cloud.sh"
        return 1
    fi

    sleep 2
    if mediamtx_running; then
        log "MediaMTX started successfully"
    else
        log "WARNING: MediaMTX may not have started correctly"
    fi
}

# Start ngrok tunnel if requested
start_ngrok_tunnel() {
    if ! cmd_exists ngrok; then
        log "ngrok not found — skipping tunnel"
        return 0
    fi

    log "Starting ngrok TCP tunnel for port 1935..."
    ngrok tcp 1935 >> "${LOGS_DIR}/ngrok.log" 2>&1 &
    NGROK_PID=$!
    echo "${NGROK_PID}" > "${LOGS_DIR}/ngrok.pid"
    sleep 3

    # Print the public URL
    if curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -q '"public_url"'; then
        LOCAL_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4)
        log "ngrok tunnel active: ${LOCAL_URL}"
        info "Use this RTMP URL in OBS: rtmp://${LOCAL_URL//tcp:\/\//}"
    else
        log "WARNING: Could not get ngrok tunnel URL — check ${LOGS_DIR}/ngrok.log"
    fi
}

# Stop DLC session gracefully
stop_session() {
    if ! is_running; then
        log "No DLC session running"
        return 0
    fi

    log "Stopping DLC session (sending SIGINT to tmux pane)..."
    # Send Ctrl+C to the main pane to trigger graceful shutdown
    tmux send-keys -t "${TMUX_SESSION}" C-c

    # Wait up to 10 seconds for the session to stop
    local waited=0
    while is_running && [ ${waited} -lt 10 ]; do
        sleep 1
        waited=$((waited + 1))
    done

    if ! is_running; then
        log "DLC session stopped gracefully"
    else
        log "Session still running after 10s — sending kill signal..."
        tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
        log "Session killed"
    fi

    # Stop ngrok tunnel if running
    if [ -f "${LOGS_DIR}/ngrok.pid" ]; then
        NGROK_PID=$(cat "${LOGS_DIR}/ngrok.pid")
        if kill -0 "${NGROK_PID}" 2>/dev/null; then
            kill "${NGROK_PID}" 2>/dev/null || true
            rm -f "${LOGS_DIR}/ngrok.pid"
            log "ngrok tunnel stopped"
        fi
    fi
}

# Show status of running components
show_status() {
    echo ""
    echo "=========================================="
    echo " Deep-Live-Cam Deployment Status"
    echo "=========================================="
    echo ""

    # DLC session
    if is_running; then
        echo "DLC Session:    RUNNING (tmux: ${TMUX_SESSION})"
        # Show last few lines of log
        if [ -f "${DLC_LOG}" ]; then
            echo "Last log lines:"
            tail -5 "${DLC_LOG}" | sed 's/^/  /'
        fi
    else
        echo "DLC Session:    NOT RUNNING"
    fi
    echo ""

    # MediaMTX
    if mediamtx_running; then
        echo "MediaMTX:       RUNNING (port 1935)"
    else
        echo "MediaMTX:       NOT RUNNING"
    fi
    echo ""

    # ngrok
    if [ -f "${LOGS_DIR}/ngrok.pid" ] && kill -0 "$(cat "${LOGS_DIR}/ngrok.pid" 2>/dev/null)" 2>/dev/null; then
        echo "ngrok Tunnel:   ACTIVE"
        if curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -q '"public_url"'; then
            PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4)
            echo "  Public URL:   ${PUBLIC_URL}"
        fi
    else
        echo "ngrok Tunnel:   NOT ACTIVE"
    fi
    echo ""

    # GPU
    if cmd_exists nvidia-smi; then
        echo "GPU:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>/dev/null | head -5 | sed 's/^/  /'
    else
        echo "GPU:            Not detected (nvidia-smi not available)"
    fi
    echo ""

    # Disk space
    echo "Disk Space:"
    df -h "${WORKSPACE}" 2>/dev/null | tail -1 | sed 's/^/  /'
    echo ""

    # Logs directory
    echo "Logs:"
    if [ -d "${LOGS_DIR}" ]; then
        ls -lh "${LOGS_DIR}"/*.log 2>/dev/null | sed 's/^/  /' || echo "  No log files yet"
    else
        echo "  Logs directory not found"
    fi
    echo ""
    echo "=========================================="
}

# Attach to existing tmux session
attach_session() {
    if tmux list-sessions 2>/dev/null | grep -q "${TMUX_SESSION}"; then
        log "Attaching to existing session '${TMUX_SESSION}'..."
        tmux attach -t "${TMUX_SESSION}" || {
            # If already attached elsewhere, detach first
            tmux detach-client -t "${TMUX_SESSION}" 2>/dev/null || true
            tmux attach -t "${TMUX_SESSION}"
        }
    else
        log "No session to attach to"
    fi
}

# Build the actual run command from args
build_run_cmd() {
    local cmd_str="$@"
    if [ -z "${cmd_str}" ]; then
        # Use default command
        set -- "${DEFAULT_RUN_CMD[@]}"
    else
        eval "set -- ${cmd_str}"
    fi
    echo "$@"
}

# Run the DLC pipeline inside tmux
run_pipeline() {
    local run_cmd="$1"
    local start_tunnel="${2:-false}"

    # Ensure logs directory exists
    mkdir -p "${LOGS_DIR}"

    # Ensure MediaMTX
    ensure_mediamtx

    # Stop any existing session first
    if is_running; then
        log "Existing session found — stopping it first..."
        tmux send-keys -t "${TMUX_SESSION}" C-c 2>/dev/null || true
        sleep 2
        tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
        sleep 1
    fi

    log "Starting DLC pipeline in tmux session '${TMUX_SESSION}'..."
    log "Run command: ${run_cmd}"

    # Create and run inside tmux session
    tmux new-session -d -s "${TMUX_SESSION}" -x 200 -y 50

    # Set working directory and activate venv, then run the pipeline
    tmux send-keys -t "${TMUX_SESSION}" \
        "cd ${PROJECT_DIR} && source ${VENV_DIR}/bin/activate && exec ${run_cmd} 2>&1 | tee ${DLC_LOG}" ENTER

    # Give it a moment to start
    sleep 2

    # Check if the process is still alive
    local pane_pid
    pane_pid=$(tmux display-message -t "${TMUX_SESSION}" -p '#{pane_pid}' 2>/dev/null || echo "")

    if [ -n "${pane_pid}" ] && kill -0 "${pane_pid}" 2>/dev/null; then
        log "Pipeline started successfully in tmux session '${TMUX_SESSION}'"
        info "Log file: ${DLC_LOG}"
        info "Attach: tmux attach -t ${TMUX_SESSION}"
        info "Stop:   $(basename "$0") --stop"
    else
        log "WARNING: Pipeline may have failed to start"
        info "Check log: ${DLC_LOG}"
        [ -f "${DLC_LOG}" ] && tail -10 "${DLC_LOG}" | sed 's/^/  /' || true
    fi

    # Start ngrok tunnel if requested
    if [ "${start_tunnel}" = "true" ]; then
        start_ngrok_tunnel
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local action="run"
    local run_cmd=""
    local do_tunnel=false
    local do_attach=false

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --help|-h)
                usage
                exit 0
                ;;
            --stop)
                action="stop"
                shift
                ;;
            --status)
                action="status"
                shift
                ;;
            --run-command)
                action="run"
                run_cmd="$2"
                shift 2
                ;;
            --tunnel)
                do_tunnel=true
                shift
                ;;
            --no-tunnel)
                do_tunnel=false
                shift
                ;;
            --attach)
                do_attach=true
                shift
                ;;
            --verbose)
                set -x
                shift
                ;;
            *)
                log "Unknown argument: $1"
                usage >&2
                exit 1
                ;;
        esac
    done

    case "${action}" in
        stop)
            stop_session
            ;;
        status)
            show_status
            ;;
        run)
            # Verify prerequisites
            if ! verify_setup; then
                log "Setup verification failed. Run scripts/setup_cloud.sh first."
                exit 1
            fi

            # Build command
            local full_cmd
            full_cmd=$(build_run_cmd "${run_cmd}")

            # Handle attach mode (just attach to existing session)
            if ${do_attach}; then
                attach_session
                exit $?
            fi

            # Run the pipeline
            run_pipeline "${full_cmd}" "${do_tunnel}"
            ;;
    esac
}

main "$@"
