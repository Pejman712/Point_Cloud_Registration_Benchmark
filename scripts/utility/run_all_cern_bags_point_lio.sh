#!/usr/bin/env bash
set -Eeuo pipefail

# Edit these if needed.
DATASET_ROOT="${DATASET_ROOT:-$HOME/point_cloud_registeration_benchmark/dataset/CERN/unitree_unilidar_L1}"
OUT_DIR="${OUT_DIR:-$HOME/point_cloud_registeration_benchmark/results/tum_unitree_unilidar_L1}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"

TOPIC="${TOPIC:-/aft_mapped_to_init}"
RECORDER="${RECORDER:-$HOME/point_cloud_registeration_benchmark/scripts/utility/record_aft_mapped_to_tum.py}"

LAUNCH_CMD="${LAUNCH_CMD:-ros2 launch point_lio mapping_unilidar_l1.launch.py}"
LAUNCH_ARGS="${LAUNCH_ARGS:-}"

TYPE="${TYPE:-auto}"
DELIMITER="${DELIMITER:-space}"

PLAY_RATE="${PLAY_RATE:-1.0}"
BAG_PLAY_ARGS="${BAG_PLAY_ARGS:-}"

STARTUP_WAIT="${STARTUP_WAIT:-10}"
TOPIC_WAIT="${TOPIC_WAIT:-20}"
POST_BAG_WAIT="${POST_BAG_WAIT:-3}"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

LAUNCH_PID=""
REC_PID=""
BAG_PID=""

stop_group() {
    local pid="${1:-}"

    if [[ -z "$pid" ]]; then
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
        sleep 2
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup_current() {
    stop_group "$BAG_PID"
    stop_group "$REC_PID"
    stop_group "$LAUNCH_PID"

    BAG_PID=""
    REC_PID=""
    LAUNCH_PID=""
}

trap 'echo "Interrupted. Cleaning up."; cleanup_current; exit 130' INT TERM

mapfile -t BAGS < <(find "$DATASET_ROOT" -name metadata.yaml -printf '%h\n' | sort)

if [[ "${#BAGS[@]}" -eq 0 ]]; then
    echo "No ROS 2 bags found under: $DATASET_ROOT"
    exit 1
fi

echo "Found ${#BAGS[@]} bags."
echo "Output directory: $OUT_DIR"
echo

for BAG_DIR in "${BAGS[@]}"; do
    BAG_NAME="$(basename "$BAG_DIR")"
    OUT_FILE="$OUT_DIR/${BAG_NAME}.tum"

    LAUNCH_LOG="$LOG_DIR/${BAG_NAME}_point_lio.log"
    REC_LOG="$LOG_DIR/${BAG_NAME}_recorder.log"
    PLAY_LOG="$LOG_DIR/${BAG_NAME}_bag_play.log"

    echo "============================================================"
    echo "Bag: $BAG_NAME"
    echo "Path: $BAG_DIR"
    echo "Output: $OUT_FILE"
    echo "============================================================"

    rm -f "$OUT_FILE"

    echo "Starting Point-LIO..."
    setsid bash -lc "$LAUNCH_CMD $LAUNCH_ARGS" > "$LAUNCH_LOG" 2>&1 &
    LAUNCH_PID="$!"

    sleep "$STARTUP_WAIT"

    echo "Checking whether $TOPIC is visible..."
    TOPIC_FOUND=0

    for ((i = 0; i < TOPIC_WAIT; i++)); do
        if ros2 topic list 2>/dev/null | grep -qx "$TOPIC"; then
            TOPIC_FOUND=1
            break
        fi
        sleep 1
    done

    if [[ "$TOPIC_FOUND" -eq 0 ]]; then
        echo "Warning: $TOPIC was not visible before playback."
        echo "The recorder will still try to auto-detect it while the bag is playing."
    fi

    echo "Starting trajectory recorder..."
    setsid python3 "$RECORDER" \
        --topic "$TOPIC" \
        --type "$TYPE" \
        --output "$OUT_FILE" \
        --delimiter "$DELIMITER" \
        > "$REC_LOG" 2>&1 &
    REC_PID="$!"

    sleep 2

    echo "Playing bag..."
    # shellcheck disable=SC2086
    setsid ros2 bag play "$BAG_DIR" --rate "$PLAY_RATE" $BAG_PLAY_ARGS > "$PLAY_LOG" 2>&1 &
    BAG_PID="$!"

    wait "$BAG_PID" || echo "Warning: ros2 bag play returned non-zero for $BAG_NAME"
    BAG_PID=""

    sleep "$POST_BAG_WAIT"

    echo "Stopping recorder and Point-LIO..."
    stop_group "$REC_PID"
    REC_PID=""

    stop_group "$LAUNCH_PID"
    LAUNCH_PID=""

    if [[ -f "$OUT_FILE" ]]; then
        LINE_COUNT="$(wc -l < "$OUT_FILE")"
    else
        LINE_COUNT=0
    fi

    echo "Finished $BAG_NAME: $LINE_COUNT poses saved."
    echo
done

echo "All bags processed."
echo "TUM files are in: $OUT_DIR"
