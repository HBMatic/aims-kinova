#!/bin/sh
# macOS only. Run the read-only RealSense check with the access macOS requires:
#   sudo sh scripts/realsense_mac.sh            (from the repo root; needs .venv, see docs/04)
# macOS's camera service (UVCAssistant/VDCAssistant) grabs every UVC camera and respawns within seconds,
# so it is stopped in a loop while the check runs, then allowed to restart. Your Mac's other cameras are
# unavailable for those ~25 s. Previews go to logs/realsense_*.png.
cd "$(dirname "$0")/.." || exit 1
( while :; do killall UVCAssistant VDCAssistant 2>/dev/null; sleep 0.2; done ) &
K=$!
sleep 1
PYTHONUNBUFFERED=1 .venv/bin/python scripts/realsense_check.py logs/realsense 2>&1 | tee logs/realsense_check.log
kill $K
chmod a+r logs/realsense_* 2>/dev/null
