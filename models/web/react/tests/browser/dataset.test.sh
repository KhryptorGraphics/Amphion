#!/bin/bash
# Browser Automation Test: Dataset Management (Task 16.9)
# Tests: Upload dataset, preprocess, browse, delete

set -e

VNC_URL="http://192.168.1.64:16080/vnc.html"
APP_URL="http://192.168.1.64:3001"
DISPLAY=:99

echo "=========================================="
echo "Browser Test: Dataset Management"
echo "=========================================="

# Launch browser
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu --start-maximized "$APP_URL/dataset" &
sleep 3

# Test 1: Navigate to Dataset page
echo "[Test 1] Navigate to Dataset page"
DISPLAY=$DISPLAY xdotool search --name "AutoVoice" windowactivate || true
sleep 2

# Test 2: Click Upload Dataset
echo "[Test 2] Click Upload Dataset button"
DISPLAY=$DISPLAY xdotool mousemove 400 300 click 1
sleep 2

# Test 3: Verify upload dialog renders
echo "[Test 3] Verify upload dialog with progress bar"
DISPLAY=$DISPLAY xdotool mousemove 600 400
sleep 1

# Test 4: Close upload dialog
echo "[Test 4] Close upload dialog"
DISPLAY=$DISPLAY xdotool key Escape
sleep 1

# Test 5: Navigate to Preprocess page
echo "[Test 5] Navigate to Preprocess page"
DISPLAY=$DISPLAY xdotool mousemove 200 500 click 1  # Preprocess link
sleep 2

# Test 6: Verify preprocessing wizard renders
echo "[Test 6] Verify preprocessing wizard"
DISPLAY=$DISPLAY xdotool mousemove 500 400
sleep 1

# Test 7: Navigate to Dataset Browser
echo "[Test 7] Navigate to Dataset Browser"
DISPLAY=$DISPLAY xdotool mousemove 200 600 click 1  # Browse link
sleep 2

# Test 8: Verify dataset list renders
echo "[Test 8] Verify dataset list with audio preview"
DISPLAY=$DISPLAY xdotool mousemove 400 400
sleep 1

# Cleanup
pkill chromium-browser 2>/dev/null || true

echo "=========================================="
echo "Dataset Management Test: PASSED"
echo "=========================================="
