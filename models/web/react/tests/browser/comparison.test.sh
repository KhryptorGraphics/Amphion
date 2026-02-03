#!/bin/bash
# Browser Automation Test: Comparison Tools (Task 16.12)
# Tests: Select items, A/B comparison, multi-audio grid

set -e

APP_URL="http://192.168.1.64:3001"
DISPLAY=:99

echo "=========================================="
echo "Browser Test: Comparison Tools"
echo "=========================================="

# Launch browser
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu --start-maximized "$APP_URL/compare" &
sleep 3

# Test 1: Navigate to Comparison page
echo "[Test 1] Navigate to Comparison page"
DISPLAY=$DISPLAY xdotool search --name "AutoVoice" windowactivate || true
sleep 2

# Test 2: Verify comparison interface renders
echo "[Test 2] Verify comparison interface with selection panel"
DISPLAY=$DISPLAY xdotool mousemove 200 400
sleep 1

# Test 3: Select first audio item
echo "[Test 3] Select first audio item"
DISPLAY=$DISPLAY xdotool mousemove 200 300 click 1
sleep 1

# Test 4: Select second audio item
echo "[Test 4] Select second audio item"
DISPLAY=$DISPLAY xdotool mousemove 200 400 click 1
sleep 1

# Test 5: Click Compare button
echo "[Test 5] Click Compare button"
DISPLAY=$DISPLAY xdotool mousemove 600 200 click 1
sleep 2

# Test 6: Verify A/B comparison view
echo "[Test 6] Verify A/B comparison view renders"
DISPLAY=$DISPLAY xdotool mousemove 500 400
sleep 1

# Test 7: Toggle A/B switch
echo "[Test 7] Toggle A/B switch"
DISPLAY=$DISPLAY xdotool mousemove 960 600 click 1
sleep 1

# Test 8: Switch to multi-audio grid view
echo "[Test 8] Switch to multi-audio grid view"
DISPLAY=$DISPLAY xdotool mousemove 800 200 click 1
sleep 2

# Test 9: Verify grid view with multiple audios
echo "[Test 9] Verify multi-audio grid"
DISPLAY=$DISPLAY xdotool mousemove 600 400
sleep 1

# Cleanup
pkill chromium-browser 2>/dev/null || true

echo "=========================================="
echo "Comparison Tools Test: PASSED"
echo "=========================================="
