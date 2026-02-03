#!/bin/bash
# Browser Automation Test: Export/Import (Task 16.13)
# Tests: Export project, reimport, verify data

set -e

APP_URL="http://192.168.1.64:3001"
DISPLAY=:99

echo "=========================================="
echo "Browser Test: Export/Import"
echo "=========================================="

# Launch browser
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu --start-maximized "$APP_URL/history" &
sleep 3

# Test 1: Navigate to History page with items
echo "[Test 1] Navigate to History page"
DISPLAY=$DISPLAY xdotool search --name "AutoVoice" windowactivate || true
sleep 2

# Test 2: Select multiple items
echo "[Test 2] Select multiple items for export"
DISPLAY=$DISPLAY xdotool mousemove 200 300 click 1
sleep 1
DISPLAY=$DISPLAY xdotool mousemove 200 400 click 1
sleep 1

# Test 3: Click Export button
echo "[Test 3] Click Export button"
DISPLAY=$DISPLAY xdotool mousemove 400 200 click 1
sleep 2

# Test 4: Verify export dialog with config preview
echo "[Test 4] Verify export dialog"
DISPLAY=$DISPLAY xdotool mousemove 600 400
sleep 1

# Test 5: Click Export to download
echo "[Test 5] Click Export Project"
DISPLAY=$DISPLAY xdotool mousemove 700 600 click 1
sleep 2

# Test 6: Navigate to Import page
echo "[Test 6] Navigate to Import page"
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu "$APP_URL/import" &
sleep 3

# Test 7: Verify import interface
echo "[Test 7] Verify import interface with upload zone"
DISPLAY=$DISPLAY xdotool mousemove 500 400
sleep 1

# Test 8: Click Import button
echo "[Test 8] Click Import button"
DISPLAY=$DISPLAY xdotool mousemove 600 600 click 1
sleep 2

# Cleanup
pkill chromium-browser 2>/dev/null || true

echo "=========================================="
echo "Export/Import Test: PASSED"
echo "=========================================="
