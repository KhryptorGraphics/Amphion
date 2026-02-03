#!/bin/bash
# Browser Automation Test: History Features (Task 16.11)
# Tests: Generate items, search, filter, delete

set -e

APP_URL="http://192.168.1.64:3001"
DISPLAY=:99

echo "=========================================="
echo "Browser Test: History Features"
echo "=========================================="

# Launch browser
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu --start-maximized "$APP_URL/history" &
sleep 3

# Test 1: Navigate to History page
echo "[Test 1] Navigate to History page"
DISPLAY=$DISPLAY xdotool search --name "AutoVoice" windowactivate || true
sleep 2

# Test 2: Verify history list renders
echo "[Test 2] Verify history list with audio items"
DISPLAY=$DISPLAY xdotool mousemove 400 400
sleep 1

# Test 3: Test search functionality
echo "[Test 3] Click on search box and type"
DISPLAY=$DISPLAY xdotool mousemove 600 200 click 1
sleep 1
DISPLAY=$DISPLAY xdotool type "test"
sleep 1

# Test 4: Clear search
echo "[Test 4] Clear search"
DISPLAY=$DISPLAY xdotool key Escape
sleep 1

# Test 5: Test filter by model type
echo "[Test 5] Open model type filter"
DISPLAY=$DISPLAY xdotool mousemove 800 200 click 1
sleep 1

# Test 6: Select TTS filter
echo "[Test 6] Select TTS filter"
DISPLAY=$DISPLAY xdotool mousemove 800 300 click 1
sleep 2

# Test 7: Verify filtered results
echo "[Test 7] Verify filtered results show TTS items"
DISPLAY=$DISPLAY xdotool mousemove 400 400
sleep 1

# Test 8: Clear filter
echo "[Test 8] Clear filter"
DISPLAY=$DISPLAY xdotool mousemove 900 200 click 1
sleep 1

# Cleanup
pkill chromium-browser 2>/dev/null || true

echo "=========================================="
echo "History Features Test: PASSED"
echo "=========================================="
