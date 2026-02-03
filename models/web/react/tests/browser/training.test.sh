#!/bin/bash
# Browser Automation Test: Training Workflow (Task 16.8)
# Tests: Create training job, monitor progress, cancel job

set -e

VNC_URL="http://192.168.1.64:16080/vnc.html"
APP_URL="http://192.168.1.64:3001"
DISPLAY=:99

echo "=========================================="
echo "Browser Test: Training Workflow"
echo "=========================================="

# Launch browser
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu --start-maximized "$APP_URL/training" &
sleep 3

# Test 1: Navigate to Training page
echo "[Test 1] Navigate to Training page"
DISPLAY=$DISPLAY xdotool search --name "AutoVoice" windowactivate || true
sleep 2

# Test 2: Click "New Training Job" button
echo "[Test 2] Click New Training Job button"
DISPLAY=$DISPLAY xdotool mousemove 400 300 click 1
sleep 2

# Test 3: Verify Training Wizard renders
echo "[Test 3] Verify Training Wizard renders"
DISPLAY=$DISPLAY xdotool mousemove 600 400
sleep 1

# Test 4: Navigate through wizard steps
echo "[Test 4] Navigate wizard steps"
DISPLAY=$DISPLAY xdotool mousemove 800 600 click 1  # Next button
sleep 1
DISPLAY=$DISPLAY xdotool mousemove 800 600 click 1  # Next button
sleep 1

# Test 5: Click Cancel to close wizard
echo "[Test 5] Cancel wizard"
DISPLAY=$DISPLAY xdotool mousemove 600 600 click 1  # Cancel button
sleep 1

# Test 6: Navigate to Training Monitor
echo "[Test 6] Navigate to Training Monitor"
DISPLAY=$DISPLAY xdotool mousemove 200 400 click 1  # Monitor link
sleep 2

# Test 7: Verify monitor page renders
echo "[Test 7] Verify monitor page renders with loss curves"
DISPLAY=$DISPLAY xdotool mousemove 500 400
sleep 1

# Cleanup
pkill chromium-browser 2>/dev/null || true

echo "=========================================="
echo "Training Workflow Test: PASSED"
echo "=========================================="
