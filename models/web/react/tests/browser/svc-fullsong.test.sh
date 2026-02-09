#!/bin/bash
# Browser Automation Test: SVC Full Song Mode
# Tests: Toggle, upload, convert, download via VevoSing page

set -e

VNC_URL="http://192.168.1.64:16080/vnc.html"
APP_URL="https://aphion.giggahost.com"
DISPLAY=:99
SCREENSHOT_DIR="/tmp/svc-fullsong-screenshots"

# Test audio files (must exist before running)
CONTENT_AUDIO="/tmp/test_content_song.wav"
REFERENCE_AUDIO="/tmp/test_reference_voice.wav"

mkdir -p "$SCREENSHOT_DIR"

echo "=========================================="
echo "Browser Test: SVC Full Song Mode"
echo "=========================================="
echo "URL: $APP_URL"
echo "VNC: $VNC_URL"
echo ""

# Verify test files exist
if [ ! -f "$CONTENT_AUDIO" ] || [ ! -f "$REFERENCE_AUDIO" ]; then
    echo "ERROR: Test audio files missing!"
    echo "  Expected: $CONTENT_AUDIO"
    echo "  Expected: $REFERENCE_AUDIO"
    exit 1
fi

# Clean up any existing browser
pkill -f "chromium.*aphion" 2>/dev/null || true
sleep 1

# Test 1: Navigate to VevoSing page
echo "[Test 1] Navigate to VevoSing SVC page"
DISPLAY=$DISPLAY chromium-browser --no-sandbox --disable-gpu --start-maximized \
    "$APP_URL/svc/vevosing" &>/dev/null &
BROWSER_PID=$!
sleep 5

# Take initial screenshot
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/01_page_loaded.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/01_page_loaded.png"

# Test 2: Focus browser window
echo "[Test 2] Focus browser window"
DISPLAY=$DISPLAY xdotool search --name "aphion" windowactivate 2>/dev/null || \
    DISPLAY=$DISPLAY xdotool search --name "Amphion" windowactivate 2>/dev/null || \
    DISPLAY=$DISPLAY xdotool search --name "VevoSing" windowactivate 2>/dev/null || true
sleep 1

# Test 3: Scroll down to find Full Song Mode card
echo "[Test 3] Scroll down to Full Song Mode section"
DISPLAY=$DISPLAY xdotool key --clearmodifiers Page_Down
sleep 1
DISPLAY=$DISPLAY xdotool key --clearmodifiers Page_Down
sleep 1
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/02_scrolled_down.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/02_scrolled_down.png"

# Test 4: Find and click Full Song Mode toggle
# The toggle is typically a switch/checkbox element - we'll click in the area
echo "[Test 4] Click Full Song Mode toggle"
# Search for text "Full Song Mode" on screen using xdotool
# Typically around center-right of the page
DISPLAY=$DISPLAY xdotool mousemove 960 500 click 1
sleep 1
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/03_toggle_clicked.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/03_toggle_clicked.png"

# Test 5: Scroll back up to the top for file uploads
echo "[Test 5] Scroll back to top"
DISPLAY=$DISPLAY xdotool key --clearmodifiers Home
sleep 1

# Test 6: Upload content audio
echo "[Test 6] Upload content audio file"
# Find the first file input for content audio
# Use keyboard shortcut to open file dialog or click the upload area
DISPLAY=$DISPLAY xdotool mousemove 480 400 click 1
sleep 2
# Type the file path in the file dialog
DISPLAY=$DISPLAY xdotool type --clearmodifiers "$CONTENT_AUDIO"
sleep 1
DISPLAY=$DISPLAY xdotool key --clearmodifiers Return
sleep 2
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/04_content_uploaded.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/04_content_uploaded.png"

# Test 7: Upload reference audio
echo "[Test 7] Upload reference audio file"
DISPLAY=$DISPLAY xdotool mousemove 480 600 click 1
sleep 2
DISPLAY=$DISPLAY xdotool type --clearmodifiers "$REFERENCE_AUDIO"
sleep 1
DISPLAY=$DISPLAY xdotool key --clearmodifiers Return
sleep 2
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/05_reference_uploaded.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/05_reference_uploaded.png"

# Test 8: Click Convert button
echo "[Test 8] Click Convert Singing Voice button"
DISPLAY=$DISPLAY xdotool key --clearmodifiers Page_Down
sleep 1
# Convert button is typically at the bottom of the form
DISPLAY=$DISPLAY xdotool mousemove 960 700 click 1
sleep 2
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/06_convert_clicked.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/06_convert_clicked.png"

# Test 9: Wait for conversion (up to 5 minutes)
echo "[Test 9] Waiting for conversion to complete (max 5 min)..."
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ... $ELAPSED seconds elapsed"
    # Take progress screenshot every 30 seconds
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/07_progress_${ELAPSED}s.png" 2>/dev/null || true
    fi
done

# Test 10: Final screenshot
echo "[Test 10] Capture final state"
DISPLAY=$DISPLAY import -window root "$SCREENSHOT_DIR/08_final_state.png" 2>/dev/null || true
echo "  Screenshot: $SCREENSHOT_DIR/08_final_state.png"

# Cleanup
echo ""
echo "Cleaning up..."
kill $BROWSER_PID 2>/dev/null || true
pkill -f "chromium.*aphion" 2>/dev/null || true

echo ""
echo "=========================================="
echo "SVC Full Song Mode Browser Test: COMPLETE"
echo "=========================================="
echo ""
echo "Screenshots saved to: $SCREENSHOT_DIR"
echo "View VNC at: $VNC_URL"
ls -la "$SCREENSHOT_DIR/"
