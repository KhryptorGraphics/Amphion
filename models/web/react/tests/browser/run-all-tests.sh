#!/bin/bash
# Master Browser Test Runner for Phase 16
# Runs all browser automation tests sequentially

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_URL="http://192.168.1.64:3001"

echo "================================================================================"
echo "                    AMPHION STUDIO - PHASE 16 BROWSER TESTS"
echo "================================================================================"
echo ""
echo "Target URL: $APP_URL"
echo "VNC Display: :99 (view at http://192.168.1.64:16080/vnc.html)"
echo ""
echo "================================================================================"

# Check if Xvfb is running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1920x1080x24 &
    sleep 2
fi

# Array of test scripts
TESTS=(
    "training.test.sh:Training Workflow (Task 16.8)"
    "dataset.test.sh:Dataset Management (Task 16.9)"
    "batch.test.sh:Batch Processing (Task 16.10)"
    "history.test.sh:History Features (Task 16.11)"
    "comparison.test.sh:Comparison Tools (Task 16.12)"
    "export-import.test.sh:Export/Import (Task 16.13)"
    "error-handling.test.sh:Error Handling (Task 16.14)"
)

PASSED=0
FAILED=0

for test_entry in "${TESTS[@]}"; do
    IFS=: read -r script description <<< "$test_entry"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Running: $description"
    echo "--------------------------------------------------------------------------------"

    if [ -f "$SCRIPT_DIR/$script" ]; then
        if bash "$SCRIPT_DIR/$script"; then
            ((PASSED++))
            echo "✓ PASSED: $description"
        else
            ((FAILED++))
            echo "✗ FAILED: $description"
        fi
    else
        echo "✗ NOT FOUND: $script"
        ((FAILED++))
    fi
done

echo ""
echo "================================================================================"
echo "                              TEST SUMMARY"
echo "================================================================================"
echo "Passed: $PASSED/${#TESTS[@]}"
echo "Failed: $FAILED/${#TESTS[@]}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL TESTS PASSED"
    echo "================================================================================"
    exit 0
else
    echo "✗ SOME TESTS FAILED"
    echo "================================================================================"
    exit 1
fi
