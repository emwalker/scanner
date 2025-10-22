#!/bin/bash
# diagnose_cpu.sh - Automated CPU performance diagnostic script
# Usage: ./diagnose_cpu.sh <PID>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PID>" >&2
    exit 1
fi

PID=$1

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Error: Process $PID does not exist or is not accessible" >&2
    exit 1
fi

echo "=== CPU Diagnostics for PID $PID ==="
echo

echo "--- Process Overview ---"
ps aux | head -1
ps aux | grep "^\S*\s*$PID\s"
echo

echo "--- Thread CPU Usage (top 15) ---"
ps -L -o pid,lwp,comm,%cpu,state,wchan -p "$PID" | head -1
ps -L -o pid,lwp,comm,%cpu,state,wchan -p "$PID" | tail -n +2 | sort -k4 -rn | head -15
echo

echo "--- Thread States Summary ---"
echo "Total threads: $(ps -L -p "$PID" | wc -l)"
echo "Running (R): $(ps -L -o state -p "$PID" | grep -c '^R' || echo 0)"
echo "Sleeping (S): $(ps -L -o state -p "$PID" | grep -c '^S' || echo 0)"
echo "Waiting (D): $(ps -L -o state -p "$PID" | grep -c '^D' || echo 0)"
echo "Zombie (Z): $(ps -L -o state -p "$PID" | grep -c '^Z' || echo 0)"
echo

echo "--- GDB Backtraces (all threads, top 10 frames) ---"
if command -v gdb &> /dev/null; then
    gdb -batch -p "$PID" -ex "thread apply all bt 10" 2>&1 | grep -E "(Thread|#[0-9]|in |at )" || echo "No backtrace data available"
else
    echo "GDB not available - install gdb for detailed backtraces"
fi
echo

echo "=== Diagnostic Tips ==="
echo "- Threads with high %CPU and WCHAN='-' are busy-waiting"
echo "- Look for repeated patterns in GDB backtraces"
echo "- Running (R) threads should be actively doing work"
echo "- Sleeping (S) threads are properly blocked"
