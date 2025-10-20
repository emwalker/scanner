# Process Debugging Skill

Use this skill when investigating performance issues like high CPU usage, busy waits, deadlocks, or thread contention in running processes.

## When to Use

- CPU fan spinning / high CPU usage
- Process consuming excessive CPU (e.g., 500%+)
- Suspected busy-wait loops
- Thread contention or deadlocks
- Unresponsive processes

## Tools and Techniques

### 1. Identify the Problem Process

```bash
ps aux | grep <process_name>
```

Look for high CPU percentage (e.g., 542% means ~5-6 cores busy).

### 2. Examine Thread-Level CPU Usage

```bash
top -H -b -n 1 -p <PID> | head -30
```

This shows individual threads with their CPU usage. Key observations:
- Running threads (R state) with high %CPU are busy-waiting
- Sleeping threads (S state) are properly blocked
- Check the WCHAN column: `-` means busy-wait, syscall names mean properly blocked

### 3. Alternative Thread View

```bash
ps -L -o pid,lwp,comm,%cpu,wchan -p <PID> | sort -k4 -rn | head -15
```

Shows threads sorted by CPU with wait channel info. WCHAN values:
- `-` = busy loop (not blocked on any syscall)
- `futex_do_wait` = properly waiting on mutex/condvar
- `do_epoll_wait` = waiting for I/O events
- `poll_schedule_timeout` = waiting in poll
- `hrtimer_nanosleep` = sleeping

### 4. Get Stack Traces with GDB

This is the most powerful tool for identifying exactly what code is running:

```bash
gdb -batch -p <PID> -ex "thread apply all bt 10" 2>&1 | grep -E "(Thread|in |at )"
```

This shows:
- All threads in the process
- Top 10 stack frames for each thread
- Exact function names and file locations
- Which threads are in busy loops (look for threads calling the same functions repeatedly)

Tips for reading backtraces:
- Threads calling `clock_gettime()` or similar repeatedly → profiling overhead in loop
- Threads in `try_lock()` or spin loops → lock contention
- Threads in `recv()` or `poll()` → properly blocked (good)
- Look for patterns across multiple busy threads

### 5. Syscall Tracing (if strace available)

```bash
timeout 2s strace -c -p <TID>
```

Shows syscall frequency. If very few syscalls → likely busy loop.

### 6. CPU Performance Profiling (if perf available)

```bash
timeout 3s perf record -g -p <PID> -F 99
perf report
```

Shows hotspot functions consuming CPU time.

## Investigation Workflow

1. **Identify high CPU process** with `ps aux`
2. **Find busy threads** with `top -H` or `ps -L`
3. **Get backtraces** with `gdb` - this is the key step!
4. **Analyze patterns** in the backtraces to find the busy loop
5. **Examine source code** at the identified locations
6. **Look for missing sleep/yield** or lock contention

## Common Busy-Wait Patterns

### Pattern 1: Loop with No Sleep
```rust
loop {
    if condition { break; }
    // No sleep here!
}
```
Fix: Add `std::thread::sleep()` or `std::thread::yield_now()`

### Pattern 2: Try-Lock Spin
```rust
loop {
    if let Ok(guard) = mutex.try_lock() {
        // work
        break;
    }
    // No sleep here!
}
```
Fix: Use blocking `lock()` or add sleep between retries

### Pattern 3: Pending State Without Sleep
```rust
loop {
    match work() {
        Ready => { /* process */ }
        Pending => { /* continues without sleep */ }
    }
}
```
Fix: Sleep when all work is pending

### Pattern 4: Forgotten Cancellation
```rust
// Thread spawned but cancellation token not called before drop
let handle = thread::spawn(|| loop { /* work */ });
// handle dropped without joining or canceling
```
Fix: Call cancellation token before dropping handles

## Example Investigation

Problem: Scanner process using 600% CPU

1. `ps aux | grep scanner` → PID 1257967, 542% CPU
2. `top -H -p 1257967` → 8 threads at 50-100% CPU each, WCHAN shows `-`
3. `gdb -batch -p 1257967 -ex "thread apply all bt 10"` → Shows all threads in `rustradio::graph::run()` calling `clock_gettime()`
4. Examined `rustradio/src/graph.rs:118` → Found `loop` with no sleep when `BlockRet::Pending`
5. Root cause: Audio graph threads not canceled when AudioEntity dropped

Solution: Cancel graph threads before clearing entities.

## Tips

- **GDB is the most reliable tool** - it shows exact code locations
- Look for **patterns across multiple threads** - same backtrace = systemic issue
- **WCHAN `-` is the smoking gun** for busy-waits
- Check for **missing Drop implementations** that should clean up resources
- Busy loops often have **profiling/timing code** in them (clock_gettime, get_cpu_time)
- Use `timeout` command to limit tool runtime on busy processes

## Related Issues

- Deadlocks: Use `gdb` to see all threads waiting on locks
- High memory: Use `ps aux` to check RSS/VSZ columns
- I/O wait: Check `top` for high `wa` percentage
- Thread leaks: Count threads with `ps -L | wc -l`
