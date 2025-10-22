# GDB Backtrace Pattern Reference

This reference provides examples of GDB backtraces for common performance issues to help with pattern recognition during investigations.

## Busy-Wait Loop Patterns

### Pattern: Polling Loop Without Sleep

**Symptoms:** High CPU usage, WCHAN shows `-`, thread in R state

**Example Backtrace:**
```
Thread 5 (Thread 0x7f8b4c7fa700 (LWP 12345)):
#0  0x00007f8b4d123456 in clock_gettime ()
#1  0x0000555555678901 in std::time::Instant::now ()
#2  0x0000555555678abc in my_app::poll_work ()
#3  0x0000555555678def in my_app::worker_thread ()
#4  0x00007f8b4d234567 in start_thread ()
```

**Interpretation:** Thread repeatedly checking time in a loop without blocking. The `clock_gettime` call indicates timing/profiling code in a tight loop.

### Pattern: Try-Lock Spin Loop

**Symptoms:** High CPU, multiple threads showing similar backtraces

**Example Backtrace:**
```
Thread 3 (Thread 0x7f8b4c6f9700 (LWP 12343)):
#0  0x0000555555789abc in parking_lot::raw_mutex::RawMutex::try_lock ()
#1  0x0000555555789def in std::sync::Mutex::try_lock ()
#2  0x0000555555789fff in my_app::acquire_resource ()
#3  0x000055555578a111 in my_app::process_loop ()
```

**Interpretation:** Thread spinning on try_lock without sleeping between attempts. This creates high CPU contention.

## Properly Blocked Patterns (Low/No CPU)

### Pattern: Waiting on Mutex

**Symptoms:** Low/no CPU usage, WCHAN shows `futex_wait`

**Example Backtrace:**
```
Thread 4 (Thread 0x7f8b4c7fa700 (LWP 12344)):
#0  0x00007f8b4d123456 in futex_wait ()
#1  0x0000555555678901 in parking_lot::raw_mutex::RawMutex::lock ()
#2  0x0000555555678abc in std::sync::Mutex::lock ()
#3  0x0000555555678def in my_app::worker_thread ()
```

**Interpretation:** Thread properly blocked waiting for a mutex. This is normal and efficient.

### Pattern: Waiting on Channel

**Symptoms:** Low/no CPU, WCHAN shows `futex_wait`

**Example Backtrace:**
```
Thread 6 (Thread 0x7f8b4c5f8700 (LWP 12346)):
#0  0x00007f8b4d123456 in futex_wait ()
#1  0x0000555555678901 in crossbeam::channel::blocking::recv ()
#2  0x0000555555678abc in my_app::consumer_thread ()
```

**Interpretation:** Thread blocked waiting for channel data. Efficient and expected.

### Pattern: Epoll Wait (I/O)

**Symptoms:** Low/no CPU, WCHAN shows `do_epoll_wait`

**Example Backtrace:**
```
Thread 2 (Thread 0x7f8b4c8fb700 (LWP 12342)):
#0  0x00007f8b4d234567 in epoll_wait ()
#1  0x0000555555789abc in mio::poll::Poll::poll ()
#2  0x0000555555789def in tokio::runtime::io::driver::Driver::turn ()
#3  0x000055555578a111 in tokio::runtime::thread_pool::worker ()
```

**Interpretation:** Tokio runtime thread waiting for I/O events. Normal async runtime behavior.

## Deadlock Patterns

### Pattern: Circular Wait

**Symptoms:** Threads blocked indefinitely, no progress

**Example - Thread A:**
```
Thread 3 (Thread 0x7f8b4c6f9700 (LWP 12343)):
#0  0x00007f8b4d123456 in futex_wait ()
#1  0x0000555555789abc in std::sync::Mutex::lock () at resource_b
#2  0x0000555555789def in my_app::process_with_b ()
```

**Example - Thread B:**
```
Thread 4 (Thread 0x7f8b4c5f8700 (LWP 12344)):
#0  0x00007f8b4d123456 in futex_wait ()
#1  0x0000555555789abc in std::sync::Mutex::lock () at resource_a
#2  0x0000555555789fed in my_app::process_with_a ()
```

**Interpretation:** Thread A holds resource_a and waits for resource_b, while Thread B holds resource_b and waits for resource_a. Classic deadlock.

## Resource Leak Patterns

### Pattern: Abandoned Thread Loop

**Symptoms:** Threads continue running after main work is done

**Example Backtrace:**
```
Thread 8 (Thread 0x7f8b4c3f6700 (LWP 12348)):
#0  0x00007f8b4d123456 in clock_gettime ()
#1  0x0000555555678901 in rustradio::graph::run ()
#2  0x0000555555678abc in std::thread::spawn::{{closure}} ()
```

**Interpretation:** Graph processing thread still running because cancellation token was never triggered. Check Drop implementations.

## Diagnostic Tips

### Identifying Systemic Issues

When multiple threads show the same backtrace pattern, it indicates a systemic issue:

```
Thread 5: my_app::worker_loop -> poll_work -> clock_gettime
Thread 6: my_app::worker_loop -> poll_work -> clock_gettime
Thread 7: my_app::worker_loop -> poll_work -> clock_gettime
Thread 8: my_app::worker_loop -> poll_work -> clock_gettime
```

All worker threads are busy-waiting in the same function. Fix the root function rather than individual threads.

### Stack Depth Analysis

**Shallow stacks (2-4 frames):** Often indicates tight loop at top level
**Deep stacks (10+ frames):** Usually indicates recursive call or complex call chain

### Function Name Patterns

**clock_gettime, get_cpu_time:** Timing code in a loop (profiling overhead)
**try_lock, spin_loop_hint:** Lock contention without proper blocking
**recv, poll, epoll_wait:** Properly blocked I/O operations (good)
**futex_wait, pthread_cond_wait:** Properly blocked synchronization (good)
