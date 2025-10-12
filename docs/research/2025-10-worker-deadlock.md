# Worker Subprocess Deadlock Prevention and Simplification Analysis

Research into whether the worker subprocess's backpressure handling state machine can be simplified by having the parent drain the data channel before sending control commands.

## Problem Statement

The worker subprocess implements a state machine with 33 lines of backpressure handling code to prevent deadlock when transitioning between scanning windows. The question was whether we could simplify this by having the parent process drain all data from the data socket before sending the next control command.

## Current Architecture

The system uses two independent Unix domain socket channels:

1. **Control channel**: Synchronous command/response (ConfigureAndStart, StopStream, Shutdown)
2. **Data channel**: Asynchronous streaming IQ samples (continuous high-volume)

### The Deadlock Scenario

When the parent transitions from Window 1 to Window 2:

1. Parent stops reading from data socket (done processing Window 1 samples)
2. Data socket buffer fills up (kernel buffer ~212KB for Unix domain sockets)
3. Worker continues streaming → `dat_sender.send()` blocks on full socket
4. While blocked on data send → worker can't read control channel
5. Parent sends `StopStream` command → worker never receives it
6. **Deadlock**: Parent waits for `StreamStopped` response that never comes

### Current Solution

The worker prevents deadlock with three mechanisms (33 total lines):

1. **Pending command queue** (lines 311, 316-320): Save command detected during streaming for next iteration
2. **Pre-send command check** (lines 432-445): Check for commands before potentially blocking on send
3. **Timeout handling** (lines 459-480): 100ms write timeout on data socket, drop packets on timeout/backpressure

This allows the worker to detect `StopStream` even when the parent has stopped consuming data.

## Alternative: Parent-Side Draining

The proposed alternative was to have the parent continue reading and draining the data socket until all queued data is consumed, then send the control command. This would prevent the socket from filling up.

### Research Findings

#### 1. Synchronous IPC Patterns

Research into synchronous IPC command/response patterns (Electron `sendSync`, QNX message passing, Chromium IPC) revealed:

- Synchronous IPC works by blocking the sender until acknowledgment arrives
- This is the source of deadlock when combined with asynchronous streaming
- Industry solutions use separate control/data channels (not multiplexed)

Source: "IPC command response pattern synchronous wait acknowledgment" search results

#### 2. Backpressure Handling

Standard approaches to backpressure in producer-consumer systems:

1. **Control the producer**: Slow down or pause based on consumer rate
2. **Buffering**: Accumulate data temporarily (bounded by memory)
3. **Drop messages**: Discard data when consumer can't keep up

All three require the producer to detect backpressure and react. No approach eliminates the need for timeout/error handling.

Source: "Backpressure explained — the resisted flow of data through software" (Medium)

#### 3. Socket Draining Patterns

Unix domain socket research revealed:

- No standard "drain and synchronize" pattern exists
- Socket flush operations are unreliable (TCP_NODELAY for network sockets, fflush() doesn't work for sockets)
- Stream sockets don't preserve message boundaries - no natural end signal
- Draining requires either timeout or explicit end-of-stream message

Source: "separate control data channel pattern Unix socket flush drain synchronization" search results

#### 4. Out-of-Band Data

Investigation into MSG_OOB and SIGURG for urgent control messages:

- TCP urgent data limited to 1 byte (not practical for commands)
- Synchronization issues with SIGURG delivery
- Documentation recommends separate control connection instead
- Unix domain sockets don't reliably support MSG_OOB

Source: "out of band data channel Unix socket SIGURG MSG_OOB" search results

## Why Parent-Side Draining Doesn't Work

### Problem 1: No End-of-Data Signal

The worker continuously streams samples while the device is active. There's no natural "finished" point to detect. How long should the parent keep reading?

- Timeout-based: Unreliable, race conditions, timing dependent
- Explicit end message: Requires additional protocol state, adds round trips
- Empty socket detection: Can't distinguish "caught up" from "worker blocked elsewhere"

### Problem 2: Wastes Resources

The parent would read and immediately discard samples it doesn't need. The worker would serialize and send packets that will be thrown away. This wastes CPU, memory allocation, and kernel socket buffer space for no benefit.

### Problem 3: Still Need Backpressure Handling

Even with draining, there are moments where the parent falls behind or transitions windows. The worker still needs:

- Write timeouts to detect when parent stopped reading
- Error handling for socket write failures
- Logic to handle partially-sent data

Result: You'd have **both** drain coordination logic **and** current backpressure handling. Net increase in complexity.

### Problem 4: Timing Dependent

Race conditions between drain completion and new data arrival:

1. Parent drains socket, sees it empty
2. Worker sends new packet (between drain check and control command)
3. Parent sends control command
4. Worker blocks on new packet send
5. Back to deadlock

Requires additional synchronization:
- Parent sends "PrepareToStop" message
- Worker acknowledges, stops sending
- Parent drains remaining data
- Parent sends "StopStream"
- More protocol complexity, more state, more round trips

### Problem 5: Semantic Confusion

The control channel already provides synchronization: `StopStream` command → `StreamStopped` response. Adding drain coordination creates two overlapping mechanisms for the same goal.

## Current Solution is Minimal

The 33 lines of backpressure handling are not "complexity to eliminate" - they're **the solution** to a fundamental problem in the architecture:

### Essential Complexity

When you have:
- High-volume asynchronous data stream (producer)
- Low-volume synchronous control commands (coordinator)
- Independent channels (can't multiplex without protocol overhead)

Then you need:
- Producer to detect consumer slowdown/stop (timeouts)
- Producer to prioritize control over data (command checking)
- Producer to handle partial failure gracefully (drop packets)

This is documented as the industry-standard pattern for producer-consumer systems with control channels.

### Lines of Code Breakdown

1. **Pending command mechanism** (3 lines): Prevents command loss when detected during streaming
2. **Pre-send command check** (12 lines): Breaks potential deadlock by checking before blocking operation
3. **Timeout error handling** (18 lines): Gracefully handles backpressure by dropping packets

Total: 33 lines for complete deadlock prevention and graceful degradation.

### Comparison to Alternatives

- **Drain coordination**: Would require 40-60 lines (PrepareToStop protocol, timeout logic, race condition handling)
- **Multiplexed channel**: Protocol overhead, less efficient, harder to reason about
- **Single-threaded event loop**: More complex state machine, callback hell
- **Pull model**: Higher latency, more round trips, less efficient for streaming

## Real-World Patterns

Similar patterns found in:

1. **Video streaming**: Drop frames when consumer can't keep up
2. **Audio systems**: Buffer underruns handled by dropping/silence
3. **Network protocols**: TCP congestion control drops packets
4. **ZeroMQ**: Built-in high water marks and message dropping

All implement some form of backpressure handling in the producer because the alternative (unbounded buffering or deadlock) is worse.

## Recommendations

1. **Keep current implementation**: It's correct, minimal, and follows industry patterns
2. **Don't add drain coordination**: Increases complexity without eliminating backpressure handling
3. **Consider this essential complexity**: Not a code smell, but the nature of producer-consumer with separate channels

If uncomfortable with the solution, the issue isn't the implementation - it's that the architecture (separate control/data channels) inherently requires this coordination. But this architecture is correct for the requirements (high-volume streaming with command control).

## Alternative Architectures Considered

### Option 1: Spawn Worker Per Window

Create new subprocess for each scanning window instead of retuning existing device.

**Pros**: No window transition, no deadlock potential

**Cons**:
- Device recreation overhead (100-500ms per window)
- Lost initial samples during device initialization
- 50+ windows in band scan = 5-25 seconds of dead time
- Doesn't meet performance requirements

### Option 2: Single Multiplexed Channel

Combine control and data into one channel with message framing.

**Pros**: Single synchronization point

**Cons**:
- Protocol overhead on every data packet (message type, framing)
- Still need backpressure handling (buffer limits)
- Harder to reason about (control messages interspersed with data)
- Less efficient (serialization overhead)

### Option 3: Pull Model

Parent requests samples instead of worker pushing them.

**Pros**: Natural backpressure (requests control rate)

**Cons**:
- Higher latency (round trip per request)
- More CPU (constant request/response overhead)
- Harder to saturate bandwidth
- Doesn't match SDR device streaming model

## Conclusion

The current worker subprocess implementation is **not overly complex** - it's the minimal correct solution to the producer-consumer deadlock problem with separate control and data channels. The 33 lines of backpressure handling represent essential complexity that cannot be eliminated without either changing the architecture (which would harm performance) or moving the complexity elsewhere (which would make it harder to reason about).

Research into IPC patterns, socket synchronization, and backpressure handling confirms that this is the industry-standard approach. Attempting to simplify by adding parent-side draining would actually increase total complexity while introducing timing dependencies and race conditions.

Recommendation: Keep the current implementation.
