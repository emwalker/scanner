# Task Scheduling Research: Delayed Resubmission Patterns

## Research Context

Investigation into task continuation patterns, cooperative yielding, and retry mechanisms with delays in concurrent systems.

## Tokio Cooperative Yielding

Tokio implements cooperative task scheduling with:

- **Budget-based yielding**: Each task has an operation budget that forces yielding after exhaustion
- **Manual yielding**: `tokio::task::yield_now().await` explicitly yields control
- **FIFO-fair semaphores**: Permits are granted in request order, preventing starvation

The Tokio semaphore (`tokio::sync::Semaphore`) is fair, meaning permits are given out in the order they were requested. The implementation uses a queue to fairly distribute permits. Cancelling a call to acquire makes you lose your place in the queue.

When a task cannot continue executing, it must yield, allowing the Tokio runtime to schedule another task. Although automatic cooperative task yielding improves performance in many cases, it cannot preempt tasks. Users must still take care to avoid both CPU intensive work and blocking APIs.

## Exponential Backoff Patterns

Industry standard patterns for retry with delays:

### Core Concepts

Exponential backoff is a technique where operations are retried by increasing wait times for a specified number of retry attempts. The most common pattern is exponential backoff, where the wait time is increased exponentially after every attempt.

### Best Practices

1. **Limit Retry Attempts**: Always set a maximum limit on retries to prevent infinite loops. If you need to retry indefinitely, ensure there is a maximum retry_delay to avoid the exponential backoff from growing too large.

2. **Implement Jitter**: Jitter adds randomness to the backoff to spread retries around in time. This helps prevent synchronized retries from many clients, which can create additional load at regular intervals. Jitter provides the best coordination between clients during high contention by maximizing the spread of retry intervals.

3. **Cap Maximum Backoff**: Exponential backoff can lead to very long backoff times because exponential functions grow quickly. Implementations typically cap their backoff to a maximum value.

4. **Consider Error Types**: Not all errors are worth retrying. For example, retrying after a 404 (Not Found) error usually doesn't make sense.

5. **Monitor and Log**: Log retry attempts and monitor the rate of retries to understand the health of external services and the network.

### Why Jitter Matters for Resource Contention

When failures are caused by overload or contention, backing off often doesn't help as much as expected due to correlation. If all failed calls back off to the same time, they cause contention or overload again when retried. Adding randomness (jitter) to request intervals prevents many requests from retrying simultaneously, which could create a "thundering herd" problem.

### Example Implementation

A maximum of three retries are configured with an increase multiplier of 1.5 seconds. If the first retry occurs after 3 seconds, the second retry occurs after 3 × 1.5 = 4.5 seconds, and the third retry occurs after 4.5 × 1.5 = 6.75 seconds.

## Rust Retry Libraries

### backoff crate

The `backoff` crate is a small library that allows retrying operations according to backoff policies. It provides:
- Error types to wrap errors as either transient or permanent
- Different backoff algorithms including exponential
- Support for both sync and async code
- `backoff::future::retry(ExponentialBackoff::default(), || async { ... })` for async operations

ExponentialBackoff increases the backoff period for each retry attempt using a randomization function that grows exponentially. The randomized interval is calculated as `retry_interval * (random value in range [1 - randomization_factor, 1 + randomization_factor])`.

### backon crate

The `backon` crate aims to make retry feel like a built-in feature with:
- Simple API: `your_fn.retry(ExponentialBuilder::default()).await`
- Support for both sync & async operations
- Retry function for all `FnMut() -> impl Future<Output=Result<T>>`
- Control over retry strategy by providing a backoff
- Defining actions to take during retries

## Task Continuation Patterns

Common patterns found in concurrent systems:

### Self-Continuation

Task reschedules itself. Instead of looping in a task, you can reschedule it using continuation methods. A pattern with delay is more clean and readable, using async/await with a do-while loop.

### Delay + Reschedule

The Task.Delay method can introduce pauses into an asynchronous method's execution and is useful for building polling loops and delaying the handling of user input. Task-based Asynchronous Pattern (TAP) uses callbacks to achieve waiting without blocking, achieved through methods such as Task.ContinueWith.

### External Timer/Scheduler

For designing delayed schedulers, a simple solution is to maintain a priority queue containing all tasks to be executed, with the most urgent task at the peek, and a thread iteratively checking the peek task and popping/running it if needed. The implementation uses a priority queue to manage tasks with wait/notify (signal) to handle concurrency problems.

### Event-Driven Delayed Actions

For event-driven systems with delayed actions, a simple solution is to create or integrate a scheduling service, where each new "delayed" event arrives and is added to the datastore for the service.

## Actor Model with Retry Patterns

The actor model provides patterns for retry with delays:

### Akka Retry Scheduling

Implementation of retries with growing delay intervals uses a parent/supervising actor that defines retries within a time window, with the worker child re-scheduling failed messages with delays.

### Ray Framework

Ray offers at-least-once execution semantics for actor tasks with automatic retries. It will retry tasks after RAY_task_retry_delay_ms until retries are consumed or the actor is ready.

### Child Session Actors

Using child session actors to handle retries, collecting responses, timeouts, and failure logic. Implementation of ack + retry patterns.

## Backpressure in Async Systems

Backpressure management is critical when systems start overloading.

### Semaphore-Based Backpressure

One approach to backpressure is using a semaphore with tokens. You acquire one at the beginning and wait for the semaphore to release a token when out of tokens. Backpressure prevents the runtime from spawning an unbounded number of tasks simultaneously. Tower's concurrency limit layer internally uses a semaphore initialized with the maximum number of concurrent requests allowed.

### Fairness vs Performance Trade-offs

The broader async ecosystem has ongoing discussions about fairness. Fair scheduling to producers is provided when wait queues use first-in, first-out ordering. There are trade-offs between preventing starvation and maximizing throughput.

## Async Delay Mechanisms

For implementing delays in async tasks:

- With async-std: `async_std::task::sleep()`
- With tokio: `tokio::time::sleep(Duration::from_millis(WAIT_TIME_INTERVAL_MS))`

### Manual Implementation

You can manually implement retry with exponential backoff by using a loop that sleeps with increasing durations:

```rust
loop {
    match operation().await {
        Ok(result) => return Ok(result),
        Err(e) if is_transient(&e) && retries < max_retries => {
            sleep(backoff).await;
            retries += 1;
            backoff *= 2;
        }
        Err(e) => return Err(e),
    }
}
```

## References

- Tokio cooperative task yielding: https://tokio.rs/blog/2020-04-preemption
- Tokio semaphore documentation: https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html
- Exponential backoff best practices: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- Rust backoff crate: https://docs.rs/backoff
- Rust backon crate: https://docs.rs/backon
- Azure Retry Pattern: https://learn.microsoft.com/en-us/azure/architecture/patterns/retry
- Akka retry scheduling: https://stackoverflow.com/questions/10364654/akka-how-to-schedule-retries-on-failure-with-growing-delay-intervals
- Ray actor fault tolerance: https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html
