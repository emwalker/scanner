# Subprocess IPC Design Patterns

This document catalogs design patterns used in subprocess inter-process communication (IPC) systems. Each pattern addresses specific challenges in coordinating, managing, and communicating between isolated processes.

## Supervisor-Worker Pattern

A hierarchical process structure where supervisor processes monitor and manage worker processes. Workers perform actual computation while supervisors handle lifecycle management, failure recovery, and coordination.

The supervisor can restart workers when they fail, creating a supervision tree that enables fault-tolerant system design. Erlang/OTP popularized this pattern with configurable restart strategies that define how many times a worker should be restarted within a given period before giving up.

**When to use**: Systems requiring high availability and automatic fault recovery. Particularly effective when workers perform independent tasks that can be safely restarted without corrupting system state.

**When NOT to use**: Simple single-process applications. Systems where failure recovery requires complex state reconstruction that supervisors cannot easily orchestrate. Avoid when the overhead of supervision monitoring exceeds the benefit of automatic recovery.

## Producer-Consumer Pattern

Processes are designated as either producers (generating work) or consumers (processing work), connected by a thread-safe shared buffer or queue. Producers add to the shared structure, consumers remove from it, with synchronization primitives coordinating access.

The pattern provides decoupling so producers and consumers operate independently, enabling concurrent operation and flexible scaling by adding more producers or consumers without significant system changes.

**When to use**: Workloads with variable production and consumption rates. Systems benefiting from buffering between stages. Process pools where worker threads act as consumers processing tasks produced by another thread.

**When NOT to use**: Systems requiring immediate synchronous feedback from processing. Low-latency applications where queue overhead is unacceptable. Single-threaded environments where the pattern adds unnecessary complexity.

## Request-Reply Pattern

A message exchange pattern where a requestor sends a request message to a replier system, which processes the request and returns a response. Implementations range from synchronous blocking calls to fully asynchronous message passing.

For asynchronous implementations, one thread sends the request and sets up a callback for the reply. A separate thread listens for replies and invokes callbacks when responses arrive. Long-running operations may return a location reference that clients poll for results.

**When to use**: RPC-style interactions between processes. Systems requiring confirmation of command execution. Applications where request-response semantics match the business logic.

**When NOT to use**: Fire-and-forget notifications where responses are unnecessary. High-throughput systems where request-reply overhead reduces performance. Scenarios where one-way message passing suffices.

## Publish-Subscribe Pattern

Publishers send messages to topics without programming them for specific receivers. Subscribers express interest in topics and receive all messages published to those topics. A PUB socket sends the same message to all subscribers.

The pattern provides complete decoupling between publishers and subscribers, enabling dynamic subscription changes without publisher awareness. However, messages are dropped if no subscribers are connected when published.

**When to use**: Broadcasting events to multiple interested parties. Systems where subscriber sets change dynamically. Event-driven architectures where producers don't need to know about consumers.

**When NOT to use**: Guaranteed message delivery is required. Systems needing exactly-once processing. Request-reply interactions. Workload distribution where each message should be processed once.

## Push-Pull (Pipeline) Pattern

Intended for task distribution in multi-stage pipelines where one or few nodes push work to many workers, who push results to collectors. Push sockets distribute messages evenly among pull clients in round-robin fashion. Unlike pub-sub, each worker receives unique messages and messages are queued rather than dropped when no recipient is available.

**When to use**: Load balancing across multiple workers. Distributed task processing like web scraping where workers process different parts simultaneously. Multi-stage data processing pipelines.

**When NOT to use**: Broadcasting the same data to all workers. Systems requiring ordered processing. Low-latency scenarios where queuing delays are problematic.

## Backpressure Flow Control

A mechanism signaling upstream components to slow incoming request rates when the system cannot keep up. TCP/IP has built-in flow control. Three main strategies exist: time-to-live (TTL) drops old messages, tail drop refuses new messages, and backpressure informs upstream systems to reduce arrival rate.

Backpressure enables automatic producer slowdown when consumers can't keep up with incoming data. Modern implementations use hop-by-hop per-flow flow control with bounded state and constant-time operations.

**When to use**: Systems with variable processing speeds. Protecting resources from overload. Distributed systems needing stability under load. Message queues and reactive programming frameworks.

**When NOT to use**: Real-time systems requiring consistent latency. Loss-tolerant streaming where dropping frames is acceptable. Systems where upstream slowdown causes worse problems than overload.

## Circuit Breaker Pattern

Monitors operations for failures and trips when failures reach a threshold, returning errors immediately without attempting the protected operation. Operates in three states: closed (requests pass normally), open (traffic halted), and half-open (test requests determine recovery).

The pattern prevents repeated attempts at operations likely to fail, enabling applications to continue without waiting for fault fixes. This prevents cascading failures across distributed systems.

**When to use**: Protecting against external service failures. Distributed systems requiring resilience. Microservices architectures needing fault isolation. Operations with unpredictable failure rates.

**When NOT to use**: Local in-process operations. Systems where failures are rare and quickly resolved. Critical operations requiring retry regardless of cost. Scenarios where failure detection threshold is unclear.

## Actor Model

Characterized by inherent concurrency, dynamic actor creation, and interaction exclusively through direct asynchronous message passing. Actors are completely isolated, never sharing memory. Each has a mailbox and isolated state.

Message passing is asynchronous with no intermediate entities like channels. Each actor possesses a mailbox and can be addressed directly. Modifying internal state happens only via messages processed one at a time, eliminating races.

**When to use**: Highly concurrent distributed systems. Applications requiring strong fault isolation. Systems benefiting from location transparency. Complex stateful services needing guaranteed serial processing.

**When NOT to use**: Simple request-response services. Systems requiring shared state access. Performance-critical code where message passing overhead is problematic. Applications with complex transactional requirements across actors.

## Saga Pattern

Maintains data consistency across services through sequences of local transactions where each service performs its operation and initiates the next step via events or messages. If a step fails, compensating transactions undo previous changes.

Two coordination approaches exist: choreography where services exchange events without a centralized controller, and orchestration where a centralized orchestrator handles all transactions and coordinates participants.

**When to use**: Distributed transactions across multiple services. Long-running business processes spanning services. Systems prioritizing availability over immediate consistency.

**When NOT to use**: Simple single-service transactions. Systems requiring ACID guarantees. Operations where compensating transactions are impossible or impractical. Applications where distributed transaction complexity isn't justified.

## Bulkhead Pattern

Isolates application elements into pools so if one fails, others continue functioning. Named after ship hull partitions preventing the entire ship from sinking if one section is compromised.

Common implementations use separate thread pools, processes, or containers to isolate resources for different components. This prevents cascading failures and limits the blast radius of failures.

**When to use**: Systems requiring fault isolation between components. Resource-intensive operations needing separation. Protecting critical paths from non-critical component failures. Cloud-native microservices architectures.

**When NOT to use**: Simple single-component systems. Resource-constrained environments where isolation overhead is prohibitive. Monolithic applications without clear component boundaries.

## Event Sourcing with CQRS

Event sourcing uses an append-only store recording actions taken on data rather than storing just the latest state. CQRS separates read and write operations into distinct pathways, with commands modifying data and queries retrieving data.

These patterns commonly combine: data management tasks respond to events, and materialized views are built from stored events. Systems process events asynchronously to update read data stores, enabling loose coupling in microservices.

**When to use**: Systems requiring complete audit trails. Applications needing event replay for debugging or recovery. Complex domains benefiting from command-query separation. Microservices communicating asynchronously.

**When NOT to use**: Simple CRUD applications. Systems requiring immediate consistency on reads. Applications where event storage growth is problematic. Scenarios where eventual consistency is unacceptable.

## Leader Election

The process of designating a single process as organizer of tasks distributed across computers. Leaders make decisions, coordinate actions, and ensure smooth system operation. This pattern reduces concurrency complexity by centralizing it and reduces partial failure modes.

Algorithms like Raft and Paxos handle leader election. Raft uses heartbeat mechanisms to maintain leader status and trigger elections when necessary. Valid algorithms must guarantee termination, uniqueness (exactly one leader), and agreement (all processes know who leads).

**When to use**: Distributed systems needing coordination. Clusters requiring a single decision maker. Systems benefiting from centralized state management. Applications using master-slave replication.

**When NOT to use**: Fully peer-to-peer systems without central coordination. Systems where single point of failure is unacceptable without proper failover. Applications requiring multiple concurrent coordinators.

## Heartbeat Health Check Pattern

Services send periodic "I'm alive" signals or respond to health check queries. Two main approaches: push-based where nodes actively send signals to a central monitor, enabling fast failure detection; and pull-based where monitoring systems periodically query nodes, reducing network traffic but increasing detection latency.

Health check APIs (like HTTP /health endpoints) return service health. Handlers perform checks on infrastructure service connections. Microservices use heartbeats to enable performance monitors, schedulers, and orchestrators to track multiple services.

**When to use**: Distributed systems requiring failure detection. Process pools needing worker monitoring. Scheduled job monitoring. High-availability systems requiring failover.

**When NOT to use**: Simple single-process applications. Systems with external monitoring sufficient for needs. Environments where heartbeat network overhead is problematic. Applications with built-in supervision mechanisms.

## Poison Pill Pattern

Signals consumers to stop by sending a special "poison pill" message to the queue indicating no more messages follow. Allows consumers to finish processing current tasks before gracefully shutting down. The poison pill must be the last item on the queue to prevent premature consumer shutdown.

**When to use**: Producer-consumer scenarios requiring coordinated shutdown. Systems needing to drain message queues before termination. Process pools requiring graceful worker termination.

**When NOT to use**: Systems requiring immediate termination. Multiple producer scenarios where coordination is complex. Applications with simple shutdown requirements not needing queue draining.

## Scatter-Gather Pattern

Sends tasks to multiple services or processing units in parallel, collects responses, and aggregates them into consolidated output. The scatter phase broadcasts requests to recipients in parallel. The gather phase collects, filters, or combines responses into a unified result.

Unlike fan-out, scatter-gather is coordinated because it expects responses and applies logic to combine, compare, and select results. This achieves significant parallelization speedup over sequential processing.

**When to use**: Querying multiple data sources in parallel. Systems needing best-of-N selection from multiple providers. Complex queries benefiting from parallel processing. Price comparison or quote aggregation systems.

**When NOT to use**: Sequential processing is required. Small operations where coordination overhead exceeds benefits. Systems where the slowest response time is unacceptable for overall latency.

## Correlation ID Pattern

A unique identifier attached to requests that remains consistent as requests pass through multiple services. Enables associating specific responses with their requests. Typically passed in message headers or envelope properties.

The correlation ID should be assigned as early as possible and propagated to all downstream components. For async operations, the correlation ID passes in message payload or metadata. Message brokers like Kafka and RabbitMQ provide header fields for this purpose.

**When to use**: Distributed tracing across services. Request-response correlation in async systems. Log aggregation requiring request tracking. Microservices architectures needing request flow visibility.

**When NOT to use**: Simple single-process applications. Systems with no distributed tracing requirements. Scenarios where request-response association is inherent in the protocol.

## Pattern Interactions

Real systems combine multiple patterns:

A supervisor-worker system might use the producer-consumer pattern for task distribution, with workers implementing the actor model for isolation. Circuit breakers protect workers from external service failures while heartbeats enable supervisors to detect worker failures. The poison pill pattern enables graceful shutdown.

Microservices architectures commonly layer saga patterns for distributed transactions over request-reply or event sourcing with CQRS for service communication. Correlation IDs track requests across sagas. Bulkheads isolate saga participants. Circuit breakers prevent saga failures from cascading.

Leader election determines which process orchestrates sagas or coordinates scatter-gather operations. Backpressure prevents downstream services from being overwhelmed by scatter-gather fan-out. Pub-sub patterns distribute events in event sourcing systems while push-pull patterns distribute work in processing pipelines.

## References

### Supervisor-Worker
- Erlang System Documentation: Design Principles - https://www.erlang.org/doc/system/design_principles.html
- Learn You Some Erlang: Who Supervises The Supervisors - https://learnyousomeerlang.com/supervisors
- Erlang Supervisor Behaviour - https://www.erlang.org/doc/system/sup_princ.html

### Producer-Consumer
- Cornell CS3110: Producer/Consumer Pattern and Thread Pools - https://www.cs.cornell.edu/courses/cs3110/2010fa/lectures/lec18.html
- Java Design Patterns: Producer-Consumer - https://java-design-patterns.com/patterns/producer-consumer/
- Azure: Competing Consumers Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/competing-consumers

### Request-Reply
- Enterprise Integration Patterns: Request-Reply - https://www.enterpriseintegrationpatterns.com/patterns/messaging/RequestReply.html
- Azure: Asynchronous Request-Reply Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/async-request-reply
- AsyncAPI: Request/Reply Pattern - https://www.asyncapi.com/docs/tutorials/getting-started/request-reply

### Publish-Subscribe and Push-Pull
- ØMQ Guide: Sockets and Patterns - https://zguide.zeromq.org/docs/chapter2/
- ZeroMQ Socket API - https://zeromq.org/socket-api/
- Learning 0MQ: Publish/Subscribe - https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html

### Backpressure
- Jay Phelps: Backpressure Explained - https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7
- USENIX NSDI '22: Backpressure Flow Control - https://www.usenix.org/conference/nsdi22/presentation/goyal
- Enterprise Integration Patterns: Queues and Flow Control - https://www.enterpriseintegrationpatterns.com/ramblings/queues_flow_control.html

### Circuit Breaker
- Martin Fowler: Circuit Breaker - https://martinfowler.com/bliki/CircuitBreaker.html
- Azure: Circuit Breaker Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
- Aerospike: Efficient Fault Tolerance with Circuit Breaker - https://aerospike.com/blog/circuit-breaker-pattern/

### Actor Model
- Distributed Programming Book: Message Passing and the Actor Model - http://dist-prog-book.com/chapter/3/message-passing.html
- Akka Documentation: How the Actor Model Meets Modern Distributed Systems - https://doc.akka.io/libraries/akka-core/current/typed/guide/actors-intro.html
- Java Design Patterns: Actor Model - https://java-design-patterns.com/patterns/actor-model/

### Saga
- Azure: Saga Design Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/saga
- Microservices.io: Saga Pattern - https://microservices.io/patterns/data/saga.html
- Temporal: Mastering Saga Patterns - https://temporal.io/blog/mastering-saga-patterns-for-distributed-transactions-in-microservices

### Bulkhead
- Azure: Bulkhead Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead
- AWS: Building Fault Tolerant Architecture with Bulkhead Pattern - https://aws.amazon.com/blogs/containers/building-a-fault-tolerant-architecture-with-a-bulkhead-pattern-on-aws-app-mesh/
- GeeksforGeeks: Bulkhead Pattern - https://www.geeksforgeeks.org/system-design/bulkhead-pattern/

### Event Sourcing with CQRS
- Azure: CQRS Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs
- Azure: Event Sourcing Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing
- Microservices.io: Event Sourcing - https://microservices.io/patterns/data/event-sourcing.html

### Leader Election
- AWS Builders' Library: Leader Election in Distributed Systems - https://aws.amazon.com/builders-library/leader-election-in-distributed-systems/
- Azure: Leader Election Pattern - https://learn.microsoft.com/en-us/azure/architecture/patterns/leader-election
- GeeksforGeeks: Leader Election in System Design - https://www.geeksforgeeks.org/leader-election-in-system-design/

### Heartbeat Health Check
- Microservices.io: Health Check API - https://microservices.io/patterns/observability/health-check-api.html
- Microsoft: Health Monitoring in .NET - https://learn.microsoft.com/en-us/dotnet/architecture/microservices/implement-resilient-applications/monitor-app-health
- Medium: Building the Right Heartbeats - https://sureshkandula.medium.com/high-availability-patterns-building-the-right-heartbeats-health-checks-for-effective-failover-4f35b54a1e1e

### Poison Pill
- Java Design Patterns: Poison Pill - https://java-design-patterns.com/patterns/poison-pill/
- DZone: Producers and Consumers - Part 3 Poison Pills - https://dzone.com/articles/producers-and-consumers-part-3

### Scatter-Gather
- Enterprise Integration Patterns: Scatter-Gather - https://www.enterpriseintegrationpatterns.com/patterns/messaging/BroadcastAggregate.html
- AWS: Scatter-Gather Pattern - https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/scatter-gather.html
- Medium: The Scatter/Gather Pattern - https://medium.com/@ynskrn54/the-scatter-gather-pattern-increasing-the-effectiveness-of-task-processing-8338b5d29931

### Correlation ID
- Enterprise Integration Patterns: Correlation Identifier - https://www.enterpriseintegrationpatterns.com/patterns/messaging/CorrelationIdentifier.html
- Microsoft Engineering Playbook: Correlation IDs - https://microsoft.github.io/code-with-engineering-playbook/observability/correlation-id/
- Peter Hilton: Correlation IDs for Microservices - https://hilton.org.uk/blog/microservices-correlation-id
