# Rust Testing Tools and Crates

This document catalogs essential tools and crates for testing Rust applications, particularly multi-threaded code.

## Table of Contents

1. [Concurrency Testing](#concurrency-testing)
2. [Property-Based Testing](#property-based-testing)
3. [Async Testing](#async-testing)
4. [Mocking and Test Doubles](#mocking-and-test-doubles)
5. [Test Organization and Utilities](#test-organization-and-utilities)
6. [Debugging and Analysis](#debugging-and-analysis)
7. [Benchmarking](#benchmarking)

---

## Concurrency Testing

### loom - Deterministic Concurrency Testing

**Purpose:** Deterministically explore all possible thread interleavings to find race conditions and other concurrency bugs.

**Installation:**
```toml
[dev-dependencies]
loom = "0.7"
```

**Key Features:**
- Exhaustively tests all possible thread interleavings
- Finds race conditions that occur rarely in normal execution
- Provides deterministic reproduction of bugs
- Supports atomic operations, mutexes, RwLocks, and channels

**Usage:**
```rust
#[cfg(loom)]
mod loom_tests {
    use loom::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use loom::thread;

    #[test]
    fn test_concurrent_access() {
        loom::model(|| {
            let flag = Arc::new(AtomicBool::new(false));
            let f1 = flag.clone();
            let f2 = flag.clone();

            let t1 = thread::spawn(move || {
                f1.store(true, Ordering::SeqCst);
            });

            let t2 = thread::spawn(move || {
                f2.load(Ordering::SeqCst)
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }
}
```

**Running Loom Tests:**
```bash
# Must compile with loom cfg
RUSTFLAGS="--cfg loom" cargo test --test loom_test --release

# Enable logging
LOOM_LOG=1 RUSTFLAGS="--cfg loom" cargo test --test loom_test --release

# Enable location tracking for debugging
LOOM_LOCATION=1 RUSTFLAGS="--cfg loom" cargo test --test loom_test --release
```

**Best Practices:**
- Keep loom tests small (few threads, few operations)
- Use `thread::yield_now()` to create more interleavings
- Run in release mode for better performance
- Use cfg attributes to separate loom tests from regular tests

**Resources:**
- GitHub: https://github.com/tokio-rs/loom
- Documentation: https://docs.rs/loom
- Tutorial: https://matklad.github.io/2024/07/05/properly-testing-concurrent-data-structures.html

---

### shuttle - Randomized Concurrency Testing

**Purpose:** Alternative to loom using randomized scheduling for testing concurrent code.

**Installation:**
```toml
[dev-dependencies]
shuttle = "0.7"
```

**Key Differences from Loom:**
- Uses randomized scheduling instead of exhaustive exploration
- Can handle larger test cases
- Better for finding bugs in complex systems where exhaustive testing is infeasible

**Resources:**
- GitHub: https://github.com/awslabs/shuttle

---

## Property-Based Testing

### proptest - Property Testing Framework

**Purpose:** Test that properties hold for many randomly generated inputs.

**Installation:**
```toml
[dev-dependencies]
proptest = "1.0"
```

**Key Features:**
- Automatically generates test inputs
- Shrinks failing cases to minimal examples
- Configurable test case count and strategies
- Reproducible failures with seeds

**Basic Usage:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_addition_commutative(a in 0..1000i32, b in 0..1000i32) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn test_vec_length(v in prop::collection::vec(any::<u32>(), 0..100)) {
        prop_assert_eq!(v.len(), v.iter().count());
    }
}
```

**Custom Strategies:**
```rust
use proptest::strategy::{Strategy, Just};

fn frequency_strategy() -> impl Strategy<Value = f64> {
    (880_000_000u64..1080_000_000u64)
        .prop_map(|f| f as f64)
}

proptest! {
    #[test]
    fn test_frequency_in_range(freq in frequency_strategy()) {
        prop_assert!(freq >= 88.0e6 && freq <= 108.0e6);
    }
}
```

**Configuration:**
```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,  // Number of test cases
        max_shrink_iters: 10000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn my_property_test(x in 0..1000) {
        // ...
    }
}
```

**Best Practices:**
- Start with simple strategies and build up complexity
- Use `prop_map` to transform generated values
- Configure appropriate number of cases (balance speed vs coverage)
- Use proptest for invariant testing, not for specific scenarios

**Resources:**
- Documentation: https://docs.rs/proptest
- Proptest Book: https://proptest-rs.github.io/proptest/
- Tutorial: https://blog.logrocket.com/property-based-testing-in-rust-with-proptest/

---

### quickcheck - Another Property Testing Framework

**Purpose:** Alternative property testing framework inspired by Haskell's QuickCheck.

**Installation:**
```toml
[dev-dependencies]
quickcheck = "1.0"
quickcheck_macros = "1.0"
```

**Usage:**
```rust
#[cfg(test)]
use quickcheck_macros::quickcheck;

#[quickcheck]
fn test_reverse_reverse(xs: Vec<i32>) -> bool {
    let reversed_twice: Vec<_> = xs.iter().cloned().rev().rev().collect();
    xs == reversed_twice
}
```

**Comparison with proptest:**
- proptest: More powerful strategies, better shrinking
- quickcheck: Simpler API, faster compilation

**Resources:**
- Documentation: https://docs.rs/quickcheck

---

## Async Testing

### tokio-test - Testing for Tokio Applications

**Purpose:** Testing utilities for async code using Tokio.

**Installation:**
```toml
[dev-dependencies]
tokio = { version = "1", features = ["test-util", "macros"] }
tokio-test = "0.4"
```

**Key Features:**
- `#[tokio::test]` macro for async tests
- Mock implementations of IO types
- Time control for deterministic async tests

**Basic Async Test:**
```rust
#[tokio::test]
async fn test_async_function() {
    let result = my_async_function().await;
    assert!(result.is_ok());
}
```

**Multi-Threaded Runtime:**
```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_async() {
    let handles: Vec<_> = (0..10)
        .map(|i| tokio::spawn(async move { i * 2 }))
        .collect();

    let results = futures::future::join_all(handles).await;
    assert_eq!(results.len(), 10);
}
```

**Testing with Paused Time:**
```rust
#[tokio::test(start_paused = true)]
async fn test_timeout() {
    let start = tokio::time::Instant::now();

    // Manually advance time
    tokio::time::advance(std::time::Duration::from_secs(10)).await;

    assert!(start.elapsed() >= std::time::Duration::from_secs(10));
}
```

**Mock IO:**
```rust
use tokio_test::io::Builder;

#[tokio::test]
async fn test_read_write() {
    let mock = Builder::new()
        .read(b"hello")
        .write(b"world")
        .build();

    // Use mock in place of real IO
}
```

**Resources:**
- Documentation: https://docs.rs/tokio-test
- Tokio Testing Guide: https://tokio.rs/tokio/topics/testing

---

### async-std::test - Testing for async-std

**Purpose:** Testing utilities for async-std runtime.

**Installation:**
```toml
[dev-dependencies]
async-std = { version = "1", features = ["attributes"] }
```

**Usage:**
```rust
#[async_std::test]
async fn test_async() {
    let result = async_function().await;
    assert!(result.is_ok());
}
```

---

## Mocking and Test Doubles

### mockall - Powerful Mocking Library

**Purpose:** Create mock implementations of traits for testing.

**Installation:**
```toml
[dev-dependencies]
mockall = "0.12"
```

**Key Features:**
- Automatic mock generation from traits
- Expectation setting and verification
- Return value configuration
- Call count verification

**Basic Usage:**
```rust
use mockall::*;

#[automock]
trait Database {
    fn get_user(&self, id: u64) -> Option<String>;
    fn save_user(&mut self, id: u64, name: String) -> Result<(), String>;
}

#[test]
fn test_with_mock() {
    let mut mock = MockDatabase::new();

    // Set expectations
    mock.expect_get_user()
        .with(eq(1))
        .times(1)
        .returning(|_| Some("Alice".to_string()));

    mock.expect_save_user()
        .with(eq(2), eq("Bob".to_string()))
        .times(1)
        .returning(|_, _| Ok(()));

    // Use mock
    assert_eq!(mock.get_user(1), Some("Alice".to_string()));
    mock.save_user(2, "Bob".to_string()).unwrap();

    // Expectations verified on drop
}
```

**When to Use:**
- External services (databases, APIs)
- Non-deterministic behavior
- Expensive operations
- Error condition testing

**When NOT to Use:**
- Simple, fast, deterministic code
- Core business logic (use real implementations)

**Resources:**
- Documentation: https://docs.rs/mockall
- Tutorial: https://blog.logrocket.com/mocking-rust-mockall-alternatives/

---

### mockito - HTTP Mocking

**Purpose:** Mock HTTP servers for testing.

**Installation:**
```toml
[dev-dependencies]
mockito = "1.0"
```

**Usage:**
```rust
#[test]
fn test_http_client() {
    let mut server = mockito::Server::new();
    let mock = server.mock("GET", "/users/1")
        .with_status(200)
        .with_body(r#"{"id": 1, "name": "Alice"}"#)
        .create();

    let response = make_request(&server.url());

    mock.assert();
    assert_eq!(response.name, "Alice");
}
```

---

## Test Organization and Utilities

### serial_test - Sequential Test Execution

**Purpose:** Run tests sequentially when they share global state.

**Installation:**
```toml
[dev-dependencies]
serial_test = "3.0"
```

**Usage:**
```rust
use serial_test::serial;

#[test]
#[serial]
fn test_with_global_state_1() {
    // Runs exclusively
}

#[test]
#[serial]
fn test_with_global_state_2() {
    // Runs after test_1 completes
}
```

**Best Practice:** Avoid global state when possible; use this as last resort.

---

### rstest - Fixture-Based Testing

**Purpose:** Parameterized tests and test fixtures.

**Installation:**
```toml
[dev-dependencies]
rstest = "0.18"
```

**Parameterized Tests:**
```rust
use rstest::rstest;

#[rstest]
#[case(2, 4)]
#[case(3, 9)]
#[case(4, 16)]
fn test_square(#[case] input: i32, #[case] expected: i32) {
    assert_eq!(input * input, expected);
}
```

**Fixtures:**
```rust
#[fixture]
fn database() -> Database {
    Database::new_test_instance()
}

#[rstest]
fn test_with_fixture(database: Database) {
    // Use database fixture
}
```

---

### test-case - Parameterized Test Macro

**Purpose:** Simple parameterized testing.

**Installation:**
```toml
[dev-dependencies]
test-case = "3.0"
```

**Usage:**
```rust
use test_case::test_case;

#[test_case(2, 4)]
#[test_case(3, 9)]
#[test_case(4, 16)]
fn test_square(input: i32, expected: i32) {
    assert_eq!(input * input, expected);
}
```

---

### tempfile - Temporary Files and Directories

**Purpose:** Create temporary files/directories that are automatically cleaned up.

**Installation:**
```toml
[dev-dependencies]
tempfile = "3.0"
```

**Usage:**
```rust
#[test]
fn test_file_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("test.txt");

    std::fs::write(&file_path, "test data").unwrap();

    // Do tests...

    // temp_dir automatically deleted on drop
}
```

---

## Debugging and Analysis

### tracing-test - Capture Logs in Tests

**Purpose:** Capture and assert on log output in tests.

**Installation:**
```toml
[dev-dependencies]
tracing-test = "0.2"
```

**Usage:**
```rust
use tracing_test::traced_test;

#[test]
#[traced_test]
fn test_with_logs() {
    tracing::info!("Test started");
    do_work();
    tracing::info!("Test completed");

    // Logs captured and displayed only on failure
}
```

---

### cargo-nextest - Next-Generation Test Runner

**Purpose:** Faster test runner with better output and parallelization.

**Installation:**
```bash
cargo install cargo-nextest
```

**Usage:**
```bash
cargo nextest run

# Retry flaky tests
cargo nextest run --retries 3

# Profile-based configuration
cargo nextest run --profile ci
```

**Benefits:**
- Faster execution through better parallelization
- Cleaner output
- Retry failed tests
- Better CI integration

**Resources:**
- Website: https://nexte.st/

---

### deflake.rs - Flaky Test Detection

**Purpose:** Detect flaky tests by running them multiple times.

**Installation:**
```bash
cargo install deflake
```

**Usage:**
```bash
# Run each test 100 times
cargo deflake --iterations 100

# Focus on specific tests
cargo deflake --test test_name --iterations 50
```

---

### cargo-tarpaulin - Code Coverage

**Purpose:** Code coverage for Rust projects.

**Installation:**
```bash
cargo install cargo-tarpaulin
```

**Usage:**
```bash
cargo tarpaulin --out Html --output-dir coverage/
```

---

### ThreadSanitizer - Race Condition Detection

**Purpose:** Detect data races at runtime.

**Usage:**
```bash
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test
```

**Note:** Requires nightly Rust.

---

## Benchmarking

### criterion - Statistical Benchmarking

**Purpose:** Accurate performance benchmarking with statistical analysis.

**Installation:**
```toml
[dev-dependencies]
criterion = "0.5"
```

**Usage:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_function(c: &mut Criterion) {
    c.bench_function("my_function", |b| {
        b.iter(|| my_function(black_box(100)))
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

**Run:**
```bash
cargo bench
```

---

## Summary: Essential Testing Tools

**Concurrency:**
- loom - Exhaustive concurrency testing
- shuttle - Randomized concurrency testing

**Property Testing:**
- proptest - Powerful property-based testing
- quickcheck - Simpler property testing

**Async:**
- tokio-test - Tokio async testing
- async-std::test - async-std testing

**Mocking:**
- mockall - Trait mocking
- mockito - HTTP mocking

**Organization:**
- serial_test - Sequential execution
- rstest - Fixtures and parameterization
- tempfile - Temporary filesystem

**Debugging:**
- tracing-test - Log capture
- cargo-nextest - Better test runner
- deflake - Flaky test detection
- ThreadSanitizer - Race detection

**Benchmarking:**
- criterion - Statistical benchmarking

Choose tools based on your specific testing needs. Most projects benefit from: proptest for properties, loom for critical concurrency, tokio-test for async code, and tempfile for filesystem tests.
