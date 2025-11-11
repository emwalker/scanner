//! Clock abstraction for testable time operations

use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, Utc};

/// Clock operations abstraction for dependency injection
pub trait Clock: Send + Sync {
    /// Get current UTC datetime
    fn now_utc(&self) -> DateTime<Utc>;

    /// Get current instant for elapsed time measurements
    fn now_instant(&self) -> Instant;

    /// Get current system time
    fn now_system_time(&self) -> SystemTime;
}

/// Production clock implementation using system time
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_utc(&self) -> DateTime<Utc> {
        Utc::now()
    }

    fn now_instant(&self) -> Instant {
        Instant::now()
    }

    fn now_system_time(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// Mock clock for testing with controllable time
#[derive(Clone)]
pub struct MockClock {
    utc_time: DateTime<Utc>,
    instant: Instant,
    system_time: SystemTime,
}

impl MockClock {
    /// Create a new mock clock with current time
    pub fn new() -> Self {
        let now_utc = Utc::now();
        Self {
            utc_time: now_utc,
            instant: Instant::now(),
            system_time: SystemTime::now(),
        }
    }

    /// Create a mock clock with specific UTC time
    pub fn with_utc_time(utc_time: DateTime<Utc>) -> Self {
        Self {
            utc_time,
            instant: Instant::now(),
            system_time: SystemTime::now(),
        }
    }

    /// Create a mock clock with specific system time
    pub fn with_system_time(system_time: SystemTime) -> Self {
        Self {
            utc_time: Utc::now(),
            instant: Instant::now(),
            system_time,
        }
    }

    /// Advance the mock clock by a duration
    pub fn advance(&mut self, duration: Duration) {
        self.utc_time += chrono::Duration::from_std(duration).unwrap();
        self.instant += duration;
        self.system_time += duration;
    }

    /// Set the UTC time to a specific value
    pub fn set_utc_time(&mut self, time: DateTime<Utc>) {
        self.utc_time = time;
    }

    /// Set the system time to a specific value
    pub fn set_system_time(&mut self, time: SystemTime) {
        self.system_time = time;
    }

    /// Create a mock clock representing time in the past
    pub fn days_ago(days: u64) -> Self {
        let past_duration = Duration::from_secs(days * 24 * 60 * 60);
        Self {
            utc_time: Utc::now() - chrono::Duration::from_std(past_duration).unwrap(),
            instant: Instant::now(), // Instant doesn't support subtraction, use current
            system_time: SystemTime::now() - past_duration,
        }
    }

    /// Create a mock clock representing time in the future
    pub fn days_from_now(days: u64) -> Self {
        let future_duration = Duration::from_secs(days * 24 * 60 * 60);
        Self {
            utc_time: Utc::now() + chrono::Duration::from_std(future_duration).unwrap(),
            instant: Instant::now(),
            system_time: SystemTime::now() + future_duration,
        }
    }
}

impl Default for MockClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MockClock {
    fn now_utc(&self) -> DateTime<Utc> {
        self.utc_time
    }

    fn now_instant(&self) -> Instant {
        self.instant
    }

    fn now_system_time(&self) -> SystemTime {
        self.system_time
    }
}

/// Extension trait for creating durations from days
pub trait DurationExt {
    fn from_days(days: u64) -> Self;
    fn from_hours(hours: u64) -> Self;
}

impl DurationExt for Duration {
    fn from_days(days: u64) -> Self {
        Duration::from_secs(days * 24 * 60 * 60)
    }

    fn from_hours(hours: u64) -> Self {
        Duration::from_secs(hours * 60 * 60)
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_system_clock() {
        let clock = SystemClock;

        let utc_before = Utc::now();
        let instant_before = Instant::now();
        let system_before = SystemTime::now();

        let utc_clock = clock.now_utc();
        let instant_clock = clock.now_instant();
        let system_clock = clock.now_system_time();

        // All times should be close to current time
        assert!(utc_clock >= utc_before);
        assert!(instant_clock >= instant_before);
        assert!(system_clock >= system_before);
    }

    #[test]
    fn test_mock_clock_basic() {
        let specific_time = Utc.with_ymd_and_hms(2023, 5, 15, 12, 0, 0).unwrap();
        let clock = MockClock::with_utc_time(specific_time);

        assert_eq!(clock.now_utc(), specific_time);
    }

    #[test]
    fn test_mock_clock_advance() {
        let mut clock = MockClock::new();
        let initial_time = clock.now_utc();

        clock.advance(<Duration as DurationExt>::from_hours(2));

        let advanced_time = clock.now_utc();
        let expected_time = initial_time + chrono::Duration::hours(2);

        assert_eq!(advanced_time, expected_time);
    }

    #[test]
    fn test_mock_clock_days_ago() {
        let clock = MockClock::days_ago(30);
        let now = Utc::now();
        let clock_time = clock.now_utc();

        let age = now.signed_duration_since(clock_time);
        assert!(age.num_days() >= 29); // Allow for some time passage during test
        assert!(age.num_days() <= 31); // But not too much
    }

    #[test]
    fn test_mock_clock_days_from_now() {
        let clock = MockClock::days_from_now(7);
        let now = Utc::now();
        let clock_time = clock.now_utc();

        let difference = clock_time.signed_duration_since(now);
        assert!(difference.num_days() >= 6); // Allow for some time passage during test
        assert!(difference.num_days() <= 8); // But not too much
    }

    #[test]
    fn test_duration_ext() {
        assert_eq!(
            <Duration as DurationExt>::from_days(1),
            Duration::from_secs(24 * 60 * 60)
        );
        assert_eq!(
            <Duration as DurationExt>::from_days(7),
            Duration::from_secs(7 * 24 * 60 * 60)
        );
        assert_eq!(
            <Duration as DurationExt>::from_hours(1),
            Duration::from_secs(60 * 60)
        );
        assert_eq!(
            <Duration as DurationExt>::from_hours(24),
            <Duration as DurationExt>::from_days(1)
        );
    }

    #[test]
    fn test_mock_clock_set_times() {
        let mut clock = MockClock::new();

        let new_utc = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        clock.set_utc_time(new_utc);
        assert_eq!(clock.now_utc(), new_utc);

        let new_system = SystemTime::UNIX_EPOCH + Duration::from_secs(1000000);
        clock.set_system_time(new_system);
        assert_eq!(clock.now_system_time(), new_system);
    }

    #[test]
    fn test_mock_clock_with_system_time() {
        let specific_system_time = SystemTime::UNIX_EPOCH + Duration::from_secs(1000000);
        let clock = MockClock::with_system_time(specific_system_time);

        assert_eq!(clock.now_system_time(), specific_system_time);
    }
}
