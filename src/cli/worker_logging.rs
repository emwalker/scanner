use std::path::Path;

pub enum WorkerType {
    Enumeration,
    Device,
}

pub struct WorkerContext {
    pub device_id: Option<String>,
    pub timestamp: Option<u128>,
    pub backend: Option<String>,
}

pub fn generate_worker_log_path(
    parent_log: Option<&str>,
    worker_type: WorkerType,
    context: &WorkerContext,
) -> Option<String> {
    let parent_log = parent_log?;

    let parent_path = Path::new(parent_log);
    let parent_dir = parent_path
        .parent()
        .and_then(|p| p.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(".");
    let parent_stem = parent_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scanner");

    match worker_type {
        WorkerType::Enumeration => {
            let backend = context.backend.as_deref().unwrap_or("unknown");
            Some(format!(
                "{}/{}-enum-{}.log",
                parent_dir, parent_stem, backend
            ))
        }
        WorkerType::Device => {
            let device_id = context.device_id.as_deref().unwrap_or("unknown");
            let timestamp = context.timestamp.unwrap_or(0);
            Some(format!(
                "{}/{}-worker-{}-{}.log",
                parent_dir, parent_stem, device_id, timestamp
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumeration_worker_default_pattern() {
        let context = WorkerContext {
            device_id: None,
            timestamp: None,
            backend: Some("soapy".to_string()),
        };

        let result =
            generate_worker_log_path(Some("/tmp/scanner.log"), WorkerType::Enumeration, &context);

        assert_eq!(result, Some("/tmp/scanner-enum-soapy.log".to_string()));
    }

    #[test]
    fn test_device_worker_default_pattern() {
        let context = WorkerContext {
            device_id: Some("sdrplay-123".to_string()),
            timestamp: Some(1234567890),
            backend: None,
        };

        let result =
            generate_worker_log_path(Some("/var/log/scanner.log"), WorkerType::Device, &context);

        assert_eq!(
            result,
            Some("/var/log/scanner-worker-sdrplay-123-1234567890.log".to_string())
        );
    }

    #[test]
    fn test_no_parent_log_returns_none() {
        let context = WorkerContext {
            device_id: Some("test".to_string()),
            timestamp: Some(123),
            backend: Some("soapy".to_string()),
        };

        let result = generate_worker_log_path(None, WorkerType::Device, &context);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parent_in_current_directory() {
        let context = WorkerContext {
            device_id: Some("device1".to_string()),
            timestamp: Some(111),
            backend: None,
        };

        let result = generate_worker_log_path(Some("scanner.log"), WorkerType::Device, &context);

        assert_eq!(result, Some("./scanner-worker-device1-111.log".to_string()));
    }
}
