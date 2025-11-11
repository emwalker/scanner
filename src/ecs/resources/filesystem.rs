//! Filesystem abstraction for testable file operations

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::SystemTime,
};

/// Filesystem operations abstraction for dependency injection
pub trait FileSystem: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Read file contents as string
    fn read_to_string(&self, path: &Path) -> Result<String, Self::Error>;

    /// Write string content to file
    fn write(&self, path: &Path, content: &str) -> Result<(), Self::Error>;

    /// Read file contents as bytes
    fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, Self::Error>;

    /// Check if a file or directory exists
    fn exists(&self, path: &Path) -> bool;

    /// Get file metadata (for checking modification times)
    fn metadata(&self, path: &Path) -> Result<FileMetadata, Self::Error>;

    /// Create directory and all parent directories
    fn create_dir_all(&self, path: &Path) -> Result<(), Self::Error>;

    /// Get the user's home directory
    fn home_dir(&self) -> Option<PathBuf>;
}

/// File metadata for checking file properties
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub modified: SystemTime,
    pub size: u64,
}

/// Production filesystem implementation using std::fs
pub struct StdFileSystem;

impl FileSystem for StdFileSystem {
    type Error = std::io::Error;

    fn read_to_string(&self, path: &Path) -> Result<String, Self::Error> {
        std::fs::read_to_string(path)
    }

    fn write(&self, path: &Path, content: &str) -> Result<(), Self::Error> {
        std::fs::write(path, content)
    }

    fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, Self::Error> {
        std::fs::read(path)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn metadata(&self, path: &Path) -> Result<FileMetadata, Self::Error> {
        let metadata = std::fs::metadata(path)?;
        Ok(FileMetadata {
            modified: metadata.modified()?,
            size: metadata.len(),
        })
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), Self::Error> {
        std::fs::create_dir_all(path)
    }

    fn home_dir(&self) -> Option<PathBuf> {
        dirs::home_dir()
    }
}

/// In-memory filesystem for testing
#[derive(Default)]
pub struct MockFileSystem {
    files: std::sync::RwLock<HashMap<PathBuf, MockFile>>,
    home_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct MockFile {
    content: Vec<u8>,
    modified: SystemTime,
}

#[derive(Debug, thiserror::Error)]
pub enum MockFileSystemError {
    #[error("No such file or directory: {path}")]
    NotFound { path: String },
    #[error("Permission denied: {path}")]
    PermissionDenied { path: String },
    #[error("File operation failed: {message}")]
    Other { message: String },
}

impl MockFileSystem {
    pub fn new() -> Self {
        Self {
            files: Default::default(),
            home_dir: Some(PathBuf::from("/tmp/test_home")),
        }
    }

    /// Set a custom home directory for testing
    pub fn with_home_dir(mut self, home_dir: PathBuf) -> Self {
        self.home_dir = Some(home_dir);
        self
    }

    /// Pre-populate a file for testing
    pub fn add_file<P: AsRef<Path>, C: AsRef<[u8]>>(&self, path: P, content: C) {
        let file = MockFile {
            content: content.as_ref().to_vec(),
            modified: SystemTime::now(),
        };
        self.files
            .write()
            .unwrap()
            .insert(path.as_ref().to_path_buf(), file);
    }

    /// Set file modification time for testing
    pub fn set_file_modified<P: AsRef<Path>>(&self, path: P, modified: SystemTime) {
        if let Some(file) = self.files.write().unwrap().get_mut(path.as_ref()) {
            file.modified = modified;
        }
    }

    /// Check if a file exists in mock filesystem
    pub fn has_file<P: AsRef<Path>>(&self, path: P) -> bool {
        self.files.read().unwrap().contains_key(path.as_ref())
    }

    /// Remove a file from mock filesystem
    pub fn remove_file<P: AsRef<Path>>(&self, path: P) {
        self.files.write().unwrap().remove(path.as_ref());
    }
}

impl FileSystem for MockFileSystem {
    type Error = MockFileSystemError;

    fn read_to_string(&self, path: &Path) -> Result<String, Self::Error> {
        let files = self.files.read().unwrap();
        let file = files
            .get(path)
            .ok_or_else(|| MockFileSystemError::NotFound {
                path: path.display().to_string(),
            })?;

        String::from_utf8(file.content.clone()).map_err(|_| MockFileSystemError::Other {
            message: "Invalid UTF-8".to_string(),
        })
    }

    fn write(&self, path: &Path, content: &str) -> Result<(), Self::Error> {
        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            self.create_dir_all(parent)?;
        }

        let file = MockFile {
            content: content.as_bytes().to_vec(),
            modified: SystemTime::now(),
        };

        self.files.write().unwrap().insert(path.to_path_buf(), file);
        Ok(())
    }

    fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, Self::Error> {
        let files = self.files.read().unwrap();
        let file = files
            .get(path)
            .ok_or_else(|| MockFileSystemError::NotFound {
                path: path.display().to_string(),
            })?;

        Ok(file.content.clone())
    }

    fn exists(&self, path: &Path) -> bool {
        self.files.read().unwrap().contains_key(path)
    }

    fn metadata(&self, path: &Path) -> Result<FileMetadata, Self::Error> {
        let files = self.files.read().unwrap();
        let file = files
            .get(path)
            .ok_or_else(|| MockFileSystemError::NotFound {
                path: path.display().to_string(),
            })?;

        Ok(FileMetadata {
            modified: file.modified,
            size: file.content.len() as u64,
        })
    }

    fn create_dir_all(&self, _path: &Path) -> Result<(), Self::Error> {
        // Mock implementation - just succeed
        // In real tests, you might want to track created directories
        Ok(())
    }

    fn home_dir(&self) -> Option<PathBuf> {
        self.home_dir.clone()
    }
}

// Implement FileSystem for &MockFileSystem to allow passing by reference
impl FileSystem for &MockFileSystem {
    type Error = MockFileSystemError;

    fn read_to_string(&self, path: &Path) -> Result<String, Self::Error> {
        (*self).read_to_string(path)
    }

    fn write(&self, path: &Path, content: &str) -> Result<(), Self::Error> {
        (*self).write(path, content)
    }

    fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, Self::Error> {
        (*self).read_bytes(path)
    }

    fn exists(&self, path: &Path) -> bool {
        (*self).exists(path)
    }

    fn metadata(&self, path: &Path) -> Result<FileMetadata, Self::Error> {
        (*self).metadata(path)
    }

    fn create_dir_all(&self, path: &Path) -> Result<(), Self::Error> {
        (*self).create_dir_all(path)
    }

    fn home_dir(&self) -> Option<PathBuf> {
        (*self).home_dir()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_std_filesystem_operations() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        let fs = StdFileSystem;

        // Test writing and reading
        fs.write(&file_path, "test content").unwrap();
        let content = fs.read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");

        // Test exists
        assert!(fs.exists(&file_path));
        assert!(!fs.exists(&temp_dir.path().join("nonexistent.txt")));

        // Test metadata
        let metadata = fs.metadata(&file_path).unwrap();
        assert_eq!(metadata.size, 12); // "test content" is 12 bytes

        // Test read_bytes
        let bytes = fs.read_bytes(&file_path).unwrap();
        assert_eq!(bytes, b"test content");
    }

    #[test]
    fn test_mock_filesystem_operations() {
        let fs = MockFileSystem::new();

        let file_path = Path::new("/test/file.txt");

        // Test writing and reading
        fs.write(file_path, "mock content").unwrap();
        let content = fs.read_to_string(file_path).unwrap();
        assert_eq!(content, "mock content");

        // Test exists
        assert!(fs.exists(file_path));
        assert!(!fs.exists(Path::new("/nonexistent.txt")));

        // Test metadata
        let metadata = fs.metadata(file_path).unwrap();
        assert_eq!(metadata.size, 12); // "mock content" is 12 bytes

        // Test read_bytes
        let bytes = fs.read_bytes(file_path).unwrap();
        assert_eq!(bytes, b"mock content");
    }

    #[test]
    fn test_mock_filesystem_pre_populated() {
        let fs = MockFileSystem::new();
        let file_path = Path::new("/existing.txt");

        fs.add_file(file_path, "pre-existing content");

        assert!(fs.has_file(file_path));
        let content = fs.read_to_string(file_path).unwrap();
        assert_eq!(content, "pre-existing content");
    }

    #[test]
    fn test_mock_filesystem_modification_time() {
        let fs = MockFileSystem::new();
        let file_path = Path::new("/time_test.txt");

        let past_time = SystemTime::now() - Duration::from_secs(3600); // 1 hour ago

        fs.add_file(file_path, "content");
        fs.set_file_modified(file_path, past_time);

        let metadata = fs.metadata(file_path).unwrap();
        assert_eq!(metadata.modified, past_time);
    }

    #[test]
    fn test_mock_filesystem_custom_home_dir() {
        let custom_home = PathBuf::from("/custom/home");
        let fs = MockFileSystem::new().with_home_dir(custom_home.clone());

        assert_eq!(fs.home_dir(), Some(custom_home));
    }

    #[test]
    fn test_mock_filesystem_file_not_found() {
        let fs = MockFileSystem::new();
        let result = fs.read_to_string(Path::new("/nonexistent.txt"));

        assert!(result.is_err());
        match result.unwrap_err() {
            MockFileSystemError::NotFound { path } => {
                assert_eq!(path, "/nonexistent.txt");
            }
            _ => panic!("Expected NotFound error"),
        }
    }
}
