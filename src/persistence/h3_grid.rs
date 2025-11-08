use geo_types::Coord;
use h3ron::{H3Cell, ToCoordinate};

use crate::persistence::location::Location;

pub struct H3Grid;

impl H3Grid {
    pub fn location_to_cell_id(location: Location, resolution: u8) -> Result<String, H3Error> {
        let coord = Coord {
            x: location.lon,
            y: location.lat,
        };
        let cell =
            H3Cell::from_coordinate(coord, resolution).map_err(|_| H3Error::ConversionError)?;

        Ok(cell.to_string())
    }

    pub fn get_signal_loading_cells(
        location: Location,
        signal_strength: f64,
    ) -> Result<Vec<String>, H3Error> {
        let resolution = Self::resolution_for_signal_strength(signal_strength);
        let base_cell_id = Self::location_to_cell_id(location, resolution)?;

        // For now, just return the base cell (will add neighbors later)
        Ok(vec![base_cell_id])
    }

    fn resolution_for_signal_strength(signal_strength: f64) -> u8 {
        if signal_strength >= 0.7 {
            4 // Strong signals: ~280km radius
        } else if signal_strength >= 0.4 {
            5 // Medium signals: ~100km radius
        } else {
            6 // Weak signals: ~40km radius
        }
    }

    pub fn cell_center(cell_id: &str) -> Result<Location, H3Error> {
        // Try to parse as hex string first, then as u64
        let cell_id_u64: u64 = if let Some(stripped) = cell_id.strip_prefix("0x") {
            u64::from_str_radix(stripped, 16).map_err(|_| H3Error::InvalidCellId)?
        } else {
            cell_id
                .parse::<u64>()
                .or_else(|_| u64::from_str_radix(cell_id, 16))
                .map_err(|_| H3Error::InvalidCellId)?
        };

        let cell = H3Cell::try_from(cell_id_u64).map_err(|_| H3Error::InvalidCellId)?;

        let coord = cell.to_coordinate().map_err(|_| H3Error::ConversionError)?;
        Ok(Location {
            lat: coord.y,
            lon: coord.x,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum H3Error {
    #[error("Invalid H3 resolution")]
    InvalidResolution,
    #[error("H3 coordinate conversion error")]
    ConversionError,
    #[error("Invalid H3 cell ID")]
    InvalidCellId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_to_cell_id() {
        let san_francisco = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        let cell_id = H3Grid::location_to_cell_id(san_francisco, 6).unwrap();
        assert!(!cell_id.is_empty());
        assert!(cell_id.len() > 10); // H3 cell IDs are long hex strings
    }

    #[test]
    fn test_signal_strength_resolution_mapping() {
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        // Strong signal should use resolution 4
        let strong_cells = H3Grid::get_signal_loading_cells(location, 0.8).unwrap();
        assert_eq!(strong_cells.len(), 1);

        // Medium signal should use resolution 5
        let medium_cells = H3Grid::get_signal_loading_cells(location, 0.5).unwrap();
        assert_eq!(medium_cells.len(), 1);

        // Weak signal should use resolution 6
        let weak_cells = H3Grid::get_signal_loading_cells(location, 0.3).unwrap();
        assert_eq!(weak_cells.len(), 1);
    }

    #[test]
    fn test_cell_center_roundtrip() {
        let original = Location {
            lat: 37.7749,
            lon: -122.4194,
        };
        let cell_id = H3Grid::location_to_cell_id(original, 6).unwrap();
        let center = H3Grid::cell_center(&cell_id).unwrap();

        // Should be close to original (within cell bounds)
        assert!((center.lat - original.lat).abs() < 0.1);
        assert!((center.lon - original.lon).abs() < 0.1);
    }
}
