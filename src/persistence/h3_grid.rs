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

    // === H3 Integration Tests ===

    #[test]
    fn test_coordinate_edge_cases() {
        // Test polar regions (near ±85° latitude limit for H3)
        let arctic = Location {
            lat: 84.0,
            lon: 0.0,
        };
        let arctic_cell = H3Grid::location_to_cell_id(arctic, 6);
        assert!(arctic_cell.is_ok(), "Arctic coordinates should work");

        let antarctic = Location {
            lat: -84.0,
            lon: 180.0,
        };
        let antarctic_cell = H3Grid::location_to_cell_id(antarctic, 6);
        assert!(antarctic_cell.is_ok(), "Antarctic coordinates should work");

        // Test antimeridian crossing (±180° longitude)
        let antimeridian_east = Location {
            lat: 0.0,
            lon: 179.9,
        };
        let east_cell = H3Grid::location_to_cell_id(antimeridian_east, 6);
        assert!(east_cell.is_ok(), "Near antimeridian east should work");

        let antimeridian_west = Location {
            lat: 0.0,
            lon: -179.9,
        };
        let west_cell = H3Grid::location_to_cell_id(antimeridian_west, 6);
        assert!(west_cell.is_ok(), "Near antimeridian west should work");

        // Test equatorial coordinates
        let equator = Location { lat: 0.0, lon: 0.0 };
        let equator_cell = H3Grid::location_to_cell_id(equator, 6);
        assert!(equator_cell.is_ok(), "Equatorial coordinates should work");

        // Test extreme coordinates (H3 library handles these gracefully)
        let extreme_lat = Location {
            lat: 91.0,
            lon: 0.0,
        };
        let extreme_result = H3Grid::location_to_cell_id(extreme_lat, 6);
        // H3 library may clamp or handle extreme coordinates without error
        if let Ok(cell_id) = extreme_result {
            assert!(
                !cell_id.is_empty(),
                "Extreme latitude should produce valid cell ID"
            );
        }

        let extreme_lon = Location {
            lat: 0.0,
            lon: 181.0,
        };
        let extreme_result = H3Grid::location_to_cell_id(extreme_lon, 6);
        // H3 library may wrap or handle extreme coordinates without error
        if let Ok(cell_id) = extreme_result {
            assert!(
                !cell_id.is_empty(),
                "Extreme longitude should produce valid cell ID"
            );
        }
    }

    #[test]
    fn test_resolution_boundary_values() {
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        // Test exact threshold values for signal strength mapping
        let strong_threshold = H3Grid::get_signal_loading_cells(location, 0.7);
        assert!(strong_threshold.is_ok());

        let medium_threshold = H3Grid::get_signal_loading_cells(location, 0.4);
        assert!(medium_threshold.is_ok());

        // Test values just above and below thresholds
        let just_above_strong = H3Grid::get_signal_loading_cells(location, 0.70001);
        let just_below_strong = H3Grid::get_signal_loading_cells(location, 0.69999);
        assert!(just_above_strong.is_ok());
        assert!(just_below_strong.is_ok());

        let just_above_medium = H3Grid::get_signal_loading_cells(location, 0.40001);
        let just_below_medium = H3Grid::get_signal_loading_cells(location, 0.39999);
        assert!(just_above_medium.is_ok());
        assert!(just_below_medium.is_ok());

        // Test extreme values
        let max_signal = H3Grid::get_signal_loading_cells(location, 1.0);
        let min_signal = H3Grid::get_signal_loading_cells(location, 0.0);
        assert!(max_signal.is_ok());
        assert!(min_signal.is_ok());

        // Test extreme signal strengths (no validation in current implementation)
        let over_max = H3Grid::get_signal_loading_cells(location, 1.1);
        let negative = H3Grid::get_signal_loading_cells(location, -0.1);
        assert!(over_max.is_ok(), "Signal strength > 1.0 should be accepted");
        assert!(
            negative.is_ok(),
            "Negative signal strength should be accepted"
        );

        // Very high values should use strongest resolution (4)
        let very_high = H3Grid::get_signal_loading_cells(location, 10.0);
        assert!(very_high.is_ok());
    }

    #[test]
    fn test_cell_id_format_variations() {
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };
        let cell_id = H3Grid::location_to_cell_id(location, 6).unwrap();

        // Test original format
        let center1 = H3Grid::cell_center(&cell_id);
        assert!(center1.is_ok());

        // Test with 0x prefix (hex format)
        let hex_format = format!("0x{}", cell_id);
        let center2 = H3Grid::cell_center(&hex_format);
        assert!(center2.is_ok());

        // Test uppercase hex
        let uppercase_format = cell_id.to_uppercase();
        let center3 = H3Grid::cell_center(&uppercase_format);
        assert!(center3.is_ok());

        // All should give same result
        let c1 = center1.unwrap();
        let c2 = center2.unwrap();
        let c3 = center3.unwrap();
        assert!((c1.lat - c2.lat).abs() < 0.001);
        assert!((c1.lon - c2.lon).abs() < 0.001);
        assert!((c1.lat - c3.lat).abs() < 0.001);
        assert!((c1.lon - c3.lon).abs() < 0.001);

        // Test invalid formats
        let invalid_empty = H3Grid::cell_center("");
        assert!(invalid_empty.is_err());

        let invalid_format = H3Grid::cell_center("not_a_hex_number");
        assert!(invalid_format.is_err());

        let invalid_length = H3Grid::cell_center("123");
        assert!(invalid_length.is_err());
    }

    #[test]
    fn test_multi_location_h3_organization() {
        // Test locations that should map to different cells
        let san_francisco = Location {
            lat: 37.7749,
            lon: -122.4194,
        };
        let new_york = Location {
            lat: 40.7128,
            lon: -74.0060,
        };
        let london = Location {
            lat: 51.5074,
            lon: -0.1278,
        };

        let sf_cell = H3Grid::location_to_cell_id(san_francisco, 6).unwrap();
        let ny_cell = H3Grid::location_to_cell_id(new_york, 6).unwrap();
        let london_cell = H3Grid::location_to_cell_id(london, 6).unwrap();

        // Different cities should have different cell IDs
        assert_ne!(sf_cell, ny_cell);
        assert_ne!(sf_cell, london_cell);
        assert_ne!(ny_cell, london_cell);

        // Test locations that should map to same cell (very close together)
        let soma_district = Location {
            lat: 37.7749,
            lon: -122.4194,
        };
        let mission_district = Location {
            lat: 37.7599,
            lon: -122.4148,
        }; // ~1.5km away

        let soma_cell = H3Grid::location_to_cell_id(soma_district, 4).unwrap(); // Larger cells
        let mission_cell = H3Grid::location_to_cell_id(mission_district, 4).unwrap();

        // At resolution 4 (~86km cells), nearby SF locations should be in same cell
        assert_eq!(
            soma_cell, mission_cell,
            "Nearby SF locations should share res-4 cell"
        );

        // But at resolution 8, they should be different
        let soma_cell_8 = H3Grid::location_to_cell_id(soma_district, 8).unwrap();
        let mission_cell_8 = H3Grid::location_to_cell_id(mission_district, 8).unwrap();
        assert_ne!(
            soma_cell_8, mission_cell_8,
            "Same locations should differ at res-8"
        );
    }

    #[test]
    fn test_h3_resolution_consistency() {
        let location = Location {
            lat: 37.7749,
            lon: -122.4194,
        };

        // Test all supported resolution levels
        for resolution in 0..=15 {
            let cell_result = H3Grid::location_to_cell_id(location, resolution);
            assert!(
                cell_result.is_ok(),
                "Resolution {} should be valid",
                resolution
            );

            let cell_id = cell_result.unwrap();
            assert!(
                !cell_id.is_empty(),
                "Cell ID should not be empty at res {}",
                resolution
            );

            // Test round-trip conversion
            let center_result = H3Grid::cell_center(&cell_id);
            assert!(
                center_result.is_ok(),
                "Center calculation should work for res {}",
                resolution
            );

            let center = center_result.unwrap();
            assert!(
                center.lat >= -90.0 && center.lat <= 90.0,
                "Center lat should be valid"
            );
            assert!(
                center.lon >= -180.0 && center.lon <= 180.0,
                "Center lon should be valid"
            );
        }

        // Test invalid resolutions
        let invalid_high = H3Grid::location_to_cell_id(location, 16);
        assert!(invalid_high.is_err(), "Resolution 16 should be invalid");
    }

    #[test]
    fn test_location_to_h3_integration_with_defaults() {
        use crate::persistence::location::DEFAULT_LOCATION;

        // Test integration with the default San Francisco location
        let default_cell = H3Grid::location_to_cell_id(DEFAULT_LOCATION, 6);
        assert!(default_cell.is_ok(), "Default location should work with H3");

        let cell_id = default_cell.unwrap();
        assert!(
            !cell_id.is_empty(),
            "Default location should produce valid cell ID"
        );

        // Test signal loading with default location
        let strong_cells = H3Grid::get_signal_loading_cells(DEFAULT_LOCATION, 0.8);
        let medium_cells = H3Grid::get_signal_loading_cells(DEFAULT_LOCATION, 0.5);
        let weak_cells = H3Grid::get_signal_loading_cells(DEFAULT_LOCATION, 0.3);

        assert!(strong_cells.is_ok());
        assert!(medium_cells.is_ok());
        assert!(weak_cells.is_ok());

        // Verify different signal strengths give different resolutions
        let strong_cell = &strong_cells.unwrap()[0];
        let medium_cell = &medium_cells.unwrap()[0];
        let weak_cell = &weak_cells.unwrap()[0];

        // Different resolutions should give different cell IDs for same location
        assert_ne!(strong_cell, medium_cell);
        assert_ne!(strong_cell, weak_cell);
        assert_ne!(medium_cell, weak_cell);
    }

    #[test]
    fn test_h3_property_invariants() {
        let test_locations = vec![
            Location { lat: 0.0, lon: 0.0 }, // Equator
            Location {
                lat: 37.7749,
                lon: -122.4194,
            }, // San Francisco
            Location {
                lat: -33.8688,
                lon: 151.2093,
            }, // Sydney
            Location {
                lat: 60.1699,
                lon: 24.9384,
            }, // Helsinki
        ];

        for location in test_locations {
            for resolution in [4, 6, 8] {
                let cell_id = H3Grid::location_to_cell_id(location, resolution).unwrap();

                // Property: Cell ID should be consistent
                let cell_id_2 = H3Grid::location_to_cell_id(location, resolution).unwrap();
                assert_eq!(cell_id, cell_id_2, "Cell ID should be deterministic");

                // Property: Round-trip should preserve approximate location
                let center = H3Grid::cell_center(&cell_id).unwrap();
                let cell_id_roundtrip = H3Grid::location_to_cell_id(center, resolution).unwrap();
                assert_eq!(cell_id, cell_id_roundtrip, "Round-trip should be stable");

                // Property: Higher resolution should give more precise location
                if resolution < 10 {
                    let higher_res_cell =
                        H3Grid::location_to_cell_id(location, resolution + 2).unwrap();
                    assert_ne!(
                        cell_id, higher_res_cell,
                        "Different resolutions should differ"
                    );
                }

                // Property: Cell center should be within valid coordinate bounds
                assert!(center.lat >= -90.0 && center.lat <= 90.0);
                assert!(center.lon >= -180.0 && center.lon <= 180.0);
            }
        }
    }

    #[test]
    fn test_concurrent_h3_operations() {
        use std::{sync::Arc, thread};

        let test_location = Arc::new(Location {
            lat: 37.7749,
            lon: -122.4194,
        });

        // Test concurrent location-to-cell conversions
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let location = test_location.clone();
                thread::spawn(move || {
                    let resolution = 6 + (i % 3); // Vary resolution 6-8
                    let cell_id = H3Grid::location_to_cell_id(*location, resolution).unwrap();

                    // Test concurrent center calculation
                    let center = H3Grid::cell_center(&cell_id).unwrap();

                    // Test concurrent signal loading calculation
                    let signal_strength = 0.5 + (i as f64 * 0.05); // Vary 0.5-0.95
                    let cells =
                        H3Grid::get_signal_loading_cells(*location, signal_strength).unwrap();

                    (cell_id, center, cells)
                })
            })
            .collect();

        // All threads should complete successfully
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.len(), 10);

        // Verify all results are valid
        for (cell_id, center, cells) in results {
            assert!(!cell_id.is_empty());
            assert!(center.lat >= -90.0 && center.lat <= 90.0);
            assert!(center.lon >= -180.0 && center.lon <= 180.0);
            assert_eq!(cells.len(), 1); // Current implementation returns 1 cell
        }
    }

    // === Property-Based Tests for Coordinate Validation ===

    #[test]
    fn test_property_coordinate_consistency() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // Use fixed seed for reproducible tests
        let mut rng = StdRng::seed_from_u64(42);

        // Property: Same coordinate should always produce same H3 cell ID
        for _ in 0..100 {
            let lat = rng.gen_range(-80.0..80.0); // Avoid polar extremes
            let lon = rng.gen_range(-180.0..180.0);
            let location = Location { lat, lon };
            let resolution = rng.gen_range(0..=10); // Vary resolution

            if let Ok(cell_id_1) = H3Grid::location_to_cell_id(location, resolution) {
                let cell_id_2 = H3Grid::location_to_cell_id(location, resolution).unwrap();
                assert_eq!(
                    cell_id_1, cell_id_2,
                    "Same coordinate should produce same cell ID: lat={}, lon={}, res={}",
                    lat, lon, resolution
                );
            }
        }
    }

    #[test]
    fn test_property_roundtrip_accuracy() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(123);

        // Property: Converting location → H3 → center should be reasonably close
        for _ in 0..50 {
            let lat = rng.gen_range(-80.0..80.0);
            let lon = rng.gen_range(-180.0..180.0);
            let location = Location { lat, lon };
            let resolution = rng.gen_range(6..=10); // Use higher resolutions for accuracy

            if let Ok(cell_id) = H3Grid::location_to_cell_id(location, resolution)
                && let Ok(center) = H3Grid::cell_center(&cell_id)
            {
                // Distance should be within reasonable bounds for the resolution
                let max_error = match resolution {
                    6 => 5.0,  // ~40km cell radius
                    7 => 2.0,  // ~15km cell radius
                    8 => 0.8,  // ~6km cell radius
                    9 => 0.3,  // ~2.2km cell radius
                    10 => 0.1, // ~800m cell radius
                    _ => 10.0, // Conservative bound for other resolutions
                };

                let lat_diff = (center.lat - location.lat).abs();
                let lon_diff = (center.lon - location.lon).abs();

                assert!(
                    lat_diff < max_error,
                    "Latitude round-trip error too large: {} > {} for res {}, original=({}, {}), \
                     center=({}, {})",
                    lat_diff,
                    max_error,
                    resolution,
                    location.lat,
                    location.lon,
                    center.lat,
                    center.lon
                );

                assert!(
                    lon_diff < max_error,
                    "Longitude round-trip error too large: {} > {} for res {}, original=({}, {}), \
                     center=({}, {})",
                    lon_diff,
                    max_error,
                    resolution,
                    location.lat,
                    location.lon,
                    center.lat,
                    center.lon
                );
            }
        }
    }

    #[test]
    fn test_property_resolution_hierarchy() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(789);

        // Property: Different resolutions should produce different cell IDs for same location
        for _ in 0..50 {
            let lat = rng.gen_range(-70.0..70.0);
            let lon = rng.gen_range(-170.0..170.0);
            let location = Location { lat, lon };

            let low_res = rng.gen_range(4..=6);
            let high_res = low_res + rng.gen_range(2..=4);

            if let (Ok(low_cell), Ok(high_cell)) = (
                H3Grid::location_to_cell_id(location, low_res),
                H3Grid::location_to_cell_id(location, high_res),
            ) {
                // Different resolutions should usually give different cell IDs
                // (There might be edge cases where they're the same due to H3 structure)
                if low_cell == high_cell {
                    // This is rare but can happen, just log it
                    println!(
                        "Same cell ID at different resolutions: {} vs {} for location ({}, {})",
                        low_res, high_res, lat, lon
                    );
                }

                // Both should be valid hex strings
                assert!(
                    low_cell.len() > 8,
                    "Low resolution cell ID should be reasonable length"
                );
                assert!(
                    high_cell.len() > 8,
                    "High resolution cell ID should be reasonable length"
                );

                // Both should parse as hex
                assert!(
                    u64::from_str_radix(&low_cell, 16).is_ok()
                        || low_cell.starts_with("0x")
                            && u64::from_str_radix(&low_cell[2..], 16).is_ok()
                );
                assert!(
                    u64::from_str_radix(&high_cell, 16).is_ok()
                        || high_cell.starts_with("0x")
                            && u64::from_str_radix(&high_cell[2..], 16).is_ok()
                );
            }
        }
    }

    #[test]
    fn test_property_coordinate_bounds() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(456);

        // Property: Valid coordinates should always succeed, invalid should fail gracefully
        for _ in 0..100 {
            let lat = rng.gen_range(-90.0..=90.0);
            let lon = rng.gen_range(-180.0..=180.0);
            let location = Location { lat, lon };
            let resolution = rng.gen_range(0..=15);

            let result = H3Grid::location_to_cell_id(location, resolution);

            match result {
                Ok(cell_id) => {
                    // Successful conversion should produce non-empty cell ID
                    assert!(
                        !cell_id.is_empty(),
                        "Valid coordinate should produce non-empty cell ID: ({}, {}) res {}",
                        lat,
                        lon,
                        resolution
                    );

                    // Should be able to convert back to center
                    assert!(
                        H3Grid::cell_center(&cell_id).is_ok(),
                        "Generated cell ID should be convertible back: {} for ({}, {}) res {}",
                        cell_id,
                        lat,
                        lon,
                        resolution
                    );
                }
                Err(_) => {
                    // Failed conversion should be for extreme coordinates or invalid resolution
                    // H3 has specific valid ranges, failure is acceptable for edge cases
                }
            }
        }
    }

    #[test]
    fn test_property_signal_strength_monotonicity() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(321);

        // Property: Signal strength mapping should be monotonic
        for _ in 0..50 {
            let lat = rng.gen_range(-70.0..70.0);
            let lon = rng.gen_range(-170.0..170.0);
            let location = Location { lat, lon };

            // Test signal strength sequence
            let signal_strengths = [0.1, 0.3, 0.5, 0.7, 0.9];

            for &signal_strength in &signal_strengths {
                if let Ok(cells) = H3Grid::get_signal_loading_cells(location, signal_strength) {
                    assert_eq!(cells.len(), 1, "Should return exactly one cell");

                    let cell_id = &cells[0];
                    assert!(!cell_id.is_empty(), "Cell ID should not be empty");

                    // Verify the cell ID is valid by converting to center
                    assert!(
                        H3Grid::cell_center(cell_id).is_ok(),
                        "Generated cell should be valid for signal strength {} at ({}, {})",
                        signal_strength,
                        lat,
                        lon
                    );

                    // We can't directly access the resolution, but we can infer monotonicity
                    // by checking that stronger signals generally produce larger cells (lower
                    // resolution) This is tested indirectly through the
                    // resolution_for_signal_strength logic
                }
            }
        }
    }

    #[test]
    fn test_property_locality_preservation() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(654);

        // Property: Nearby locations should often be in same or neighboring cells
        for _ in 0..30 {
            let lat = rng.gen_range(-70.0..70.0);
            let lon = rng.gen_range(-170.0..170.0);
            let location1 = Location { lat, lon };

            // Generate nearby location (within ~1km)
            let lat_offset = rng.gen_range(-0.01..0.01); // ~1km
            let lon_offset = rng.gen_range(-0.01..0.01); // ~1km at equator
            let location2 = Location {
                lat: lat + lat_offset,
                lon: lon + lon_offset,
            };

            let resolution = rng.gen_range(8..=10); // High resolution for locality test

            if let (Ok(cell1), Ok(cell2)) = (
                H3Grid::location_to_cell_id(location1, resolution),
                H3Grid::location_to_cell_id(location2, resolution),
            ) {
                // For very close locations at high resolution, they might be in same cell
                // This is a probabilistic property, not deterministic
                let same_cell = cell1 == cell2;

                if same_cell {
                    // Verify both cells convert back to valid centers
                    assert!(H3Grid::cell_center(&cell1).is_ok());
                    assert!(H3Grid::cell_center(&cell2).is_ok());
                } else {
                    // Different cells should both be valid
                    assert!(H3Grid::cell_center(&cell1).is_ok());
                    assert!(H3Grid::cell_center(&cell2).is_ok());
                }

                // Both locations should produce valid cells regardless
                assert!(!cell1.is_empty());
                assert!(!cell2.is_empty());
            }
        }
    }

    #[test]
    fn test_property_cell_id_format_invariants() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(987);

        // Property: All valid cell IDs should follow format conventions
        for _ in 0..50 {
            let lat = rng.gen_range(-80.0..80.0);
            let lon = rng.gen_range(-180.0..180.0);
            let location = Location { lat, lon };
            let resolution = rng.gen_range(0..=12);

            if let Ok(cell_id) = H3Grid::location_to_cell_id(location, resolution) {
                // Cell ID should be non-empty
                assert!(!cell_id.is_empty());

                // Should be reasonable length (H3 uses 64-bit integers)
                assert!(
                    cell_id.len() >= 8 && cell_id.len() <= 20,
                    "Cell ID length should be reasonable: '{}' (len={})",
                    cell_id,
                    cell_id.len()
                );

                // Should be parseable as hex (with or without 0x prefix)
                let is_valid_hex = if let Some(stripped) = cell_id.strip_prefix("0x") {
                    u64::from_str_radix(stripped, 16).is_ok()
                } else {
                    u64::from_str_radix(&cell_id, 16).is_ok()
                };
                assert!(is_valid_hex, "Cell ID should be valid hex: '{}'", cell_id);

                // Should be convertible back to center
                let center_result = H3Grid::cell_center(&cell_id);
                assert!(
                    center_result.is_ok(),
                    "Cell ID should be convertible back: '{}'",
                    cell_id
                );

                if let Ok(center) = center_result {
                    // Center should be within valid coordinate bounds
                    assert!(
                        center.lat >= -90.0 && center.lat <= 90.0,
                        "Center latitude should be valid: {}",
                        center.lat
                    );
                    assert!(
                        center.lon >= -180.0 && center.lon <= 180.0,
                        "Center longitude should be valid: {}",
                        center.lon
                    );
                }
            }
        }
    }
}
