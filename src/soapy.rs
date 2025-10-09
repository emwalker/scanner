/// Reset SoapySDR module state by unloading and reloading all modules.
/// This clears any stale mutex locks from previous runs.
pub fn reset_soapysdr_state() {
    unsafe {
        soapysdr_sys::SoapySDR_unloadModules();
        soapysdr_sys::SoapySDR_loadModules();
    }
}

/// Cleanup SoapySDR state on shutdown by unloading all modules.
pub fn cleanup_soapysdr_state() {
    unsafe {
        soapysdr_sys::SoapySDR_unloadModules();
    }
}
