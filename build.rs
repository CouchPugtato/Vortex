fn main() {
    println!("cargo:warning=Running build.rs...");

    let has_gpu_feature = std::env::var("CARGO_FEATURE_GPU").is_ok();
    let vpi_path_3 = "/opt/nvidia/vpi3";
    let vpi_path_2 = "/opt/nvidia/vpi2";
    let has_vpi = std::path::Path::new(vpi_path_3).exists() || std::path::Path::new(vpi_path_2).exists();

    println!("cargo:warning=GPU Feature Enabled: {}", has_gpu_feature);
    println!("cargo:warning=VPI Detected on Host: {}", has_vpi);

    if has_gpu_feature || has_vpi {
        println!("cargo:warning=Enabling VPI compilation...");
        
        // determine include path
        let include_path = if std::path::Path::new(vpi_path_3).exists() {
            format!("{}/include", vpi_path_3)
        } else {
            format!("{}/include", vpi_path_2)
        };
        
        // determine lib path
        let lib_path = if std::path::Path::new(vpi_path_3).exists() {
             format!("{}/lib64", vpi_path_3)
        } else {
             format!("{}/lib64", vpi_path_2)
        };

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=nvvpi");
        
    } else {
        println!("cargo:warning=No GPU feature or VPI libraries found. Skipping VPI setup.");
    }
    
    println!("cargo:rerun-if-changed=src/vpi_helper.c");
    println!("cargo:rerun-if-changed=build.rs");
}
