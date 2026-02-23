fn main() {
    println!("cargo:warning=Running build.rs...");

    let has_gpu_feature = std::env::var("CARGO_FEATURE_GPU").is_ok();
    let vpi_path_3 = "/opt/nvidia/vpi3";
    let vpi_path_2 = "/opt/nvidia/vpi2";
    let has_vpi = std::path::Path::new(vpi_path_3).exists() || std::path::Path::new(vpi_path_2).exists();

    println!("cargo:warning=GPU Feature Enabled: {}", has_gpu_feature);
    println!("cargo:warning=VPI Detected on Host: {}", has_vpi);

    if has_gpu_feature {
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

    let has_tensorrt_feature = std::env::var("CARGO_FEATURE_TENSORRT").is_ok();
    if has_tensorrt_feature {
        println!("cargo:warning=TensorRT feature enabled, linking libraries...");

        let trt_paths = ["/usr/lib/aarch64-linux-gnu", "/opt/nvidia/tensorrt/lib"];
        let mut found_trt = false;

        for path in trt_paths.iter() {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
                found_trt = true;
            }
        }

        let mut build = cc::Build::new();
        build.cpp(true).file("src/yolo_trt.cpp");

        let include_paths = [
            "/usr/include/aarch64-linux-gnu",
            "/usr/include",
            "/usr/local/cuda/include",
        ];
        for path in include_paths.iter() {
            if std::path::Path::new(path).exists() {
                build.include(path);
            }
        }

        build.flag_if_supported("-std=c++14");
        build.compile("yolo_trt");

        if found_trt {
            println!("cargo:rustc-link-lib=nvinfer");
            println!("cargo:rustc-link-lib=nvinfer_plugin");
            println!("cargo:rustc-link-lib=cudart");
        } else {
            println!("cargo:warning=TensorRT libraries not found in standard locations.");
        }

        let cuda_paths = ["/usr/local/cuda/lib64", "/usr/lib/aarch64-linux-gnu"];
        for path in cuda_paths.iter() {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
                break;
            }
        }

        let nvparsers_paths = [
            "/usr/lib/aarch64-linux-gnu/libnvparsers.so",
            "/opt/nvidia/tensorrt/lib/libnvparsers.so",
        ];
        if nvparsers_paths.iter().any(|p| std::path::Path::new(p).exists()) {
            println!("cargo:rustc-link-lib=nvparsers");
        } else {
            println!("cargo:warning=libnvparsers.so not found; skipping link.");
        }
    }
    
    println!("cargo:rerun-if-changed=src/vpi_helper.c");
    println!("cargo:rerun-if-changed=src/yolo_trt.cpp");
    println!("cargo:rerun-if-changed=src/yolo_trt.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TENSORRT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPU");
}
