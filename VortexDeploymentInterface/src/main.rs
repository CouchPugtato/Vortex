#![windows_subsystem = "windows"]

use eframe::egui;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use walkdir::WalkDir;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Vortex Deployment Tool",
        options,
        Box::new(|_cc| Box::new(DeploymentApp::default())),
    )
}

#[derive(Serialize, Deserialize)]
struct AppConfig {
    host: String,
    port: String,
    user: String,
    pass: String,
    remote_path: String,
    local_path: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct CameraConfig {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    tag_size_m: f64,
    #[serde(default)]
    x_offset: f64,
    #[serde(default)]
    y_offset: f64,
    #[serde(default)]
    z_offset: f64,
    #[serde(default)]
    pitch_deg: f64,
    #[serde(default)]
    yaw_deg: f64,
    #[serde(default)]
    roll_deg: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ProcessingConfig {
    #[serde(default = "default_smoothing_alpha")]
    smoothing_alpha: f64,
    #[serde(
        default = "default_resolution_scale_factor",
        alias = "gpu_scale_factor"
    )]
    resolution_scale_factor: f32,
    #[serde(default = "default_yolo_obj_width_m")]
    yolo_obj_width_m: f64,
    #[serde(default = "default_yolo_obj_height_m")]
    yolo_obj_height_m: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct RuntimeConfig {
    camera: CameraConfig,
    #[serde(default)]
    processing: ProcessingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            host: "192.168.55.1".to_owned(),
            port: "22".to_owned(),
            user: "jetson".to_owned(),
            pass: "".to_owned(),
            remote_path: "/home/jetson/deployments".to_owned(),
            local_path: "".to_owned(),
        }
    }
}

struct DeploymentApp {
    config: AppConfig,
    runtime_config: RuntimeConfig,
    runtime_config_path: String,
    logs: String,
    is_deploying: bool,
    log_receiver: Receiver<String>,
    log_sender: Sender<String>,
}

impl Default for DeploymentApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        
        let config = if let Ok(mut file) = std::fs::File::open("vortex_config.json") {
            let mut data = String::new();
            if file.read_to_string(&mut data).is_ok() {
                serde_json::from_str(&data).unwrap_or_default()
            } else {
                AppConfig::default()
            }
        } else {
            AppConfig::default()
        };

        let runtime_config_path = default_runtime_config_path();
        let runtime_config = load_runtime_config(Path::new(&runtime_config_path))
            .unwrap_or_default();

        Self {
            config,
            runtime_config,
            runtime_config_path,
            logs: "Ready...\n".to_owned(),
            is_deploying: false,
            log_receiver: rx,
            log_sender: tx,
        }
    }
}

impl eframe::App for DeploymentApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.log_receiver.try_recv() {
            self.logs.push_str(&msg);
            self.logs.push('\n');
            if msg.contains("Deployment Finished") || msg.contains("Deployment Failed") {
                self.is_deploying = false;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Vortex Deployment Tool");
            ui.add_space(10.0);

            ui.group(|ui| {
                ui.label("Connection Details");
                egui::Grid::new("conn_grid").striped(true).show(ui, |ui| {
                    ui.label("Host:");
                    ui.add(egui::TextEdit::singleline(&mut self.config.host).desired_width(600.0));
                    ui.end_row();

                    ui.label("User:");
                    ui.add(egui::TextEdit::singleline(&mut self.config.user).desired_width(600.0));
                    ui.end_row();

                    ui.label("Password:");
                    ui.add(egui::TextEdit::singleline(&mut self.config.pass).password(true).desired_width(600.0));
                    ui.end_row();

                    ui.label("Remote Path:");
                    ui.add(egui::TextEdit::singleline(&mut self.config.remote_path).desired_width(600.0));
                    ui.end_row();
                });
            });

            ui.add_space(10.0);

            ui.group(|ui| {
                ui.label("Selection");
                ui.horizontal(|ui| {
                    ui.label("Local Folder:");
                    ui.add(egui::TextEdit::singleline(&mut self.config.local_path).desired_width(550.0));
                    if ui.button("Browse").clicked() {
                        if let Some(path) = FileDialog::new().pick_folder() {
                            self.config.local_path = path.display().to_string();
                            let discovered = runtime_config_path_for_local(&self.config.local_path);
                            self.runtime_config_path = discovered.display().to_string();
                            match load_runtime_config(&discovered) {
                                Ok(cfg) => {
                                    self.runtime_config = cfg;
                                    self.logs.push_str(&format!(
                                        "Loaded runtime config: {}\n",
                                        self.runtime_config_path
                                    ));
                                }
                                Err(e) => {
                                    self.logs.push_str(&format!(
                                        "Runtime config not loaded ({}): {}\n",
                                        self.runtime_config_path, e
                                    ));
                                }
                            }
                        }
                    }
                });
            });

            ui.add_space(20.0);

            ui.group(|ui| {
                ui.label("Runtime Config (config/config.json)");
                ui.horizontal(|ui| {
                    ui.label("Config File:");
                    ui.add(egui::TextEdit::singleline(&mut self.runtime_config_path).desired_width(500.0));

                    if ui.button("Load").clicked() {
                        let path = PathBuf::from(self.runtime_config_path.clone());
                        match load_runtime_config(&path) {
                            Ok(cfg) => {
                                self.runtime_config = cfg;
                                self.logs.push_str(&format!("Loaded runtime config: {}\n", path.display()));
                            }
                            Err(e) => {
                                self.logs.push_str(&format!("Failed to load runtime config: {}\n", e));
                            }
                        }
                    }

                    if ui.button("Save").clicked() {
                        let path = PathBuf::from(self.runtime_config_path.clone());
                        match save_runtime_config(&path, &self.runtime_config) {
                            Ok(()) => {
                                self.logs.push_str(&format!("Saved runtime config: {}\n", path.display()));
                            }
                            Err(e) => {
                                self.logs.push_str(&format!("Failed to save runtime config: {}\n", e));
                            }
                        }
                    }
                });

                ui.add_space(8.0);
                ui.label("Camera");
                egui::Grid::new("runtime_camera_grid").striped(true).show(ui, |ui| {
                    ui.label("fx"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.fx).speed(0.1)); ui.end_row();
                    ui.label("fy"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.fy).speed(0.1)); ui.end_row();
                    ui.label("cx"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.cx).speed(0.1)); ui.end_row();
                    ui.label("cy"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.cy).speed(0.1)); ui.end_row();
                    ui.label("k1"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.k1).speed(0.0001)); ui.end_row();
                    ui.label("k2"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.k2).speed(0.0001)); ui.end_row();
                    ui.label("p1"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.p1).speed(0.0001)); ui.end_row();
                    ui.label("p2"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.p2).speed(0.0001)); ui.end_row();
                    ui.label("k3"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.k3).speed(0.0001)); ui.end_row();
                    ui.label("tag_size_m"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.tag_size_m).speed(0.001)); ui.end_row();
                    ui.label("x_offset"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.x_offset).speed(0.001)); ui.end_row();
                    ui.label("y_offset"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.y_offset).speed(0.001)); ui.end_row();
                    ui.label("z_offset"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.z_offset).speed(0.001)); ui.end_row();
                    ui.label("pitch_deg"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.pitch_deg).speed(0.1)); ui.end_row();
                    ui.label("yaw_deg"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.yaw_deg).speed(0.1)); ui.end_row();
                    ui.label("roll_deg"); ui.add(egui::DragValue::new(&mut self.runtime_config.camera.roll_deg).speed(0.1)); ui.end_row();
                });

                ui.add_space(8.0);
                ui.label("Processing");
                egui::Grid::new("runtime_processing_grid").striped(true).show(ui, |ui| {
                    ui.label("smoothing_alpha"); ui.add(egui::DragValue::new(&mut self.runtime_config.processing.smoothing_alpha).speed(0.01).clamp_range(0.0..=1.0)); ui.end_row();
                    ui.label("resolution_scale_factor"); ui.add(egui::DragValue::new(&mut self.runtime_config.processing.resolution_scale_factor).speed(0.01).clamp_range(0.1..=1.0)); ui.end_row();
                    ui.label("yolo_obj_width_m"); ui.add(egui::DragValue::new(&mut self.runtime_config.processing.yolo_obj_width_m).speed(0.01)); ui.end_row();
                    ui.label("yolo_obj_height_m"); ui.add(egui::DragValue::new(&mut self.runtime_config.processing.yolo_obj_height_m).speed(0.01)); ui.end_row();
                });
            });

            ui.add_space(10.0);

            if self.is_deploying {
                ui.add(egui::Spinner::new());
                ui.label("Deploying...");
            } else {
                if ui.button("Deploy to Jetson").clicked() {
                    if self.config.local_path.is_empty() {
                        self.logs.push_str("Error: No local folder selected.\n");
                    } else {
                        if self.runtime_config_path.trim().is_empty() {
                            let discovered = runtime_config_path_for_local(&self.config.local_path);
                            self.runtime_config_path = discovered.display().to_string();
                        }
                        let runtime_path = PathBuf::from(self.runtime_config_path.clone());
                        if let Err(e) = save_runtime_config(&runtime_path, &self.runtime_config) {
                            self.logs.push_str(&format!("Failed to save runtime config before deploy: {}\n", e));
                            return;
                        }

                        self.is_deploying = true;
                        self.logs.push_str("Starting deployment...\n");
                        
                        let host = self.config.host.clone();
                        let port = "22".to_owned();
                        let user = self.config.user.clone();
                        let pass = self.config.pass.clone();
                        let local = self.config.local_path.clone();
                        let remote = self.config.remote_path.clone();
                        let tx = self.log_sender.clone();
                        let ctx_clone = ctx.clone();

                        thread::spawn(move || {
                            match deploy(host, port, user, pass, local, remote, &tx) {
                                Ok(_) => {
                                    let _ = tx.send("Deployment Finished Successfully!".to_owned());
                                }
                                Err(e) => {
                                    let _ = tx.send(format!("Deployment Failed: {}", e));
                                }
                            }
                            ctx_clone.request_repaint();
                        });
                    }
                }
            }

            ui.add_space(10.0);
            ui.separator();
            ui.label("Logs:");
            
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.logs)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                        .lock_focus(true)
                );
            });
        });
    }
}

fn deploy(
    host: String,
    port: String,
    user: String,
    pass: String,
    local_path: String,
    remote_base: String,
    tx: &Sender<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tx.send(format!("Checking connection to {}:{}...", host, port));
    
    // check TCP connectivity first with timeout
    let addr = format!("{}:{}", host, port);
    if let Ok(socket_addr) = addr.parse::<std::net::SocketAddr>() {
        match TcpStream::connect_timeout(&socket_addr, std::time::Duration::from_secs(3)) {
            Ok(_) => {
                let _ = tx.send("Connection established. Authenticating...".to_owned());
            },
            Err(e) => {
                return Err(format!("Could not connect to {}. Check USB connection. Error: {}", addr, e).into());
            }
        }
    } else {
        // fallback for non-IP hosts
        if TcpStream::connect(&addr).is_err() {
             return Err(format!("Could not connect to {}. Check USB connection.", addr).into());
        }
    }

    let tcp = TcpStream::connect(format!("{}:{}", host, port))?;
    let mut sess = ssh2::Session::new()?;
    sess.set_tcp_stream(tcp);
    sess.handshake()?;

    let _ = tx.send("Authenticating...".to_owned());
    sess.userauth_password(&user, &pass)?;

    if !sess.authenticated() {
        return Err("Authentication failed".into());
    }

    let _ = tx.send("Connected. Initializing SFTP...".to_owned());
    let sftp = sess.sftp()?;

    let local_path_buf = Path::new(&local_path);
    let folder_name = local_path_buf.file_name()
        .ok_or("Invalid local path")?
        .to_str()
        .ok_or("Invalid characters in path")?;

    let remote_target = Path::new(&remote_base).join(folder_name);
    let remote_target_str = remote_target.to_str().ok_or("Invalid remote path")?.replace("\\", "/");

    let _ = tx.send(format!("Creating remote directory: {}", remote_target_str));
    
    // mkdir -p style creation
    match sess.channel_session() {
        Ok(mut channel) => {
            if let Err(e) = channel.exec(&format!("mkdir -p '{}'", remote_target_str)) {
                 return Err(format!("Failed to execute mkdir command: {}", e).into());
            }
            let mut s = String::new();
            let _ = channel.read_to_string(&mut s); 
            let _ = channel.wait_close();
            
            if let Ok(status) = channel.exit_status() {
                if status != 0 {
                    return Err(format!("Failed to create remote directory. Exit code: {}", status).into());
                }
            }
        },
        Err(e) => {
             return Err(format!("Failed to open SSH channel for mkdir: {}", e).into());
        }
    }

    for entry in WalkDir::new(&local_path).into_iter().filter_entry(|e| !is_target_dir(e.path())) {
        let entry = entry?;
        let path = entry.path();

        if has_target_component(path) {
            continue;
        }
        
        let relative_path = path.strip_prefix(&local_path)?;
        let relative_path_str = relative_path.to_str().ok_or("Invalid path")?.replace("\\", "/");
        
        if relative_path_str.is_empty() { continue; }

        let remote_file_path = Path::new(&remote_target_str).join(relative_path);
        let remote_file_path_str = remote_file_path.to_str().ok_or("Invalid remote path")?.replace("\\", "/");

        if path.is_dir() {
             if sftp.stat(Path::new(&remote_file_path_str)).is_err() {
                 let _ = sftp.mkdir(Path::new(&remote_file_path_str), 0o755);
             }
        } else {
            let _ = tx.send(format!("Uploading: {}", relative_path_str));
            let mut local_file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(e) => return Err(format!("Failed to open local file {}: {}", path.display(), e).into()),
            };
            
            let mut remote_file = match sftp.create(Path::new(&remote_file_path_str)) {
                Ok(f) => f,
                Err(e) => return Err(format!("Failed to create remote file {}: {}", remote_file_path_str, e).into()),
            };
            
            if let Err(e) = std::io::copy(&mut local_file, &mut remote_file) {
                 return Err(format!("Failed to upload content for {}: {}", relative_path_str, e).into());
            }
        }
    }

    Ok(())
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            fx: 709.5,
            fy: 709.5,
            cx: 960.0,
            cy: 600.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
            tag_size_m: 0.16,
            x_offset: 0.0,
            y_offset: 0.0,
            z_offset: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            roll_deg: 0.0,
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            smoothing_alpha: default_smoothing_alpha(),
            resolution_scale_factor: default_resolution_scale_factor(),
            yolo_obj_width_m: default_yolo_obj_width_m(),
            yolo_obj_height_m: default_yolo_obj_height_m(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            processing: ProcessingConfig::default(),
        }
    }
}

const fn default_smoothing_alpha() -> f64 { 0.1 }
const fn default_resolution_scale_factor() -> f32 { 1.0 }
const fn default_yolo_obj_width_m() -> f64 { 0.30 }
const fn default_yolo_obj_height_m() -> f64 { 0.30 }

fn runtime_config_path_for_local(local_path: &str) -> PathBuf {
    Path::new(local_path).join("config").join("config.json")
}

fn default_runtime_config_path() -> String {
    let local = PathBuf::from("config").join("config.json");
    if local.exists() {
        return local.display().to_string();
    }

    let parent = PathBuf::from("..").join("config").join("config.json");
    if parent.exists() {
        return parent.display().to_string();
    }

    local.display().to_string()
}

fn load_runtime_config(path: &Path) -> Result<RuntimeConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("{} ({})", path.display(), e))?;
    serde_json::from_str(&data).map_err(|e| format!("{} ({})", path.display(), e))
}

fn save_runtime_config(path: &Path, config: &RuntimeConfig) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("{} ({})", parent.display(), e))?;
    }

    let json = serde_json::to_string_pretty(config)
        .map_err(|e| e.to_string())?;
    fs::write(path, json)
        .map_err(|e| format!("{} ({})", path.display(), e))
}

fn has_target_component(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == "target")
}

fn is_target_dir(path: &Path) -> bool {
    path.file_name().is_some_and(|name| name == "target")
}
