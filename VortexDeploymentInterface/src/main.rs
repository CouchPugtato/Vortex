#![windows_subsystem = "windows"]

use eframe::egui;
use rfd::FileDialog;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use walkdir::WalkDir;
use serde::{Deserialize, Serialize};

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

        Self {
            config,
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
                        }
                    }
                });
            });

            ui.add_space(20.0);

            if self.is_deploying {
                ui.add(egui::Spinner::new());
                ui.label("Deploying...");
            } else {
                if ui.button("Deploy to Jetson").clicked() {
                    if self.config.local_path.is_empty() {
                        self.logs.push_str("Error: No local folder selected.\n");
                    } else {
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

    for entry in WalkDir::new(&local_path) {
        let entry = entry?;
        let path = entry.path();
        
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
