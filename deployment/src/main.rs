use anyhow::{ensure, Context, Result};
use clap::Parser;
use common::*;
use serde_json::{json, Value};
use std::path::PathBuf;
use vortexctl::{
    activation, common, initialize, jobs, launcher, offline, package, retention, service,
    validation,
};

#[derive(Parser, Debug)]
#[command(version, about = "Native Vortex deployment controller")]
struct Args {
    #[arg(value_parser=["launch","pack","stage","start-job","status","activate","rollback","recover","service-run","install-service","preflight","diagnostics","diagnostics-stop","initialize-submit","initialize-run","cleanup","offline-create","offline-download","validate-device"])]
    action: String,
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    source: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    controller: Option<PathBuf>,
    #[arg(long)]
    device_config: Option<PathBuf>,
    #[arg(long)]
    runtime_config: Option<PathBuf>,
    #[arg(long)]
    tag_map: Option<PathBuf>,
    #[arg(long)]
    archive: Option<PathBuf>,
    #[arg(long)]
    sha256: Option<String>,
    #[arg(long)]
    release_id: Option<String>,
    #[arg(long)]
    job_id: Option<String>,
    #[arg(long)]
    kind: Option<String>,
    #[arg(long)]
    operation_id: Option<String>,
    #[arg(long)]
    legacy_root: Option<PathBuf>,
    #[arg(long, default_value = "vortex-vision.service")]
    service: String,
    #[arg(long, default_value = "vortex")]
    user: String,
    #[arg(long)]
    client_id: Option<String>,
    #[arg(long)]
    build: bool,
    #[arg(long)]
    online: bool,
    #[arg(long)]
    with_debug_bridge: bool,
    #[arg(long)]
    build_cache: Option<PathBuf>,
    #[arg(long)]
    validate_config: bool,
    #[arg(long)]
    check: bool,
    #[arg(long)]
    check_debug_bridge: bool,
    #[arg(long)]
    debug_bridge: bool,
    #[arg(long, default_value = "0")]
    camera: u64,
    #[arg(long, default_value = "3")]
    keep_releases: usize,
    #[arg(long, default_value = "14")]
    max_age_days: u64,
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    offline_bundle: Option<PathBuf>,
    #[arg(long)]
    offline_sha256: Option<String>,
    #[arg(long)]
    packages: Option<PathBuf>,
    #[arg(long, default_value = "cpu")]
    profile: String,
    #[arg(long, default_value = "30")]
    duration_seconds: u64,
}
fn required<T>(v: Option<T>, name: &str) -> Result<T> {
    v.with_context(|| format!("--{name} is required"))
}
fn run(a: Args) -> Result<Value> {
    if a.action == "offline-download" {
        return offline::download(&required(a.output, "output")?);
    }
    if a.action == "offline-create" {
        return offline::create(
            &required(a.source, "source")?,
            &required(a.output, "output")?,
            &required(a.packages, "packages")?,
            &a.profile,
        );
    }
    if a.action == "pack" {
        return package::pack(
            &required(a.source, "source")?,
            &required(a.output, "output")?,
            &required(a.device_config, "device-config")?,
            a.runtime_config.as_deref(),
            a.tag_map.as_deref(),
            &required(a.controller, "controller")?,
        );
    }
    if a.action == "launch" {
        let source = std::fs::canonicalize(a.source.unwrap_or(std::env::current_dir()?))?;
        let c = read(source.join(a.device_config.unwrap_or("config/device.json".into())))?;
        launcher::resources(&source, &c)?;
        ensure!(
            [
                a.build,
                a.validate_config,
                a.check,
                a.check_debug_bridge,
                a.debug_bridge
            ]
            .into_iter()
            .filter(|x| *x)
            .count()
                <= 1,
            "Choose one launch action"
        );
        ensure!(
            a.build || !(a.online || a.with_debug_bridge || a.build_cache.is_some()),
            "Build flags require --build"
        );
        if a.validate_config {
            return Ok(json!({"valid":true}));
        }
        if a.build {
            launcher::build(
                &source,
                &c,
                a.online,
                a.with_debug_bridge,
                a.build_cache.as_deref(),
                None,
            )?;
            return Ok(json!({"built":true}));
        }
        if a.check || a.check_debug_bridge {
            return Ok(
                json!({"executable":launcher::checked_binary(&source,&c,if a.check_debug_bridge{launcher::BRIDGE}else{launcher::MAIN})?}),
            );
        }
        launcher::exec(launcher::command(
            &source,
            &c,
            if a.debug_bridge { Some(a.camera) } else { None },
        )?)?;
        return Ok(Value::Null);
    }
    let root = root(&required(a.root, "root")?)?;
    match a.action.as_str() {
        "validate-device" => validation::validate(&root, a.duration_seconds),
        "cleanup" => retention::cleanup(&root, a.keep_releases, a.max_age_days, a.apply),
        "stage" => package::stage(
            &root,
            &required(a.archive, "archive")?,
            &required(a.sha256, "sha256")?,
            &required(a.release_id, "release-id")?,
            a.with_debug_bridge,
        ),
        "start-job" => jobs::submit(
            &root,
            &required(a.job_id, "job-id")?,
            &required(a.kind, "kind")?,
            a.release_id.as_deref(),
            a.sha256.as_deref(),
        ),
        "status" => {
            if let Some(identity) = a.operation_id {
                let job = activation::operation_status(&root, &identity)?;
                jobs::status(&root, &identity, job)
            } else if let Some(identity) = a.job_id {
                let job = optional(
                    root.join("jobs").join(format!("{}.json", id(&identity)?)),
                    json!({"operation_id":identity,"state":"starting"}),
                )?;
                jobs::status(&root, &identity, job)
            } else if let Some(identity) = a.release_id {
                id(&identity)?;
                let job = optional(
                    root.join("jobs").join(format!("{identity}.json")),
                    json!({"release_id":identity,"state":"starting","activated":false}),
                )?;
                jobs::status(&root, &identity, job)
            } else {
                activation::overview(&root)
            }
        }
        "preflight" => activation::preflight(&root, &required(a.release_id, "release-id")?),
        "activate" | "rollback" => activation::activate(
            &root,
            a.release_id.as_deref(),
            a.action == "rollback",
            &required(a.operation_id, "operation-id")?,
            &activation::Systemd::new(&root)?,
        ),
        "recover" => {
            let _lock = Lock::acquire(&lock_path()?)?;
            activation::recover(&root, Some(&activation::Systemd::new(&root)?))
        }
        "service-run" => {
            activation::service_run(&root)?;
            Ok(Value::Null)
        }
        "install-service" => service::install(
            &root,
            &required(a.legacy_root, "legacy-root")?,
            &a.service,
            &a.user,
        ),
        "diagnostics" | "diagnostics-stop" => service::diagnostics(
            &root,
            a.camera,
            &required(a.client_id, "client-id")?,
            a.action == "diagnostics-stop",
            &activation::Systemd::new(&root)?,
        ),
        "initialize-submit" | "initialize-run" => {
            let legacy = required(a.legacy_root, "legacy-root")?;
            let job = required(a.job_id, "job-id")?;
            let release = required(a.release_id, "release-id")?;
            let sha = required(a.sha256, "sha256")?;
            let o = initialize::Options {
                root: &root,
                legacy: &legacy,
                user: &a.user,
                service: &a.service,
                job: &job,
                release: &release,
                sha: &sha,
                offline:match (&a.offline_bundle,&a.offline_sha256){(Some(path),Some(sha))=>Some((path,sha)),(None,None)=>None,_=>anyhow::bail!("Offline initialization requires both --offline-bundle and --offline-sha256")},
            };
            if a.action == "initialize-submit" {
                initialize::submit(&o)
            } else {
                initialize::initialize(&o)
            }
        }
        _ => unreachable!(),
    }
}
fn main() {
    match run(Args::parse()) {
        Ok(v) => {
            if !v.is_null() {
                println!("{v}");
            }
        }
        Err(e) => {
            eprintln!("Vortex operation failed: {e:#}");
            std::process::exit(1);
        }
    }
}
