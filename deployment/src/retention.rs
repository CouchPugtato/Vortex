//! Conservative, lock-protected retention. Unknown/live work and recovery inputs
//! are retained; deletion requires an explicit --apply.
use crate::{common::*, jobs};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    time::{Duration, SystemTime},
};
fn refs(v: &Value, protected: &mut BTreeSet<String>) -> Result<()> {
    match v {
        Value::Object(map) => {
            if let Some(value) = map.get("release_id") {
                let name = value.as_str().context("Invalid release reference")?;
                id(name)?;
                protected.insert(name.into());
            }
            for child in map.values() {
                refs(child, protected)?;
            }
        }
        Value::Array(a) => {
            for v in a {
                refs(v, protected)?
            }
        }
        _ => {}
    }
    Ok(())
}
fn age(path: &Path, days: u64) -> Result<bool> {
    Ok(SystemTime::now()
        .duration_since(fs::metadata(path)?.modified()?)
        .unwrap_or_default()
        >= Duration::from_secs(days.saturating_mul(86400)))
}
fn safe_tree(root: &Path, relative: &str) -> Result<()> {
    let p = child(root, relative)?;
    for e in walkdir::WalkDir::new(&p) {
        ensure!(
            !e?.file_type().is_symlink(),
            "Refusing cleanup of a tree containing symlinks: {relative}"
        );
    }
    Ok(())
}
pub fn cleanup(root: &Path, keep: usize, days: u64, apply: bool) -> Result<Value> {
    let config = optional(root.join("managed-service.json"), Value::Null)?;
    let lock = if cfg!(target_os = "linux") && config["installed"] == true {
        crate::service::account(string(&config, "user")?)?
            .home
            .join(".local/state/vortex/deployment.lock")
    } else {
        ensure!(
            uid() != 0,
            "Run cleanup as the runtime account until managed service setup is complete"
        );
        lock_path()?
    };
    cleanup_with_lock(root, keep, days, apply, &lock)
}
pub fn cleanup_with_lock(
    root: &Path,
    keep: usize,
    days: u64,
    apply: bool,
    lock: &Path,
) -> Result<Value> {
    ensure!(
        keep >= 1 && days >= 1,
        "Retention must keep at least one release and one day"
    );
    let _lock = Lock::acquire(lock)?;
    ensure!(
        !root.join("activation.json").exists(),
        "Recover the unfinished activation before cleanup"
    );
    let mut protected = BTreeSet::new();
    for name in [
        "selection.json",
        "active.json",
        "previous.json",
        "activation.json",
    ] {
        refs(&optional(root.join(name), Value::Null)?, &mut protected)?;
    }
    let mut jobs = BTreeMap::new();
    let mut protected_jobs = BTreeSet::new();
    if root.join("jobs").exists() {
        for e in fs::read_dir(child(root, "jobs")?)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            if !name.ends_with(".json") || name.ends_with(".unit.json") {
                continue;
            }
            let identity = name.trim_end_matches(".json");
            id(identity)?;
            let path = child(root, &format!("jobs/{name}"))?;
            let job = read(&path)?;
            if !jobs::TERMINAL.contains(&job["state"].as_str().unwrap_or("")) {
                protected_jobs.insert(identity.to_owned());
                refs(&job, &mut protected)?;
                protected.insert(identity.into());
            }
            jobs.insert(identity.to_owned(), job);
        }
    }
    let mut releases = Vec::new();
    if root.join("releases").exists() {
        for e in fs::read_dir(child(root, "releases")?)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().to_string();
            id(&n)?;
            let p = child(root, &format!("releases/{n}"))?;
            ensure!(p.is_dir(), "Unexpected release resource");
            releases.push((fs::metadata(p)?.modified()?, n));
        }
    }
    releases.sort_by(|a, b| b.cmp(a));
    protected.extend(releases.iter().take(keep).map(|(_, n)| n.clone()));
    // Pending setup jobs protect both the package and their extracted bootstrap.
    for (identity, job) in &jobs {
        if job["release_id"]
            .as_str()
            .is_some_and(|r| protected.contains(r))
        {
            protected_jobs.insert(identity.clone());
        }
    }
    let mut candidates = Vec::<String>::new();
    for category in ["releases", "incoming", "staging", "initialization"] {
        let folder = child(root, category)?;
        if !folder.exists() {
            continue;
        }
        for entry in fs::read_dir(folder)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            id(&name)?;
            let relative = format!("{category}/{name}");
            let path = child(root, &relative)?;
            if protected.contains(&name) || protected_jobs.contains(&name) || !age(&path, days)? {
                continue;
            }
            // Staging/bootstrap trees without a terminal owner may still be in use.
            if ["staging", "initialization"].contains(&category)
                && !jobs
                    .get(&name)
                    .is_some_and(|j| jobs::TERMINAL.contains(&j["state"].as_str().unwrap_or("")))
            {
                continue;
            }
            // Incoming packages without a job are retained: an upload may be ongoing.
            if category == "incoming" && !jobs.contains_key(&name) {
                continue;
            }
            safe_tree(root, &relative)?;
            candidates.push(relative);
        }
    }
    for (identity, job) in &jobs {
        if protected_jobs.contains(identity)
            || protected.contains(identity)
            || !jobs::TERMINAL.contains(&job["state"].as_str().unwrap_or(""))
            || !age(&root.join("jobs").join(format!("{identity}.json")), days)?
        {
            continue;
        }
        for suffix in [
            "json",
            "log",
            "controller.log",
            "claim",
            "dispatch",
            "unit.json",
        ] {
            let r = format!("jobs/{identity}.{suffix}");
            let p = child(root, &r)?;
            if p.exists() {
                regular(&p)?;
                candidates.push(r);
            }
        }
    }
    candidates.sort();
    if apply {
        for relative in &candidates {
            safe_tree(root, relative)?;
            let p = child(root, relative)?;
            if p.is_dir() {
                fs::remove_dir_all(&p)?;
            } else {
                fs::remove_file(&p)?;
            }
            sync_dir(p.parent().unwrap())?;
        }
    }
    Ok(
        json!({"schema_version":1,"applied":apply,"keep_releases":keep,"minimum_age_days":days,"protected_releases":protected,"protected_jobs":protected_jobs,"paths":candidates}),
    )
}
