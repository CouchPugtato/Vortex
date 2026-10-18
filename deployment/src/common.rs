use anyhow::{bail, ensure, Context, Result};
use fs2::FileExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use wait_timeout::ChildExt;

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
pub fn token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}
pub fn read(path: impl AsRef<Path>) -> Result<Value> {
    let p = path.as_ref();
    serde_json::from_reader(File::open(p).with_context(|| format!("Reading {}", p.display()))?)
        .map_err(Into::into)
}
pub fn optional(path: impl AsRef<Path>, default: Value) -> Result<Value> {
    if path.as_ref().exists() {
        read(path)
    } else {
        Ok(default)
    }
}
pub fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("Missing string: {key}"))
}
pub fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    let _ = path;
    Ok(())
}
pub fn mode(path: &Path, bits: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(bits))?;
    }
    let _ = (path, bits);
    Ok(())
}
pub fn atomic_bytes(path: &Path, bytes: &[u8], bits: u32) -> Result<()> {
    let parent = path.parent().context("Missing parent")?;
    fs::create_dir_all(parent)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    mode(temp.path(), bits)?;
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|e| e.error)?;
    sync_dir(parent)
}
pub fn atomic(path: impl AsRef<Path>, data: &Value) -> Result<()> {
    atomic_bytes(path.as_ref(), &serde_json::to_vec_pretty(data)?, 0o644)
}
pub fn hash_bytes(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
pub fn hash(path: impl AsRef<Path>) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn checksum(s: &str) -> Result<()> {
    ensure!(
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "Invalid SHA256 checksum"
    );
    Ok(())
}
pub fn id(s: &str) -> Result<&str> {
    ensure!(
        !s.is_empty()
            && s.len() <= 96
            && s.as_bytes()[0].is_ascii_alphanumeric()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        "Invalid release/job ID"
    );
    Ok(s)
}
pub fn simple(s: &str) -> Result<()> {
    ensure!(
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b)),
        "Use simple absolute Linux paths and account names"
    );
    Ok(())
}
pub fn service(s: &str) -> Result<()> {
    ensure!(
        s.ends_with(".service")
            && s.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
        "Invalid service name"
    );
    Ok(())
}
pub fn home() -> Result<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .context("Missing home directory")
}
pub fn lock_path() -> Result<PathBuf> {
    Ok(home()?.join(".local/state/vortex/deployment.lock"))
}
pub struct Lock(File);
impl Lock {
    pub fn acquire(path: &Path) -> Result<Self> {
        fs::create_dir_all(path.parent().context("Lock parent")?)?;
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        f.try_lock_exclusive()
            .context("Another Vortex deployment is in progress")?;
        Ok(Self(f))
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}
pub fn claim(root: &Path, identity: &str, suffix: &str) -> Result<()> {
    id(identity)?;
    fs::create_dir_all(root.join("jobs"))?;
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("jobs").join(format!("{identity}.{suffix}")))?
        .write_all(std::process::id().to_string().as_bytes())?;
    Ok(())
}
pub fn root(path: &Path) -> Result<PathBuf> {
    ensure!(path.is_absolute(), "Managed root must be absolute");
    fs::create_dir_all(path)?;
    let p = fs::canonicalize(path)?;
    #[cfg(windows)]
    {
        return Ok(PathBuf::from(
            p.to_string_lossy().trim_start_matches(r"\\?\"),
        ));
    }
    #[cfg(not(windows))]
    Ok(p)
}
pub fn child(root: &Path, relative: &str) -> Result<PathBuf> {
    ensure!(
        !relative.is_empty()
            && !relative.contains('\\')
            && !relative.contains(':')
            && !relative.chars().any(char::is_control),
        "Unsafe managed path"
    );
    ensure!(
        relative
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != ".."),
        "Unsafe managed path"
    );
    let mut p = root.to_path_buf();
    for part in relative.split('/') {
        p.push(part);
        ensure!(
            !fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink()),
            "Symlink in managed path: {}",
            p.display()
        );
    }
    Ok(p)
}
pub fn regular(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "Expected regular file: {}",
        path.display()
    );
    Ok(())
}
pub fn copy(source: &Path, dest: &Path, bits: u32) -> Result<()> {
    regular(source)?;
    let before = hash(source)?;
    let parent = dest.parent().context("Copy destination parent")?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    std::io::copy(&mut File::open(source)?, temporary.as_file_mut())?;
    ensure!(
        hash(temporary.path())? == before && hash(source)? == before,
        "Source changed while packaging"
    );
    mode(temporary.path(), bits)?;
    temporary.as_file().sync_all()?;
    temporary.persist(dest).map_err(|e| e.error)?;
    sync_dir(parent)
}
pub fn linux() -> Result<()> {
    ensure!(
        cfg!(target_os = "linux"),
        "This operation requires Linux with systemd"
    );
    Ok(())
}
pub fn uid() -> u32 {
    #[cfg(unix)]
    {
        unsafe { libc::getuid() }
    }
    #[cfg(not(unix))]
    {
        u32::MAX
    }
}
pub fn admin() -> Result<()> {
    linux()?;
    ensure!(uid() == 0, "Administrator privileges required");
    Ok(())
}
pub fn user_env(cmd: &mut Command) {
    let dir = format!("/run/user/{}", uid());
    cmd.env("XDG_RUNTIME_DIR", &dir)
        .env("DBUS_SESSION_BUS_ADDRESS", format!("unix:path={dir}/bus"));
}
// Temporary files keep subprocess output bounded in memory and avoid pipe deadlocks.
pub fn output(cmd: &mut Command, seconds: u64) -> Result<Output> {
    let stdout = tempfile::tempfile()?;
    let stderr = tempfile::tempfile()?;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?)
        .spawn()
        .with_context(|| format!("Starting {cmd:?}"))?;
    let status = match child.wait_timeout(Duration::from_secs(seconds))? {
        Some(s) => s,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            bail!("Command timed out: {cmd:?}")
        }
    };
    use std::io::{Seek, SeekFrom};
    fn tail(mut f: File) -> Result<Vec<u8>> {
        let n = f.metadata()?.len();
        f.seek(SeekFrom::Start(n.saturating_sub(4 * 1024 * 1024)))?;
        let mut v = Vec::new();
        f.read_to_end(&mut v)?;
        Ok(v)
    }
    Ok(Output {
        status,
        stdout: tail(stdout)?,
        stderr: tail(stderr)?,
    })
}
pub fn checked(cmd: &mut Command, seconds: u64) -> Result<Output> {
    let o = output(cmd, seconds)?;
    ensure!(
        o.status.success(),
        "Command failed ({cmd:?}): {}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    Ok(o)
}
pub fn logged(cmd: &mut Command, log: &File, seconds: u64) -> Result<()> {
    let mut c = cmd
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log.try_clone()?)
        .spawn()?;
    let s = match c.wait_timeout(Duration::from_secs(seconds))? {
        Some(s) => s,
        None => {
            let _ = c.kill();
            let _ = c.wait();
            bail!("Command timed out")
        }
    };
    ensure!(s.success(), "Command failed: {cmd:?}");
    Ok(())
}
pub fn properties(bytes: &[u8]) -> Value {
    let mut v = json!({});
    for l in String::from_utf8_lossy(bytes).lines() {
        if let Some((k, s)) = l.split_once('=') {
            v[k] = json!(s);
        }
    }
    v
}
pub fn executable() -> Result<PathBuf> {
    Ok(std::env::current_exe()?)
}
