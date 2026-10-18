use super::*;
fn preview() -> (tempfile::TempDir, Fake, Value) {
    let (t, _, s) = staged();
    atomic(t.path().join("selection.json"), &s).unwrap();
    let report = json!({"schema_version":1,"release_id":s["release_id"],"launch_token":"token","pid":123,"diagnostics_version":1,"timestamp_ms":(now()*1000.) as u64,"watchdog":"healthy","cameras":[{"index":0,"fps":30}]});
    atomic(t.path().join("health.json"), &report).unwrap();
    let state = json!({"schema_version":1,"release_id":s["release_id"],"launch_token":"token","pid":123,"camera_index":0,"client_id":"client","timestamp_ms":(now()*1000.) as u64,"latency_ms":10,"frame_id":"frame-123-1.jpg"});
    fs::create_dir(t.path().join("diagnostics")).unwrap();
    atomic(t.path().join("diagnostics/preview.json"), &state).unwrap();
    fs::write(
        t.path().join("diagnostics/frame-123-1.jpg"),
        b"\xff\xd8\xffimage",
    )
    .unwrap();
    let m = Fake::new();
    m.active.set(true);
    (t, m, state)
}
#[test]
fn paired_preview_is_returned_without_service_control() {
    let (t, m, _) = preview();
    let v = service::diagnostics(t.path(), 0, "client", false, &m).unwrap();
    assert_eq!(v["ready"], true);
    assert_eq!(v["state"]["fps"], 30);
    assert!(v["jpeg_base64"].as_str().unwrap().starts_with("/9j/"));
    assert_eq!(m.starts.get() + m.stops.get(), 0);
    let lease = read(t.path().join("diagnostics/request.json")).unwrap();
    assert_eq!(lease["client_id"], "client");
    assert!(lease["expires_ms"].as_f64().unwrap() > now() * 1000.);
}
#[test]
fn stale_preview_identity_is_never_displayed() {
    let (t, m, s) = preview();
    for (key, bad) in [
        ("schema_version", json!(2)),
        ("release_id", json!("old")),
        ("launch_token", json!("old")),
        ("pid", json!(122)),
        ("camera_index", json!(1)),
        ("client_id", json!("other")),
        ("timestamp_ms", json!(0)),
        ("latency_ms", json!(2000)),
        ("latency_ms", json!(-1)),
    ] {
        let mut state = s.clone();
        state[key] = bad;
        atomic(t.path().join("diagnostics/preview.json"), &state).unwrap();
        let v = service::diagnostics(t.path(), 0, "client", false, &m).unwrap();
        assert_eq!(v["ready"], false, "{key}");
        assert!(v["jpeg_base64"].is_null());
    }
}
#[test]
fn stale_runtime_health_never_creates_lease() {
    let (t, m, _) = preview();
    let mut r = read(t.path().join("health.json")).unwrap();
    for (key, bad) in [
        ("diagnostics_version", json!(0)),
        ("pid", json!(5)),
        ("launch_token", json!("old")),
        ("timestamp_ms", json!(0)),
    ] {
        let original = r.clone();
        r[key] = bad;
        atomic(t.path().join("health.json"), &r).unwrap();
        assert_eq!(
            service::diagnostics(t.path(), 0, "client", false, &m).unwrap()["ready"],
            false
        );
        assert!(!t.path().join("diagnostics/request.json").exists());
        r = original;
    }
}
#[test]
fn another_live_client_retains_lease() {
    let (t, m, _) = preview();
    let lease = json!({"client_id":"other","pid":123,"launch_token":"token","expires_ms":(now()*1000.) as u64+3000});
    atomic(t.path().join("diagnostics/request.json"), &lease).unwrap();
    let v = service::diagnostics(t.path(), 0, "client", false, &m).unwrap();
    assert_eq!(v["ready"], false);
    assert_eq!(
        read(t.path().join("diagnostics/request.json")).unwrap(),
        lease
    );
}
#[test]
fn expired_lease_can_be_acquired() {
    let (t, m, _) = preview();
    atomic(
        t.path().join("diagnostics/request.json"),
        &json!({"client_id":"other","pid":123,"launch_token":"token","expires_ms":0}),
    )
    .unwrap();
    assert_eq!(
        service::diagnostics(t.path(), 0, "client", false, &m).unwrap()["ready"],
        true
    );
    assert_eq!(
        read(t.path().join("diagnostics/request.json")).unwrap()["client_id"],
        "client"
    );
}
#[test]
fn missing_rotated_image_is_waiting_state() {
    let (t, m, _) = preview();
    fs::remove_file(t.path().join("diagnostics/frame-123-1.jpg")).unwrap();
    assert_eq!(
        service::diagnostics(t.path(), 0, "client", false, &m).unwrap()["ready"],
        false
    );
}
#[test]
fn image_paths_and_sizes_are_restricted() {
    let (t, m, s) = preview();
    for name in [
        "../outside.jpg",
        "frame-1-a.jpg",
        "/etc/passwd",
        "frame--1.jpg",
    ] {
        let mut state = s.clone();
        state["frame_id"] = json!(name);
        atomic(t.path().join("diagnostics/preview.json"), &state).unwrap();
        assert!(service::diagnostics(t.path(), 0, "client", false, &m).is_err());
    }
    atomic(t.path().join("diagnostics/preview.json"), &s).unwrap();
    for bytes in [vec![0; 20], vec![0; 512 * 1024 + 1]] {
        fs::write(t.path().join("diagnostics/frame-123-1.jpg"), bytes).unwrap();
        assert!(service::diagnostics(t.path(), 0, "client", false, &m).is_err());
    }
}
#[test]
fn unconfigured_camera_is_rejected() {
    let (t, m, _) = preview();
    assert!(service::diagnostics(t.path(), 1, "client", false, &m).is_err());
}
#[test]
fn process_change_during_read_discards_preview() {
    let (t, _, _) = preview();
    struct Restart(Cell<u32>);
    impl Manager for Restart {
        fn info(&self) -> Result<Value> {
            let calls = self.0.get();
            self.0.set(calls + 1);
            Ok(json!({"active":true,"pid":if calls==0{123}else{124}}))
        }
        fn stop(&self) -> Result<()> {
            panic!("Must not stop production")
        }
        fn start(&self) -> Result<()> {
            panic!("Must not start production")
        }
    }
    let m = Restart(Cell::new(0));
    let v = service::diagnostics(t.path(), 0, "client", false, &m).unwrap();
    assert_eq!(v["ready"], false);
    assert!(v["jpeg_base64"].is_null());
}
