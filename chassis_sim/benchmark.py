"""Run repeatable camera/route/mount experiments, then replay saved failures."""
from dataclasses import asdict
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import subprocess
import shutil
import time

import numpy as np

from camera_effects import CameraEffects, PROFILES, Sensor
from drive import HERE, Harness, Telemetry, build_parser, mount_rotation
from failure_capture import CapturePolicy, classify, json_safe
from replay_captures import replay_cases
from routes import ROUTES, MOUNT_PRESETS, Route, mounts_for


def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return dict(count=0, mean=None, p95=None, max=None)
    return dict(count=len(values), mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95)), max=float(max(values)))


def pose_errors(ghost, truth):
    if ghost is None or not all(np.isfinite(v).all() for v in ghost):
        return None, None
    p, r = ghost
    angle = math.atan2(r[1, 0], r[0, 0])-truth[2]
    return float(np.linalg.norm(p-[*truth[:2], 0.])), abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle))))


def summarize(frames, experiment=False):
    delivered = [f for f in frames if not f['dropped']]
    visible = [f for f in delivered if f['expected_ids']]
    reasons = {}
    for frame in delivered:
        for reason in frame['reasons']:
            reasons[reason] = reasons.get(reason, 0)+1
    return dict(frames=len(frames), dropped=len(frames)-len(delivered), delivered=len(delivered),
                valid=sum(f['valid'] for f in delivered), visible_frames=len(visible),
                visible_pose_rate=sum(f['valid'] for f in visible)/len(visible) if visible else None,
                invalid=sum(not f['valid'] for f in delivered),
                display_valid=sum(f.get('display_valid', False) for f in frames),
                bad_frames=sum(bool(f['reasons']) for f in delivered), reasons=reasons,
                configuration_experiment=experiment,
                position_m=stats(f['position_m'] for f in delivered),
                heading_deg=stats(f['heading_deg'] for f in delivered),
                raw_delivery_position_m=stats(f['raw_delivery_position_m'] for f in frames),
                predicted_delivery_position_m=stats(f['predicted_delivery_position_m'] for f in frames),
                predicted_delivery_heading_deg=stats(f['predicted_delivery_heading_deg'] for f in frames),
                latency_ms=stats(f['latency_ms'] for f in frames),
                render_ms=stats(f['render_ms'] for f in frames),
                vortex_ms=stats(f['vortex_ms'] for f in delivered))


def compare_runs(current, baseline):
    old = {r['id']: r for r in baseline['runs']}
    changes = []
    for run in current:
        before = old.get(run['id'])
        if before is None:
            changes.append(dict(id=run['id'], status='new'))
            continue
        if before['signature'] != run['signature']:
            changes.append(dict(id=run['id'], status='incomparable'))
            continue
        deltas = {key: run['summary'][key]-before['summary'][key]
                  for key in ('bad_frames', 'invalid', 'dropped')}
        for metric in ('position_m', 'heading_deg', 'latency_ms', 'predicted_delivery_position_m'):
            a, b = before['summary'][metric]['p95'], run['summary'][metric]['p95']
            deltas[metric+'_p95'] = None if a is None or b is None else b-a
        changes.append(dict(id=run['id'], status='regressed' if deltas['bad_frames'] > 0 else
                            'improved' if deltas['bad_frames'] < 0 else 'unchanged', deltas=deltas))
    changes.extend(dict(id=key, status='removed') for key in sorted(set(old)-{r['id'] for r in current}))
    return changes


def run_route(harness, route, profile, preset, args, telemetry=None):
    harness.sensor = Sensor(CameraEffects(**asdict(PROFILES[profile])), args.seed)
    actual, assumed = mounts_for(preset, harness.actual)
    initial = harness.initial.copy()
    base_forward, forward = mount_rotation(harness.actual)[:, 2], mount_rotation(actual)[:, 2]
    initial[2] += math.atan2(base_forward[1], base_forward[0])-math.atan2(forward[1], forward[0])
    trajectory = route.timeline(initial, harness.field)
    policy = CapturePolicy(position_m=args.capture_position_m, heading_deg=args.capture_heading_deg)
    frames = []
    last_raw, last_captured = None, None
    # Samples at fixed virtual timestamps; CPU speed cannot change trajectory.
    for index in range(math.floor(route.duration*args.sample_hz)+1):
        timestamp = index/args.sample_hz
        truth = trajectory.pose(timestamp)
        result = harness.localize(truth, actual, assumed, timestamp, route.command(timestamp), realtime=False)
        ghost, _, report, _, latency = result
        dropped = report['sensor']['dropped_frame']
        outcome = classify(ghost, truth, report, report['visibility'],
                           [t['id'] for t in harness.tags], policy)
        delivered_at = timestamp+latency
        delivered_truth = trajectory.pose(delivered_at)
        if not dropped:
            last_raw, last_captured = ghost, timestamp
        predicted = trajectory.prediction(last_raw, last_captured, delivered_at) if last_captured is not None else None
        raw_error, _ = pose_errors(last_raw, delivered_truth)
        predicted_error, predicted_heading = pose_errors(predicted, delivered_truth)
        frames.append(dict(index=index, timestamp_s=timestamp, chassis=truth.tolist(),
            valid=ghost is not None, display_valid=predicted is not None, dropped=dropped,
            reasons=[] if dropped else outcome['reasons'], expected_ids=outcome['expected_ids'],
            detected_ids=report['detected_ids'], position_m=outcome['position_error_m'],
            heading_deg=outcome['heading_error_deg'], latency_ms=latency*1000,
            render_ms=report['render_ms'], vortex_ms=report['vortex_ms'],
            raw_delivery_position_m=raw_error, predicted_delivery_position_m=predicted_error,
            predicted_delivery_heading_deg=predicted_heading, sensor=report['sensor']))
        if telemetry is not None:
            telemetry.publish_actual(delivered_truth, actual)
            if not dropped:
                telemetry.publish_result(result, display_raw=False)
            telemetry.publish_display(predicted, latency)
        if index % max(1, round(args.sample_hz*2)) == 0:
            print(f'  {timestamp:.1f}/{route.duration:.1f}s', flush=True)
    settings = dict(route=route.segments, initial=initial.tolist(), actual=actual, assumed=assumed,
                    camera=harness.camera, tags=harness.tags, field=harness.field, renderer=args.renderer,
                    effects=asdict(harness.sensor.settings), seed=args.seed, sample_hz=args.sample_hz,
                    position_m=policy.position_m, heading_deg=policy.heading_deg)
    settings['planar_localization'] = not args.free_pose
    # Compare inputs, not source hashes: code changes are what the baseline evaluates.
    if harness.field_asset is not None:
        settings['asset_sha256'] = hashlib.sha256((harness.field_asset/'model.glb').read_bytes()).hexdigest()
        settings['asset_config_sha256'] = hashlib.sha256((harness.field_asset/'config.json').read_bytes()).hexdigest()
    summary = summarize(frames, actual != assumed)
    # All-drop/no-visible runs are inconclusive, never a clean success.
    summary['status'] = ('experiment' if actual != assumed else 'inconclusive' if not summary['visible_frames']
                         else 'fail' if summary['bad_frames'] > args.max_bad_frames else 'pass')
    return dict(id=f'{route.name}/{profile}/{preset}', signature=hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()).hexdigest(), settings=settings, summary=summary, frames=frames)


def write_reports(output, report):
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(json_safe(report), indent=2), encoding='utf-8')
    with (output/'frames.csv').open('w', newline='', encoding='utf-8') as file:
        keys = ['run', 'index', 'timestamp_s', 'valid', 'dropped', 'position_m', 'heading_deg',
                'raw_delivery_position_m', 'predicted_delivery_position_m', 'latency_ms', 'render_ms', 'vortex_ms']
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for run in report['runs']:
            for frame in run['frames']:
                writer.writerow(dict(frame, run=run['id']))
    rows = ['# Simulator benchmark', '', '| Route / camera / mount | Status | Bad / delivered | Position p95 (m) | Heading p95 (deg) | Latency p95 (ms) |',
            '|---|---|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    for run in report['runs']:
        s = run['summary']
        rows.append(f"| {run['id']} | {s['status']} | {s['bad_frames']}/{s['delivered']} | "
                    f"{number(s['position_m']['p95'])} | {number(s['heading_deg']['p95'])} | {number(s['latency_ms']['p95'])} |")
    rows += ['', 'Raw errors use image capture time. Delivery errors use the chassis at estimated delivery time.',
             'Latency includes measured CPU work plus injected delay; virtual timestamps do not model a queued camera pipeline.',
             'Mount experiments report errors without gating the detector suite. Inconclusive means no delivered clearly visible tags.',
             '', f"Captured replay suites: {len(report.get('capture_replays', []))}."]
    for replay in report.get('capture_replays', []):
        rows.append(f"- {replay['root']}: {replay['passed']}/{replay['total']} pass; {replay.get('changes', {})}")
    if 'changes' in report:
        rows += ['', 'Baseline changes:']+[f"- {c['id']}: {c['status']}" for c in report['changes']]
    (output/'report.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--routes', nargs='+', choices=ROUTES, default=list(ROUTES))
    parser.add_argument('--route-file', type=Path, help='JSON {name, segments: [[seconds, forward, left, turn], ...]}')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    parser.add_argument('--mount-presets', nargs='+', choices=MOUNT_PRESETS, default=list(MOUNT_PRESETS))
    parser.add_argument('--sample-hz', type=float, default=10)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path, help='Previous benchmark report.json')
    parser.add_argument('--max-bad-frames', type=int, default=0)
    parser.add_argument('--skip-capture-replay', action='store_true')
    parser.add_argument('--capture-replay-dir', type=Path, action='append', help='Additional capture root (repeatable)')
    parser.add_argument('--publish', action='store_true', help='Publish scripted samples to AdvantageScope over NT4')
    args = parser.parse_args(argv)
    if not math.isfinite(args.sample_hz) or not 0 < args.sample_hz <= 120 or args.seed < 0 or args.max_bad_frames < 0:
        parser.error('Sample rate must be (0,120], seed and bad-frame allowance nonnegative')
    if args.headless_frames or args.advantagescope or args.launch_advantagescope:
        parser.error('Use --publish to stream benchmark poses; launch AdvantageScope separately')
    if any(getattr(args, key) is not None for key in asdict(CameraEffects())):
        parser.error('Benchmark uses named --profiles; individual camera overrides belong to drive.py')
    if args.use_config_mount:
        args.mounts = None
    output = (args.output or HERE/'generated'/f'benchmark_{time.time_ns()}').resolve()
    if output.exists():
        parser.error('Choose a fresh output directory so previous reports are preserved')
    baseline = json.loads(args.baseline.read_text(encoding='utf-8')) if args.baseline else None
    routes = [Route.load(name, ROUTES[name]) for name in args.routes]
    if args.route_file:
        custom = json.loads(args.route_file.read_text(encoding='utf-8'))
        routes = [Route.load(custom['name'], custom['segments'])]
    args.max_error_meters = args.capture_position_m
    # Harness performs the same production replay build used by interactive driving.
    harness = Harness(args)
    telemetry = None
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE.parent, capture_output=True,
                              text=True).stdout.strip() if shutil.which('git') else 'unknown'
    sources = ('src/detector.rs', 'src/pose.rs', 'src/tag_refine.rs', 'src/config.rs',
               'simulation_tests/replay/src/main.rs', 'chassis_sim/drive.py',
               'chassis_sim/camera_effects.py', 'chassis_sim/routes.py', 'chassis_sim/benchmark.py')
    report = dict(schema_version=1, seed=args.seed, runs=[], capture_replays=[], output=str(output),
                  provenance=dict(code_revision=revision,
                      replay_sha256=hashlib.sha256(Path(harness.command[0]).read_bytes()).hexdigest(),
                      source_sha256={p: hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest() for p in sources}))
    try:
        telemetry = Telemetry(harness.root) if args.publish else None
        output.mkdir(parents=True)
        for route, profile, preset in itertools.product(routes, args.profiles, args.mount_presets):
            print(f'RUN {route.name}/{profile}/{preset}', flush=True)
            report['runs'].append(run_route(harness, route, profile, preset, args, telemetry))
            s = report['runs'][-1]['summary']
            print(f"  {s['status']}: {s['bad_frames']} bad / {s['delivered']} delivered, "
                  f"{s['dropped']} dropped, position p95={s['position_m']['p95']}", flush=True)
            write_reports(output, report)
        if baseline:
            report['changes'] = compare_runs(report['runs'], baseline)
    finally:
        harness.close()  # Flush asynchronous captures before automatic replay.
        if telemetry is not None:
            telemetry.instance.stopServer()
    if harness.capture is not None:
        report['captures'] = harness.capture.status()
    if not args.skip_capture_replay:
        roots = [HERE.parent/'simulation_tests/captured_cases', args.capture_dir]+(args.capture_replay_dir or [])
        seen = set()
        for root in roots:
            root = root.resolve()
            if root in seen or not root.is_dir():
                continue
            seen.add(root)
            root_id = root.relative_to(HERE.parent).as_posix() if root.is_relative_to(HERE.parent) else str(root)
            old = next((r for r in (baseline or {}).get('capture_replays', [])
                        if r.get('root_id', r['root']) == root_id), None)
            result = replay_cases(root, Path(harness.command[0]), baseline=old,
                                  output_path=output/f'capture-replay-{len(seen)}.json', planar=not args.free_pose)
            report['capture_replays'].append(dict(result, root=str(root), root_id=root_id))
    write_reports(output, report)
    print(f'Reports: {output}/report.md and report.json; frames.csv', flush=True)
    return int(any(r['summary']['status'] in ('fail', 'inconclusive') for r in report['runs']) or
               any(r['passed'] != r['total'] for r in report['capture_replays']))


if __name__ == '__main__':
    sys.exit(main())
