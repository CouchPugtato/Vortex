"""Replay saved pixels through current Vortex and assert chassis position/heading."""
import argparse
from dataclasses import fields
import json
import hashlib
import os
from pathlib import Path
import sys

import cv2

from drive import HERE, StreamReplay, robot_pose
from failure_capture import CapturePolicy, classify, json_safe


def compare_reports(current, baseline):
    """Stable relative IDs allow comparisons across machines and checkouts."""
    old = {c.get('case_id', c['case']): c for c in baseline['cases']}
    changes = dict(fixed=[], regressed=[], improved=[], worsened=[], new=[], removed=[], incomparable=[])
    seen = set()
    for case in current['cases']:
        key = case.get('case_id', case['case'])
        seen.add(key)
        before = old.get(key)
        if before is None:
            changes['new'].append(key)
            continue
        if before.get('input_signature') != case.get('input_signature'):
            changes['incomparable'].append(key)
            continue
        if before['passed'] != case['passed']:
            changes['fixed' if case['passed'] else 'regressed'].append(key)
        for metric, epsilon in (('position_error_m', .01), ('heading_error_deg', .25)):
            a, b = before['evaluation'].get(metric), case['evaluation'].get(metric)
            if a is not None and b is not None and abs(a-b) > epsilon:
                changes['improved' if b < a else 'worsened'].append(dict(case=key, metric=metric, before=a, after=b))
    changes['removed'] = sorted(set(old)-seen)
    return changes


def replay_cases(root, executable, include_experiments=False, baseline=None, output_path=None, planar=True):
    cases = [root/'capture.json'] if (root/'capture.json').is_file() else sorted(root.rglob('capture.json'))
    env = os.environ.copy()
    if os.name == 'nt':
        env['PATH'] = 'C:/msys64/mingw64/bin' + os.pathsep + env.get('PATH', '')
    results = []
    stream = None
    previous_header = None
    try:
        for capture_path in cases:
            case = capture_path.parent
            capture = json.loads(capture_path.read_text(encoding='utf-8'))
            if capture['category'] == 'configuration_experiment' and not include_experiments:
                continue
            manifest = json.loads((case/'manifest.json').read_text(encoding='utf-8'))
            manifest['frames'] = []
            if planar and capture.get('context', {}).get('renderer') in ('field', 'fast') and capture.get('context',{}).get('planar_localization',True):
                manifest['robot_mount'] = capture['frame']['assumed_mount']
            else:
                manifest.pop('robot_mount',None)
            header = json.dumps(manifest, sort_keys=True)
            if previous_header != header:
                if stream is not None:
                    stream.close()
                stream = StreamReplay(executable, manifest, env, case)
                previous_header = header
            image = cv2.imread(str(case/'image.png'), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (manifest['height'], manifest['width']):
                raise ValueError(f'Invalid saved image: {case}')
            report = stream.request(image)
            estimate = report['joint_pose']
            if estimate is None and report['poses']:
                estimate = min(report['poses'], key=lambda p: p['reprojection_rmse_px'])
            frame = capture['frame']
            ghost = None if estimate is None else robot_pose(estimate['camera_position'],
                estimate['camera_rotation'], frame['assumed_mount'])
            policy = CapturePolicy(**{f.name: capture['policy'][f.name] for f in fields(CapturePolicy)
                                       if f.name in capture['policy']})
            outcome = classify(ghost, frame['chassis_pose'], report, frame['visibility'],
                               [tag['id'] for tag in manifest['tags']], policy)
            passed = not outcome['reasons']
            signature = hashlib.sha256((case/'image.png').read_bytes()+json.dumps(
                dict(manifest=manifest, truth=frame['chassis_pose'], mount=frame['assumed_mount'],
                     visibility=frame['visibility'], policy=capture['policy']), sort_keys=True).encode()).hexdigest()
            results.append(dict(case=str(case), case_id=case.relative_to(root).as_posix(), input_signature=signature,
                                passed=passed, evaluation=outcome, result=report))
            print(f"{'PASS' if passed else 'FAIL'} {case.name}: {outcome['reasons']}")
    finally:
        if stream is not None:
            stream.close()
    output = dict(total=len(results), passed=sum(result['passed'] for result in results), cases=results)
    if baseline is not None:
        output['changes'] = compare_reports(output, baseline)
    destination = output_path or root/'capture_replay_report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(output), indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='One case folder, or a capture directory')
    parser.add_argument('--replay-executable', type=Path, default=HERE.parent/'simulation_tests/replay/target'/(
        'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt'
        else 'release/vortex-synthetic-replay'))
    parser.add_argument('--include-experiments', action='store_true')
    parser.add_argument('--baseline', type=Path, help='Previous report to compare without overwriting it')
    parser.add_argument('--report', type=Path, help='Output report (defaults to capture directory)')
    parser.add_argument('--free-pose', action='store_true', help='Disable the calibrated level-chassis model')
    args = parser.parse_args()
    if not args.path.is_dir() or not args.replay_executable.is_file():
        parser.error('Capture directory and built replay executable must exist')
    destination = (args.report or args.path/'capture_replay_report.json').resolve()
    if args.baseline and args.baseline.resolve() == destination:
        parser.error('Baseline and output report must be different files')
    baseline = json.loads(args.baseline.read_text(encoding='utf-8')) if args.baseline else None
    result = replay_cases(args.path.resolve(), args.replay_executable.resolve(), args.include_experiments,
                          baseline, destination, planar=not args.free_pose)
    if 'changes' in result:
        print('Changes: '+', '.join(f'{key}={len(value)}' for key, value in result['changes'].items()))
    print(f"Captured cases: {result['passed']}/{result['total']} pass")
    return int(result['passed'] != result['total']) if result['total'] else 2


if __name__ == '__main__':
    sys.exit(main())
