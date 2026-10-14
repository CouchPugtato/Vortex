"""Replay saved pixels through current Vortex and assert chassis position/heading."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

import cv2

from drive import HERE, StreamReplay, robot_pose
from failure_capture import CapturePolicy, classify, json_safe


def replay_cases(root, executable, include_experiments=False):
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
            results.append(dict(case=str(case), passed=passed, evaluation=outcome, result=report))
            print(f"{'PASS' if passed else 'FAIL'} {case.name}: {outcome['reasons']}")
    finally:
        if stream is not None:
            stream.close()
    output = dict(total=len(results), passed=sum(result['passed'] for result in results), cases=results)
    (root/'capture_replay_report.json').write_text(json.dumps(json_safe(output), indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='One case folder, or a capture directory')
    parser.add_argument('--replay-executable', type=Path, default=HERE.parent/'simulation_tests/replay/target'/(
        'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt'
        else 'release/vortex-synthetic-replay'))
    parser.add_argument('--include-experiments', action='store_true')
    args = parser.parse_args()
    if not args.path.is_dir() or not args.replay_executable.is_file():
        parser.error('Capture directory and built replay executable must exist')
    result = replay_cases(args.path.resolve(), args.replay_executable.resolve(), args.include_experiments)
    print(f"Captured cases: {result['passed']}/{result['total']} pass")
    return int(result['passed'] != result['total']) if result['total'] else 2


if __name__ == '__main__':
    sys.exit(main())
