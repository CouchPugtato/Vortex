"""Copy a reviewed capture out of generated storage into a permanent test directory."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('destination', type=Path, help='New case directory; existing paths are never overwritten')
    args = parser.parse_args()
    capture = json.loads((args.case/'capture.json').read_text(encoding='utf-8'))
    if capture['category'] == 'configuration_experiment':
        parser.error('Mount mismatch experiments cannot be promoted as detector regressions')
    shutil.copytree(args.case, args.destination)
    capture['review_status'] = 'reviewed'
    (args.destination/'capture.json').write_text(json.dumps(capture, indent=2), encoding='utf-8')
    print(f'Permanent case: {args.destination}')


if __name__ == '__main__':
    main()
