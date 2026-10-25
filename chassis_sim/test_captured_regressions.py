"""Reproduce reviewed simulator failures using their frozen camera pixels."""
import os
import unittest

from drive import HERE
from replay_captures import replay_cases


class CapturedRegressionTests(unittest.TestCase):
    def test_reviewed_small_tag_captures_pass_position_and_heading(self):
        if not list((HERE.parent/'simulation_tests/captured_cases').rglob('capture.json')):
            self.skipTest('Reviewed capture pixels are local ignored fixtures; none available')
        executable = HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt'
            else 'release/vortex-synthetic-replay')
        if not executable.exists():
            self.skipTest('Build the release replay before running captured regressions')
        report = replay_cases(HERE.parent/'simulation_tests/captured_cases', executable)
        self.assertEqual(report['total'], 4)
        self.assertEqual(report['passed'], report['total'],
                         [case['evaluation'] for case in report['cases'] if not case['passed']])


if __name__ == '__main__':
    unittest.main()
