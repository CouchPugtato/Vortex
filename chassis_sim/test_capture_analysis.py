"""Evidence-based triage of saved failures without guessing root causes."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_captures import diagnose, main, summarize


class CaptureAnalysisTests(unittest.TestCase):
    def test_manual_cases_without_automatic_failure_remain_for_review(self):
        capture=dict(trigger='manual',failure=dict(reasons=[]))
        self.assertEqual(diagnose(capture,dict(reasons=[]),{})['primary'],'manual_review')
        self.assertEqual(diagnose(capture,dict(reasons=['position_error']),{})['primary'],'position_error')
        capture['failure']['reasons']=['position_error']
        self.assertEqual(diagnose(capture,dict(reasons=[]),{})['primary'],'resolved')

    def test_missing_detection_and_rejected_pose_are_distinct(self):
        outcome=dict(reasons=['missing_pose_with_visible_tags'],expected_ids=[18])
        self.assertEqual(diagnose({},outcome,dict(detected_ids=[]))['primary'],'missed_tags')
        self.assertEqual(diagnose({},outcome,dict(detected_ids=[18]))['primary'],'pose_rejected')

    def test_ambiguity_requires_similar_cost_and_distinct_candidates(self):
        outcome=dict(reasons=['position_error'])
        candidate=dict(id=18,reprojection_rmse_px=.1,camera_position=[0,0,0])
        other=dict(candidate,camera_position=[1,0,0],reprojection_rmse_px=.2)
        self.assertEqual(diagnose({},outcome,dict(pose_candidates=[candidate]))['primary'],'position_error')
        self.assertEqual(diagnose({},outcome,dict(pose_candidates=[candidate,other]))['primary'],'ambiguous_pose')
        other['reprojection_rmse_px']=1.
        self.assertEqual(diagnose({},outcome,dict(pose_candidates=[candidate,other]))['primary'],'position_error')

    def test_rotation_and_original_latency_are_independent_indicators(self):
        capture=dict(frame=dict(result=dict(render_ms=20,vortex_ms=70,sensor=dict(injected_latency_ms=30))))
        analysis=diagnose(capture,dict(reasons=['tilt_error'],tilt_error_deg=12),dict(vortex_ms=1))
        self.assertEqual(analysis['primary'],'incorrect_rotation')
        self.assertEqual(analysis['recorded_latency_ms'],120)
        self.assertIn('latency',analysis['labels'])
        self.assertEqual(diagnose(capture,dict(reasons=[]),{})['primary'],'resolved')

    def test_experiments_separate_and_group_totals_do_not_double_count(self):
        cases=[dict(case_id='experiment',analysis=diagnose(dict(category='configuration_experiment'),
                     dict(reasons=['heading_error','position_error']),{})),
               dict(case_id='regression',analysis=diagnose({},dict(reasons=['heading_error','position_error']),{}))]
        groups=summarize(cases)
        self.assertEqual(sum(g['count'] for g in groups.values()),2)
        self.assertEqual(groups['calibration_experiment']['count'],1)
        self.assertEqual(groups['incorrect_rotation']['indicators']['position_error'],1)

    def test_cli_reports_errors_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'one';case.mkdir()
            metadata=json.dumps(dict(failure=dict(reasons=['position_error']),frame=dict(result={})))
            (case/'capture.json').write_text(metadata)
            (case/'image.png').write_bytes(b'untouched')
            broken=root/'broken';broken.mkdir();(broken/'capture.json').write_text('{')
            self.assertEqual(main([str(root)]),1)
            report=json.loads((root/'analysis/analysis.json').read_text())
            self.assertEqual(report['total'],1)
            self.assertEqual(len(report['errors']),1)
            self.assertIn('Recorded capture', (root/'analysis/analysis.md').read_text(encoding='utf-8'))
            self.assertEqual((case/'capture.json').read_text(),metadata)
            self.assertEqual((case/'image.png').read_bytes(),b'untouched')
