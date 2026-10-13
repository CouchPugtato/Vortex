"""Check image/geometry conventions independently of Vortex's Rust solver."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from generate import HERE, camera_pose, project, tag_geometry, expand_cases


class GeneratorTests(unittest.TestCase):
    def test_extended_suite_coverage(self):
        tags = {t["ID"] for t in json.loads((HERE.parent / "config/2026-rebuilt-welded.json").read_text())["tags"]}
        cases = expand_cases(json.loads((HERE / "scenarios_extended.json").read_text()), tags)
        self.assertEqual(len(cases), 256)
        self.assertEqual(len({case["name"] for case in cases}), len(cases))
        self.assertEqual({c["tag_id"] for c in cases if c["group"] == "field_tags"}, tags)

    def test_sweep_cartesian_product(self):
        spec = {"sweeps": [{"name": "grid", "tag_ids": [1, 2],
                 "parameters": {"distance_m": [1, 2], "lateral_m": [-1, 1]}}]}
        cases = expand_cases(spec, {1, 2})
        self.assertEqual({(c["tag_id"], c["distance_m"], c["lateral_m"]) for c in cases},
                         {(t, d, l) for t in [1, 2] for d in [1, 2] for l in [-1, 1]})

    def test_invalid_sweep_rejected(self):
        for parameters, ids in [({"distance_m": []}, [1]), ({"distance_m": [1]}, [99]),
                                 ({"name": ["override"]}, [1])]:
            with self.subTest(parameters=parameters, ids=ids), self.assertRaises(ValueError):
                expand_cases({"sweeps": [{"name": "invalid", "tag_ids": ids,
                                         "parameters": parameters}]}, {1})

    def test_empty_suite_rejected(self):
        with self.assertRaises(ValueError):
            expand_cases({}, {1})

    def test_named_translation_axes(self):
        tag = tag_geometry({"ID": 1, "pose": {"translation": {"z": 3, "x": 1, "y": 2},
              "rotation": {"quaternion": {"W": 1, "X": 0, "Y": 0, "Z": 0}}}})
        self.assertEqual(tag["position"], [1, 2, 3])

    def test_target_projects_to_principal_point(self):
        tag = {"position": [1, 2, 3], "rotation": np.eye(3).tolist()}
        eye, r = camera_pose(tag, {"distance_m": 2, "lateral_m": 0.4})
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1)
        c = {"fx": 600, "fy": 590, "cx": 960, "cy": 470}
        uv, depth = project(np.array([tag["position"]]), eye, r, c)
        np.testing.assert_allclose(uv[0], [960, 470])
        self.assertGreater(depth[0], 0)

    def test_reproducible_images_and_independent_tag_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = [Path(temp) / name for name in ("a", "b")]
            for out in (first, second):
                subprocess.run([sys.executable, str(HERE / "generate.py"), "--output", str(out)], check=True,
                               stdout=subprocess.DEVNULL)
            manifest = json.loads((first / "manifest.json").read_text())
            detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
            for frame in manifest["frames"]:
                self.assertEqual((first / frame["image"]).read_bytes(), (second / frame["image"]).read_bytes())
                image = cv2.imread(str(first / frame["image"]), cv2.IMREAD_GRAYSCALE)
                _, ids, _ = detector.detectMarkers(image)
                found = [] if ids is None else ids.ravel().tolist()
                for expected in frame["expected_ids"]:
                    self.assertIn(expected, found, frame["name"])
                if not frame["expected_ids"]:
                    self.assertEqual(found, [], frame["name"])
            self.assertEqual(manifest["camera"]["k1"], 0)


if __name__ == "__main__":
    unittest.main()
