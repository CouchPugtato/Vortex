"""Check image/geometry conventions independently of Vortex's Rust solver."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from generate import HERE, camera_pose, project, tag_geometry, expand_cases, tag_texture


class GeneratorTests(unittest.TestCase):
    def test_stress_suite_coverage_and_seeded_cases(self):
        tags = {t["ID"] for t in json.loads((HERE.parent / "config/2026-rebuilt-welded.json").read_text())["tags"]}
        cases = expand_cases(json.loads((HERE / "scenarios_stress.json").read_text()), tags)
        self.assertEqual(len(cases), 1024)
        self.assertEqual(len({c["name"] for c in cases}), 1024)
        self.assertEqual(sum(c.get("single_tag_only", False) for c in cases), 832)
        self.assertEqual({c["tag_id"] for c in cases if c["group"] == "single_geometry"}, tags)
        self.assertTrue(all("noise_seed" in c for c in cases))

    def test_case_defaults_can_be_overridden(self):
        spec = {"case_defaults": {"noise_seed": 2026}, "cases": [
            {"name": "custom", "group": "custom", "noise_seed": 42}],
            "sweeps": [{"name": "grid", "parameters": {"distance_m": [1], "noise_seed": [7]}}]}
        cases = expand_cases(spec, {1})
        self.assertEqual([c["noise_seed"] for c in cases], [42, 7])
        self.assertEqual(cases[0]["group"], "custom")

    def test_single_tag_scene_and_isolated_noise_reproduction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"width": 1920, "height": 1080, "camera_index": 0, "seed": 2026,
                    "cases": [{"name": "single", "tag_id": 1, "distance_m": 1.0,
                               "single_tag_only": True, "noise_sigma": 3, "noise_seed": 42},
                              {"name": "other_seed", "tag_id": 1, "distance_m": 1.0,
                               "single_tag_only": True, "noise_sigma": 3, "noise_seed": 43}]}
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            for folder, selector in [("all", []), ("one", ["--case", "single"])]:
                subprocess.run([sys.executable, str(HERE / "generate.py"), "--scenarios", str(spec_path),
                                "--output", str(root / folder), *selector], check=True,
                               stdout=subprocess.DEVNULL)
            full = json.loads((root / "all/manifest.json").read_text())
            one = json.loads((root / "one/manifest.json").read_text())
            f = full["frames"][0]
            self.assertEqual(f["allowed_ids"], [1])
            self.assertEqual(f["projected_ids"], [1])
            self.assertEqual((root / "all" / f["image"]).read_bytes(),
                             (root / "one" / one["frames"][0]["image"]).read_bytes())
            self.assertNotEqual((root / "all" / f["image"]).read_bytes(),
                                (root / "all" / full["frames"][1]["image"]).read_bytes())
            detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
            _, ids, _ = detector.detectMarkers(cv2.imread(str(root / "all" / f["image"])))
            self.assertEqual(ids.ravel().tolist(), [1])

    def test_texture_matches_official_apriltag_orientation(self):
        reference = cv2.imread(str(HERE / "reference/tag36h11_1.png"), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(reference)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        actual = cv2.resize(tag_texture(dictionary, 1), (10, 10), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(actual, reference)

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
