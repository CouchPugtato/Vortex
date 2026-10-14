"""Prepare a Blender scene manifest, or render fast pinhole fixtures without Blender."""
import argparse
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Map tag: x outward, y right when viewed from front, z up.
# Solver tag basis matches src/main.rs::map_tag_from_solver_tag_rotation.
MAP_FROM_SOLVER = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)


def expand_cases(spec, tag_ids):
    """Expand configurable Cartesian sweeps into named, reproducible cases."""
    cases = [dict(case, group=case.get("group", "baseline")) for case in spec.get("cases", [])]
    for sweep in spec.get("sweeps", []):
        ids = sweep.get("tag_ids", [1])
        if ids == "all":
            ids = sorted(tag_ids)
        if not isinstance(ids, list) or not ids or any(tag not in tag_ids for tag in ids):
            raise ValueError(f"Invalid tag_ids in sweep: {sweep['name']}")
        parameters = sweep["parameters"]
        if any(not isinstance(values, list) or not values for values in parameters.values()):
            raise ValueError(f"Sweep parameters must be nonempty lists: {sweep['name']}")
        if "name" in parameters or "tag_id" in parameters or "group" in parameters:
            raise ValueError("Sweep parameters cannot override name, tag_id, or group")
        for i, values in enumerate(product(ids, *parameters.values())):
            cases.append(dict(zip(parameters, values[1:]), tag_id=values[0],
                              name=f"{sweep['name']}_{i:04d}", group=sweep["name"]))
    if not cases:
        raise ValueError("Scenario suite is empty")
    return cases


def rotation(q):
    w, x, y, z = [q[k] for k in ("W", "X", "Y", "Z")]
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0:
        raise ValueError("Zero field-map quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tag_geometry(raw):
    p = raw["pose"]
    return {"id": raw["ID"], "position": [p["translation"][k] for k in ("x", "y", "z")],
            "rotation": rotation(p["rotation"]["quaternion"]).tolist()}


def camera_pose(tag, case):
    r = np.array(tag["rotation"])
    target = np.array(tag["position"])
    eye = target + r @ np.array([case["distance_m"], case.get("lateral_m", 0),
                                case.get("vertical_m", 0)])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return eye, np.column_stack((right, down, forward))


def corners(tag, size):
    s = size / 2
    # Texture TL, TR, BR, BL. Positive solver y is image down.
    local = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
    r = np.array(tag["rotation"]) @ MAP_FROM_SOLVER
    return local @ r.T + tag["position"]


def project(points, eye, r, c):
    pc = (points - eye) @ r
    uv = np.column_stack((c["fx"] * pc[:, 0] / pc[:, 2] + c["cx"],
                          c["fy"] * pc[:, 1] / pc[:, 2] + c["cy"]))
    return uv.astype(np.float32), pc[:, 2]


def degrade(image, case, rng):
    sigma = case.get("blur_sigma", 0)
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    image = image.astype(float) * case.get("brightness", 1)
    if case.get("noise_sigma", 0) > 0:
        image += rng.normal(0, case["noise_sigma"], image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def tag_texture(dictionary, tag_id):
    # OpenCV's AprilTag dictionary is rotated 180 degrees relative to official
    # AprilRobotics images. Match native AprilTag pose/corner conventions.
    marker = cv2.rotate(cv2.aruco.generateImageMarker(dictionary, tag_id, 320), cv2.ROTATE_180)
    return cv2.copyMakeBorder(marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)


def render_fast(frame, tags, camera, width, height, textures):
    image = np.full((height, width), 180, dtype=np.uint8)
    eye, r = np.array(frame["camera_position"]), np.array(frame["camera_rotation"])
    expected = []
    if frame["case"].get("hide_tags"):
        return image, expected
    for tag in sorted(tags, key=lambda t: np.linalg.norm(np.array(t["position"])-eye), reverse=True):
        normal = np.array(tag["rotation"])[:, 0]
        if np.dot(eye - tag["position"], normal) <= 0:
            continue
        uv, depth = project(corners(tag, camera["tag_size_m"]), eye, r, camera)
        if np.any(depth <= 0.05):
            continue
        # Only fully visible tags count as required detections in baseline fixtures.
        full = bool(np.all(uv[:, 0] > 8) and np.all(uv[:, 0] < width-8)
                    and np.all(uv[:, 1] > 8) and np.all(uv[:, 1] < height-8))
        if full and abs(cv2.contourArea(uv)) >= 225:
            expected.append(tag["id"])
        # 320px black-border tag plus a 40px white margin, physically scaled.
        padded, _ = project(corners(tag, camera["tag_size_m"] * 1.25), eye, r, camera)
        src = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32)
        h = cv2.getPerspectiveTransform(src, padded)
        warped = cv2.warpPerspective(textures[tag["id"]], h, (width, height),
                                     flags=cv2.INTER_LINEAR, borderValue=180)
        mask = cv2.warpPerspective(np.full((400, 400), 255, np.uint8), h, (width, height))
        image[mask > 250] = warped[mask > 250]
    return image, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fast", "blender"), default="fast")
    parser.add_argument("--config", type=Path, default=HERE.parent / "config/config.json")
    parser.add_argument("--field", type=Path, default=HERE.parent / "config/2026-rebuilt-welded.json")
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json")
    parser.add_argument("--output", type=Path, default=HERE / "generated/fast")
    args = parser.parse_args()
    spec = json.loads(args.scenarios.read_text())
    runtime = json.loads(args.config.read_text())
    camera = dict(runtime.get("camera_profiles", {}).get(str(spec["camera_index"]), runtime["camera"]))
    # Baseline renders use an ideal pinhole; never pair them with real lens distortion.
    for key in ("k1", "k2", "k3", "p1", "p2"):
        camera[key] = 0.0
    runtime["camera"] = camera
    runtime["camera_profiles"] = {str(spec["camera_index"]): camera}
    tags = [tag_geometry(t) for t in json.loads(args.field.read_text())["tags"]]
    by_id = {t["id"]: t for t in tags}
    width, height = spec["width"], spec["height"]
    if width <= 0 or height <= 0 or camera["fx"] <= 0 or camera["fy"] <= 0:
        raise ValueError("Invalid image dimensions or focal lengths")
    out = args.output.resolve()
    # Reusing a populated directory could mix frames from different scene settings.
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)
    textures = {}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag in tags:
        texture = tag_texture(dictionary, tag["id"])
        textures[tag["id"]] = texture
        cv2.imwrite(str(out / "textures" / f"{tag['id']}.png"), texture)
    rng = np.random.default_rng(spec.get("seed", 2026))
    frames = []
    names = set()
    cases = expand_cases(spec, by_id)
    for i, case in enumerate(cases):
        name = case["name"]
        if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name) or name in names:
            raise ValueError(f"Invalid or duplicate case name: {name}")
        names.add(name)
        if case["distance_m"] <= 0:
            raise ValueError("distance_m must be positive")
        eye, r = camera_pose(by_id[case["tag_id"]], case)
        frame = {"name": name, "group": case["group"], "image": f"images/{i:04d}_{name}.png", "case": case,
                 "camera_position": eye.tolist(), "camera_rotation": r.tolist(),
                 "expected_ids": []}
        image, expected = render_fast(frame, tags, camera, width, height, textures)
        frame["projected_ids"] = expected
        frame["expected_ids"] = [] if case.get("hide_tags") else [case["tag_id"]]
        if frame["expected_ids"] and case["tag_id"] not in expected:
            raise ValueError(f"Target tag is too small or outside image: {name}")
        if args.backend == "fast":
            cv2.imwrite(str(out / frame["image"]), degrade(image, case, rng))
        frames.append(frame)
    (out / "runtime_config.json").write_text(json.dumps(runtime, indent=2))
    manifest = {"schema_version": 1, "backend": args.backend, "width": width, "height": height,
                "camera_index": spec["camera_index"], "seed": spec.get("seed", 2026),
                "camera": camera, "tags": tags, "frames": frames}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(frames)} cases in {out}")
    if args.backend == "blender":
        print(f'Next: blender --background --python "{HERE / "render_blender.py"}" -- --dataset "{out}"')


if __name__ == "__main__":
    main()
