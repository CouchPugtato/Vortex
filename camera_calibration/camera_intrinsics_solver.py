import json
from pathlib import Path

import cv2
import numpy as np


CHECKERBOARD = (9, 6)
SQUARE_SIZE_M = 0.0222
MIN_FRAMES = 10
WINDOW_NAME = "Camera Intrinsics Calibration"
OUTPUT_JSON = "camera_intrinsics_result.json"
OUTPUT_NPZ = "camera_intrinsics_data.npz"
MAX_FRAME_RMSE_PX = 1.0
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080


def build_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    return cap


def reprojection_stats(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    total_sq_error = 0.0
    total_points = 0
    frame_errors = []

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        observed = imgp.reshape(-1, 2)
        diff = observed - projected
        sq_error = np.sum(diff * diff, axis=1)
        rmse = float(np.sqrt(np.mean(sq_error)))
        frame_errors.append(rmse)
        total_sq_error += float(np.sum(sq_error))
        total_points += observed.shape[0]

    overall_rmse = float(np.sqrt(total_sq_error / max(total_points, 1)))
    return overall_rmse, frame_errors


def calibrate_with_rejection(objpoints, imgpoints, image_size):
    kept_indices = list(range(len(objpoints)))
    rejected_frames = []

    while True:
        subset_objpoints = [objpoints[i] for i in kept_indices]
        subset_imgpoints = [imgpoints[i] for i in kept_indices]
        ok, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            subset_objpoints,
            subset_imgpoints,
            image_size,
            None,
            None,
        )
        if not ok:
            return None

        overall_rmse, frame_errors = reprojection_stats(
            subset_objpoints,
            subset_imgpoints,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
        )

        worst_local_idx = int(np.argmax(frame_errors))
        worst_rmse = frame_errors[worst_local_idx]
        if worst_rmse <= MAX_FRAME_RMSE_PX or len(kept_indices) <= MIN_FRAMES:
            return {
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "overall_rmse": overall_rmse,
                "frame_errors": frame_errors,
                "kept_indices": kept_indices,
                "rejected_frames": rejected_frames,
            }

        rejected_frames.append(
            {
                "saved_frame_index": kept_indices[worst_local_idx],
                "rmse_px": float(worst_rmse),
            }
        )
        del kept_indices[worst_local_idx]


def save_results(
    camera_index,
    image_size,
    objpoints,
    imgpoints,
    camera_matrix,
    dist_coeffs,
    rvecs,
    tvecs,
    kept_indices,
    rejected_frames,
):
    dist = dist_coeffs.reshape(-1)
    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    p1 = float(dist[2]) if len(dist) > 2 else 0.0
    p2 = float(dist[3]) if len(dist) > 3 else 0.0
    k3 = float(dist[4]) if len(dist) > 4 else 0.0

    overall_rmse, frame_errors = reprojection_stats(
        objpoints,
        imgpoints,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
    )

    result = {
        "camera_index": camera_index,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "frames_used": len(objpoints),
        "checkerboard": {
            "columns": CHECKERBOARD[0],
            "rows": CHECKERBOARD[1],
            "square_size_m": SQUARE_SIZE_M,
        },
        "camera": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "k3": k3,
        },
        "opencv": {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeff": dist_coeffs.tolist(),
        },
        "reprojection_rmse_px": overall_rmse,
        "frame_rmse_px": frame_errors,
        "kept_saved_frame_indices": kept_indices,
        "rejected_frames": rejected_frames,
        "auto_reject_max_frame_rmse_px": MAX_FRAME_RMSE_PX,
    }

    output_path = Path(OUTPUT_JSON)
    output_path.write_text(json.dumps(result, indent=2))
    np.savez(
        OUTPUT_NPZ,
        mtx=camera_matrix,
        dist=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )

    return result


def main():
    current_camera_index = 0
    cap = open_camera(current_camera_index)
    if cap is None:
        print(f"Error: Could not open camera {current_camera_index}")
        return

    objp = build_object_points()
    objpoints = []
    imgpoints = []
    image_size = None

    print("=================================================================")
    print("Camera Intrinsics Calibration Tool")
    print(f"Searching for {CHECKERBOARD[0]}x{CHECKERBOARD[1]} checkerboard corners.")
    print("This solves intrinsics and distortion together.")
    print(f"Requested capture mode: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print("-----------------------------------------------------------------")
    print("Controls:")
    print("  [S] - Save current frame (if checkerboard found)")
    print("  [C] - Calibrate using saved frames")
    print("  [N] - Switch to Next Camera")
    print("  [R] - Reset saved frames")
    print("  [Q] - Quit")
    print("=================================================================")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture image from camera {current_camera_index}")
            cv2.waitKey(500)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        display_frame = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners_refined, found)
            status_text = "Checkerboard FOUND! Press 'S' to save."
            status_color = (0, 255, 0)
        else:
            corners_refined = None
            status_text = "Searching for checkerboard..."
            status_color = (0, 0, 255)

        cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(
            display_frame,
            f"Saved Frames: {len(objpoints)} / {MIN_FRAMES}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Camera: {current_camera_index}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"Resolution: {gray.shape[1]}x{gray.shape[0]}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if not found or corners_refined is None:
                print("Cannot save: checkerboard not detected in this frame.")
                continue
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined.copy())
            print(f"Frame captured. Total saved: {len(objpoints)}")

        elif key == ord("c"):
            if len(objpoints) < MIN_FRAMES:
                print(f"Not enough frames. Need {MIN_FRAMES}, have {len(objpoints)}.")
                continue
            print("Calibrating intrinsics and distortion with automatic outlier rejection...")
            calibration = calibrate_with_rejection(objpoints, imgpoints, image_size)
            if calibration is None:
                print("Calibration failed.")
                continue

            result = save_results(
                current_camera_index,
                image_size,
                [objpoints[i] for i in calibration["kept_indices"]],
                [imgpoints[i] for i in calibration["kept_indices"]],
                calibration["camera_matrix"],
                calibration["dist_coeffs"],
                calibration["rvecs"],
                calibration["tvecs"],
                calibration["kept_indices"],
                calibration["rejected_frames"],
            )

            camera = result["camera"]
            print("\n=============================================")
            print("CALIBRATION SUCCESSFUL")
            print("=============================================")
            print(f"RMSE: {result['reprojection_rmse_px']:.4f} px")
            print(
                f"Frames kept: {len(result['kept_saved_frame_indices'])} / {len(objpoints)} "
                f"(rejected {len(result['rejected_frames'])})"
            )
            print("Camera config values:")
            print(json.dumps(camera, indent=2))
            if result["rejected_frames"]:
                print("Rejected saved frames:")
                print(json.dumps(result["rejected_frames"], indent=2))
            print(f"Saved to '{OUTPUT_JSON}' and '{OUTPUT_NPZ}'")
            print("=============================================\n")

        elif key == ord("r"):
            objpoints.clear()
            imgpoints.clear()
            print("Saved frames reset.")

        elif key == ord("n"):
            print("Switching to next camera...")
            current_camera_index += 1
            cap.release()
            cap = open_camera(current_camera_index)
            if cap is None:
                print(f"Camera {current_camera_index} not found. Looping back to 0.")
                current_camera_index = 0
                cap = open_camera(current_camera_index)
            if cap is None:
                print("Error: No cameras found.")
                break
            objpoints.clear()
            imgpoints.clear()
            print(f"Switched to camera {current_camera_index}. Saved frames reset.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
