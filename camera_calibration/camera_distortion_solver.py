import cv2
import numpy as np
import os
import json

def main():
    CHECKERBOARD = (9, 6) # number of internal squares
    
    SQUARE_SIZE = 0.0222 # 22.2mm, only affects the translation vector unit, not distortion/matrix.
    
    current_camera_index = 0
    
    MIN_FRAMES = 10 # minimum needed for calculating

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    cap = cv2.VideoCapture(current_camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {current_camera_index}")
        return

    print("=================================================================")
    print(f"Camera Calibration Tool")
    print(f"Searching for {CHECKERBOARD[0]}x{CHECKERBOARD[1]} checkerboard corners.")
    print("-----------------------------------------------------------------")
    print("Controls:")
    print("  [S] - Save current frame (if checkerboard found)")
    print("  [C] - Calibrate using saved frames (requires at least 10)")
    print("  [N] - Switch to Next Camera")
    print("  [Q] - Quit")
    print("=================================================================")

    valid_frames = 0
    
    cv2.namedWindow('Camera Calibration', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture image from camera {current_camera_index}")
            cv2.waitKey(500)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(
            gray, 
            CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display_frame = frame.copy()

        # if found, draw corners
        if ret_corners:
            # refine corner locations
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            # draw corners to display
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_corners)
            status_text = "Checkerboard FOUND! Press 'S' to save."
            color = (0, 255, 0)
        else:
            status_text = "Searching for checkerboard..."
            color = (0, 0, 255)

        # UI Overlay
        cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Saved Frames: {valid_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Camera Calibration', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # [S] Save Frame
        if key == ord('s'):
            if ret_corners:
                objpoints.append(objp)
                imgpoints.append(corners2)
                valid_frames += 1
                print(f"Frame captured! Total: {valid_frames}")
            else:
                print("Cannot save: Checkerboard not detected in this frame.")

        # [C] Calibrate
        elif key == ord('c'):
            if valid_frames < MIN_FRAMES:
                print(f"Not enough frames. Need {MIN_FRAMES}, have {valid_frames}.")
            else:
                print("Calibrating... This may take a moment.")
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                print("\n=============================================")
                print("CALIBRATION SUCCESSFUL!")
                print("=============================================")
                print(f"Reprojection Error: {ret}")
                print("\nCamera Matrix (K):\n", mtx)
                print("\nDistortion Coefficients (D) [k1, k2, p1, p2, k3]:\n", dist)
                
                # save to file
                data = {
                    "camera_matrix": mtx.tolist(),
                    "dist_coeff": dist.tolist(),
                    "reprojection_error": ret
                }
                
                with open("calibration_result.json", "w") as f:
                    json.dump(data, f, indent=4)
                
                np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                
                print("\nResults saved to 'calibration_result.json' and 'calibration_data.npz'")
                print("=============================================")

        # [N] Next Camera
        elif key == ord('n'):
            print("Switching to next camera...")
            current_camera_index += 1
            cap.release()
            cap = cv2.VideoCapture(current_camera_index)
            
            # if next camera not found, loop back to 0
            if not cap.isOpened():
                print(f"Camera {current_camera_index} not found. looping back to 0.")
                current_camera_index = 0
                cap = cv2.VideoCapture(current_camera_index)
            
            if cap.isOpened():
                print(f"Successfully switched to Camera {current_camera_index}")
                # reset calibration data because we changed cameras
                objpoints = []
                imgpoints = []
                valid_frames = 0
                print("WARNING: Calibration frames reset due to camera switch.")
            else:
                print("Error: No cameras found!")
                break

        # [Q] Quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
