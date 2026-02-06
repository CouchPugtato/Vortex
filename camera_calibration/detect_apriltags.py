import cv2
import numpy as np
import json
import time

CAM_FX = 709.5  # pixels (calibrated from 3.0m real dist / 2.542m measured @ 601.2)
CAM_FY = 709.5  # pixels
CAM_CX = 960.0  # principal point x
CAM_CY = 600.0  # principal point y
TAG_SIZE_M = 0.16  # m

# Global state for distortion toggle
USE_DISTORTION = False  # Start with NO distortion correction to check linearity

def load_distortion_coefficients(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            dist_coeff = np.array(data['dist_coeff'])
            return dist_coeff
    except Exception as e:
        print(f"Error loading distortion coefficients from {json_path}: {e}")
        sys.exit(1)

def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap

def main():
    # Load distortion coefficients from result4.json (or similar)
    dist_coeffs_loaded = load_distortion_coefficients('result5.json') # Changed to result5.json as per logs, check if result4 is needed
    print(f"Loaded distortion coefficients from result5.json")
    print(dist_coeffs_loaded)
    
    global USE_DISTORTION
    global CAM_FX, CAM_FY

    # Initialize AprilTag detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Initialize Camera
    current_camera_index = 0
    cap = open_camera(current_camera_index)
    if cap is None:
        return

    # Create Window
    window_name = "AprilTag Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # Main Loop
    while True:
        if cap is None:
             break
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Decide which distortion coefficients to use
        if USE_DISTORTION:
            current_dist_coeffs = dist_coeffs_loaded
            dist_status = "ON"
            dist_color = (0, 255, 0)
        else:
            current_dist_coeffs = np.zeros(5)
            dist_status = "OFF"
            dist_color = (0, 0, 255)

        # Check actual resolution
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Adjust Camera Matrix based on actual resolution
        # User constants are for 1920x1080
        scale_x = actual_width / 1920.0
        scale_y = actual_height / 1080.0
        
        # If resolution is different, scale the matrix
        # Note: If the camera supports 1920x1080, scale will be 1.0
        final_fx = CAM_FX * scale_x
        final_fy = CAM_FY * scale_y
        final_cx = actual_width / 2.0
        final_cy = actual_height / 2.0
        
        camera_matrix = np.array([
            [final_fx, 0, final_cx],
            [0, final_fy, final_cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        corners, ids, rejected = detector.detectMarkers(gray)

        if len(corners) > 0:
            # Flatten the ArUco IDs list
            ids = ids.flatten()
            
            # Loop over the detected ArUco corners
            for i, corner in enumerate(corners):
                # Extract the marker corners
                corners_single = corner.reshape((4, 2))
                
                # Sub-pixel refinement for better precision
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # We need to pass float32 corners
                corners_subpix = cv2.cornerSubPix(gray, corners_single.astype(np.float32), (5, 5), (-1, -1), criteria)
                
                (topLeft, topRight, bottomRight, bottomLeft) = corners_subpix
                
                # Convert to integer for drawing
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # Draw the bounding box of the ArUco detection
                cv2.line(display_frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(display_frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(display_frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(display_frame, bottomLeft, topLeft, (0, 255, 0), 2)
                
                # Compute the center (using subpix corners)
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(display_frame, (cX, cY), 4, (0, 0, 255), -1)
                
                # Prepare object points (3D points of tag corners in tag coordinate system)
                obj_points = np.array([
                    [-TAG_SIZE_M / 2, TAG_SIZE_M / 2, 0],
                    [TAG_SIZE_M / 2, TAG_SIZE_M / 2, 0],
                    [TAG_SIZE_M / 2, -TAG_SIZE_M / 2, 0],
                    [-TAG_SIZE_M / 2, -TAG_SIZE_M / 2, 0]
                ], dtype=np.float32)
                
                # Image points must be float32
                image_points = corners_subpix.reshape((4, 2))
                
                # Solve PnP
                # Use ITERATIVE for better stability with potential distortion issues
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    image_points, 
                    camera_matrix, 
                    current_dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Draw axis (X: Red, Y: Green, Z: Blue)
                    # Use a larger length for axis to be visible
                    cv2.drawFrameAxes(display_frame, camera_matrix, current_dist_coeffs, rvec, tvec, 0.1)
                    
                    x, y, z = tvec.flatten()
                    
                    # Fix negative Z if it occurs (flipped solution)
                    # In standard view, Z should be positive (in front of camera)
                    if z < 0:
                        z = -z
                        x = -x
                        y = -y

                    z = z - 0.105755

                    # 1. Draw text near the tag
                    text_dist = f"Dist: {z:.2f}m"
                    text_pos_dist = (int(image_points[0][0]), int(image_points[0][1]) - 25)
                    cv2.putText(display_frame, text_dist, text_pos_dist, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                   
                    h, w = display_frame.shape[:2]
                    overlay_x = 10
                    overlay_y = h - 130
                    
                    cv2.rectangle(display_frame, (overlay_x - 5, overlay_y - 25), (overlay_x + 350, overlay_y + 110), (50, 50, 50), -1)
                    
                    cv2.putText(display_frame, f"Tag ID: {ids[i]}", (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"X Offset: {x:.3f} m", (overlay_x, overlay_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red X
                    cv2.putText(display_frame, f"Y Offset: {y:.3f} m", (overlay_x, overlay_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green Y
                    cv2.putText(display_frame, f"Z Offset: {z:.3f} m", (overlay_x, overlay_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Blue Z
                    
                    print(f"Tag ID {ids[i]}: x={x:.3f}, y={y:.3f}, z={z:.3f}")

        cv2.putText(display_frame, f"Cam: {current_camera_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Distortion: {dist_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            USE_DISTORTION = not USE_DISTORTION
            print(f"Distortion Correction: {USE_DISTORTION}")
        elif key == ord('n'):
            # Switch camera
            print("Switching camera...")
            if cap:
                cap.release()
            
            time.sleep(0.5) # resource release delay
            
            current_camera_index += 1
            if current_camera_index > 5:
                current_camera_index = 0
            
            cap = open_camera(current_camera_index)
            if cap is None:
                print(f"Camera {current_camera_index} not found, wrapping to 0")
                current_camera_index = 0
                cap = open_camera(current_camera_index)
                
            if cap is None:
                 print("Could not open any camera.")
                 break
        elif key == ord('='): # + key
            CAM_FX += 10.0
            CAM_FY += 10.0
            print(f"Increased FX/FY to {CAM_FX}")
        elif key == ord('-'): # - key
            CAM_FX -= 10.0
            CAM_FY -= 10.0
            print(f"Decreased FX/FY to {CAM_FX}")
        elif key == ord('p'):
            print(f"Current Settings: FX={CAM_FX}, FY={CAM_FY}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
