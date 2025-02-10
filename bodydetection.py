import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Parameters
SLOUCH_ANGLE_THRESHOLD = 5  # <-- Lowered threshold for easier testing
HAND_MOVEMENT_THRESHOLD = 30  # distance in pixels to consider hand movement

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three points: a, b, c, 
    with b as the vertex (the middle point).
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point
    
    # Vectors from b to a and b to c
    ba = a - b
    bc = c - b
    
    # Dot product and magnitude
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if mag_ba * mag_bc == 0:
        return 0.0
    
    # Calculate the angle in radians, then convert to degrees
    cos_angle = dot_product / (mag_ba * mag_bc)
    # Clamp to [-1, 1] to avoid floating-point errors
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def main():
    cap = cv2.VideoCapture(0)  # Use your default camera
    
    # Store previous wrist positions (for detecting hand movement)
    prev_left_wrist = None
    prev_right_wrist = None
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            # Convert back to BGR for display
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if result.pose_landmarks:
                h, w, _ = image_bgr.shape
                landmarks = result.pose_landmarks.landmark
                
                # Read relevant landmarks (no face)
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                
                # Convert normalized to pixel coords
                left_shoulder_xy = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                right_shoulder_xy = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                left_hip_xy = (int(left_hip.x * w), int(left_hip.y * h))
                right_hip_xy = (int(right_hip.x * w), int(right_hip.y * h))
                left_wrist_xy = (int(left_wrist.x * w), int(left_wrist.y * h))
                right_wrist_xy = (int(right_wrist.x * w), int(right_wrist.y * h))
                
                # --- Slouching Detection ---
                midpoint_shoulders = (
                    (left_shoulder_xy[0] + right_shoulder_xy[0]) // 2,
                    (left_shoulder_xy[1] + right_shoulder_xy[1]) // 2
                )
                midpoint_hips = (
                    (left_hip_xy[0] + right_hip_xy[0]) // 2,
                    (left_hip_xy[1] + right_hip_xy[1]) // 2
                )
                
                # Define a vertical reference point above midpoint_hips
                vertical_ref_point = (midpoint_hips[0], midpoint_hips[1] - 100)
                
                angle_torso = calculate_angle(
                    midpoint_shoulders,
                    midpoint_hips,
                    vertical_ref_point
                )
                is_slouching = angle_torso > SLOUCH_ANGLE_THRESHOLD
                
                # --- Hand Movement Detection ---
                hand_movement_detected = False
                if prev_left_wrist is not None:
                    dist_left = np.linalg.norm(
                        np.array(left_wrist_xy) - np.array(prev_left_wrist)
                    )
                    if dist_left > HAND_MOVEMENT_THRESHOLD:
                        hand_movement_detected = True
                
                if prev_right_wrist is not None:
                    dist_right = np.linalg.norm(
                        np.array(right_wrist_xy) - np.array(prev_right_wrist)
                    )
                    if dist_right > HAND_MOVEMENT_THRESHOLD:
                        hand_movement_detected = True
                
                # Update previous positions
                prev_left_wrist = left_wrist_xy
                prev_right_wrist = right_wrist_xy
                
                # --- Drawing Landmarks & Connections ---
                # 1) Draw circles for each keypoint
                points = {
                    'left_shoulder': left_shoulder_xy,
                    'right_shoulder': right_shoulder_xy,
                    'left_hip': left_hip_xy,
                    'right_hip': right_hip_xy,
                    'left_wrist': left_wrist_xy,
                    'right_wrist': right_wrist_xy
                }
                
                for point in points.values():
                    cv2.circle(image_bgr, point, 6, (0, 255, 0), -1)
                
                # 2) Draw lines connecting these keypoints
                connections = [
                    ('left_shoulder', 'right_shoulder'),
                    ('left_shoulder', 'left_hip'),
                    ('right_shoulder', 'right_hip'),
                    ('left_shoulder', 'left_wrist'),
                    ('right_shoulder', 'right_wrist'),
                    ('left_hip', 'right_hip')
                ]
                
                for (p1, p2) in connections:
                    pt1 = points[p1]
                    pt2 = points[p2]
                    cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2)
                
                # --- Status Text ---
                # 1) Slouch / Posture
                if is_slouching:
                    cv2.putText(
                        image_bgr, 
                        "Slouching Detected",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                else:
                    cv2.putText(
                        image_bgr, 
                        "Posture OK",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                # 2) Show the angle to help debug posture
                angle_text = f"Torso Angle: {angle_torso:.1f} deg"
                cv2.putText(
                    image_bgr,
                    angle_text,
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                # 3) Hand Movement
                if hand_movement_detected:
                    cv2.putText(
                        image_bgr,
                        "Hand Movement Detected",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                else:
                    cv2.putText(
                        image_bgr,
                        "Hands Still",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
            # Show the result
            cv2.imshow('Body Posture & Hand Movement Detection (No Face)', image_bgr)
            
            # Press 'q' to quit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
