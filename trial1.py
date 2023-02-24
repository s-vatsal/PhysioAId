import cv2
from posedetect import detectPose
from classifyutil import classifyPose
from mpinit import mp_pose
import matplotlib.pyplot as plt
from plot3D import plot_world_landmarks

# init plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Define threshold angles
MIN_ANGLE = 30
MAX_ANGLE = 70

# Initialize variables to keep track of exercise progress
exercise_count = 0
exercise_reps = 5

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks, world_landmarks = detectPose(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
    if landmarks:
        
        # Perform the Pose Classification.
        frame, _ = classifyPose(landmarks, frame, display=False)
        
        # Calculate left and right leg angles.
        left_leg_angle = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility * (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y - landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
        right_leg_angle = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility * (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
        

        # Check if the leg angle is within the threshold range.
        if left_leg_angle > MIN_ANGLE and left_leg_angle < MAX_ANGLE:
                exercise_count += 1
                print(f"Exercise count: {exercise_count}")
                
                # If the user has completed the required number of reps, reset the exercise count and print a message.
                if exercise_count == exercise_reps:
                    exercise_count = 0
                    print("Congratulations, you've completed the exercise!")
    
    if world_landmarks is not None:
        plot_world_landmarks(plt, ax, world_landmarks)
    
    # Display the frame.
    cv2.imshow('Pose Classification', frame)

    k = cv2.waitKey(1) & 0xFF

    if(k == 27):

        break

camera_video.release()
cv2.destroyAllWindows()
