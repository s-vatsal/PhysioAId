import cv2
from posedetect import detectPose
from classifyutil import classifyStageSim
from mpinit import mp_pose
import matplotlib.pyplot as plt
from plot3D import plot_world_landmarks
import time

stages = [1, 2, 3, 4]
index = 0

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

complete = False
# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

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
        frame, _, complete = classifyStageSim(landmarks, frame, stages[index], display=False)

    if world_landmarks is not None:
        plot_world_landmarks(plt, ax, world_landmarks)
    
    # Display the frame.
    cv2.imshow('Pose Classification', frame)

    if complete:
        cv2.waitKey(3000)

    if complete and index < len(stages) - 1:
        index += 1
    elif not complete and index < len(stages) - 1:
        pass
    elif complete and index == len(stages) - 1:
        break

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()