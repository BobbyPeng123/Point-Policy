import cv2

# Path to the input video and output image
video_path = "/home/bobby/data/processed_data/pick_bottle_from_rack/demonstration_0/videos/camera2.mp4"
output_image_path = "first_frame.jpg"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if ret:
    # Save the first frame as a JPEG image
    cv2.imwrite(output_image_path, frame)
else:
    print("Failed to read the video or no frames found.")

# Release the video capture object
cap.release()
