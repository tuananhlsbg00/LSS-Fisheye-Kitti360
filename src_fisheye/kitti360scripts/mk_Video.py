import cv2
import os

# Define the folder containing images
image_folder = 'labels'  # Replace with your folder path
output_video = 'output_video.mp4'
frame_rate = 6  # Adjust as needed

# Get the list of image files and sort them
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])[0:300]

# Read the first image to get dimensions
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Upscale dimensions
new_width, new_height = width * 5, height * 5

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'XVID' (for .avi)
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (new_width, new_height))


# Process each image
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)

    # Upscale image (use INTER_CUBIC for better quality)
    upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Write frame to video
    video_writer.write(upscaled_frame)

# Release the writer
video_writer.release()
print("Video saved as", output_video)
