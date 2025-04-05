import cv2
import os

# Define the folder containing images
image_folder = 'visualization'  # Replace with your folder path
output_video = 'Validation_Visualization_Adjusted.mp4'
frame_rate = 6  # Adjust as needed

# Get the list of image files and sort them
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# Read the first image to get dimensions
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Upscale dimensions
new_width, new_height = width * 3, height * 3

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'XVID' (for .avi)
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))


# Process each image
for i, image in enumerate(images):
    img_path = os.path.join(image_folder, image)
    print('Loader', img_path)
    frame = cv2.imread(img_path)

    # Upscale image (use INTER_CUBIC for better quality)
    upscaled_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

    # Write frame to video
    video_writer.write(upscaled_frame)

# Release the writer
video_writer.release()
print("Video saved as", output_video)
