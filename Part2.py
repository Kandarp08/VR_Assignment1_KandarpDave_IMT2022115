import cv2 
import matplotlib.pyplot as plt

# List of input images
image_paths = ["./Images/Part2_1.jpg", "./Images/Part2_2.jpg", "./Images/Part2_3.jpg"] 

# Input list for stitch function 
images = [] 

for path in image_paths:

    input_image = cv2.imread(path)
    images.append(input_image)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Find out keypoints in the image
    features = cv2.SIFT_create()
    keypoints = features.detect(gray_image, None) 

    # Image with keypoints
    image_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, 0, (0, 0, 255), 
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Display the keypoints in the image
    cv2.imshow(f"Image {len(images)} with keypoints", image_with_keypoints)
    cv2.imwrite(f"./Output/Part2_{len(images)}_Keypoints.png", image_with_keypoints)

    cv2.waitKey(0)

# Stitch images to form panorama
stitchy = cv2.Stitcher.create()
(res, panorama) = stitchy.stitch(images) 

if res != cv2.STITCHER_OK: 
	print("Stitching not successful.")

else: 
	print("Stitching successful.") 

plt.title("Stitched Panorama")
plt.imshow(panorama)
plt.axis("off")
plt.show()

cv2.imwrite("./Output/Part2_Panorama.png", panorama)