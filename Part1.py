import numpy as np
import cv2
import matplotlib.pyplot as plt

# Count the number of coins in the image
def count_coins(edges):

    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)

image = cv2.imread("./Images/coins.jpeg")            # Input image (BGR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale

# Apply Gaussian filter using a 5x5 kernel with a standard deviation of 1.5
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5) 

# Find edges using Canny algorithm
edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

# Convert edges from grayscale to BGR
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Outline the edges in the original image
combined_image = cv2.add(image, edges_colored)

# Display result of Canny edge detection
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Edge Detection")
plt.imshow(combined_image)
plt.axis("off")

# Obtain the binary image from grayscale image
ret, binary_image = cv2.threshold(gray_image, 130, 200, cv2.THRESH_BINARY)

# Noise removal using a 5x5 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1) # Apply smoothing on the binary image

# Apply dilation to the binary image
sure_bg = cv2.dilate(binary_image, kernel, iterations=1)
 
# Distance transform
dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
 
# Foreground area
ret, sure_fg = cv2.threshold(dist, 0.01 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)  
 
# Unknown area
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
 
# Add one to all labels so that background is not 0, but 1
markers += 1

# Mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply the watershed algorithm
markers = cv2.watershed(image, markers)

# Display result of region segmentation
plt.subplot(1, 2, 2)
plt.title("Region Segmentation")
plt.imshow(markers)
plt.axis("off")

plt.savefig("./Output/Part1.png")

print(f"Number of coins = {count_coins(edges)}") # Print the number of coins detected

plt.show()