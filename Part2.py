import cv2 
import matplotlib.pyplot as plt

# Match keypoints between the input images
def match_features(images, keypoints, descriptors):

    bf = cv2.BFMatcher()
    pairs = [[0, 1], [1, 2]]

    for pair in pairs:

        a = pair[0]
        b = pair[1]
    
        # Find the matches and sort them based on their distances
        matches = bf.match(descriptors[a], descriptors[b])
        matches = sorted(matches, key=lambda val: val.distance)

        # Draw the first 20 matches
        res = cv2.drawMatches(images[a], keypoints[a], images[b], keypoints[b], matches[:20], None, flags=2)
        
        cv2.imshow(f"Matches between images {a + 1} and {b + 1}", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(f"./Output/Part2_Matches_{a + 1}_{b + 1}.png", res)

# Stitch images to form panorama
def create_panorama(images):

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

# List of input images
image_paths = ["./Images/Part2_1.jpg", "./Images/Part2_2.jpg", "./Images/Part2_3.jpg"] 

# Input list for stitch function 
images = []

# List of keypoints for each input image
keypoints = []

# List of descriptors for each input image
descriptors = []

for path in image_paths:

    input_image = cv2.imread(path)
    images.append(input_image)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Find out keypoints and descriptors in the image
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_image, None) 

    keypoints.append(kp)
    descriptors.append(desc)

    # Image with keypoints
    image_with_keypoints = cv2.drawKeypoints(gray_image, kp, 0, (0, 0, 255), 
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Display the keypoints in the image
    cv2.imshow(f"Image {len(images)} with keypoints", image_with_keypoints)
    cv2.imwrite(f"./Output/Part2_{len(images)}_Keypoints.png", image_with_keypoints)

    cv2.waitKey(0)
    cv2.destroyWindow(f"Image {len(images)} with keypoints")

match_features(images, keypoints, descriptors)
create_panorama(images)