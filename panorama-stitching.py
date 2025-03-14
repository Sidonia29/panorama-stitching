import cv2
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################################################################################

# ~ INITIALIZING  ~

# Load the images I need
img1 = cv2.imread(r'C:/Users/sandr/Documents/FACULTATE/ANUL 4/SEM 1/IPIVA/TEMA 2/set5/4.png')
img2 = cv2.imread(r'C:/Users/sandr/Documents/FACULTATE/ANUL 4/SEM 1/IPIVA/TEMA 2/set5/5.png')
img3 = cv2.imread(r'C:/Users/sandr/Documents/FACULTATE/ANUL 4/SEM 1/IPIVA/TEMA 2/set5/9.png')
img4 = cv2.imread(r'C:/Users/sandr/Documents/FACULTATE/ANUL 4/SEM 1/IPIVA/TEMA 2/set5/2.png')
img5 = cv2.imread(r'C:/Users/sandr/Documents/FACULTATE/ANUL 4/SEM 1/IPIVA/TEMA 2/set5/11.png')

# Display initial images
plt.figure()
plt.subplot(151), plt.imshow(img1[:, :, ::-1]), plt.title("Image 1")
plt.subplot(152), plt.imshow(img2[:, :, ::-1]), plt.title("Image 2")
plt.subplot(153), plt.imshow(img3[:, :, ::-1]), plt.title("Image 3")
plt.subplot(154), plt.imshow(img4[:, :, ::-1]), plt.title("Image 4")
plt.subplot(155), plt.imshow(img5[:, :, ::-1]), plt.title("Image 5")
plt.show()

# Initialize ORB and BFMatcher

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15

# Convert images to grayscale

img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3Gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4Gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5Gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

###########################################################################################################################################################################

# ~ KEYPOINTS & MATCHES ~


# Detect ORB features and compute descriptors

orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2Gray, None)
keypoints3, descriptors3 = orb.detectAndCompute(img3Gray, None)
keypoints4, descriptors4 = orb.detectAndCompute(img4Gray, None)
keypoints5, descriptors5 = orb.detectAndCompute(img5Gray, None)


# Draw keypoints between img1 and img2

img1Keypoints = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2Keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.subplot(121), plt.imshow(img1Keypoints[:,:,::-1]), plt.title('Keypoints of Image 1')
plt.subplot(122), plt.imshow(img2Keypoints[:,:,::-1]), plt.title('Keypoints of Image 2')
plt.show()

# Match features between img1 and img2
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches12 = matcher.match(descriptors1, descriptors2, None)
matches12.sort(key=lambda x: x.distance, reverse=False)
numGoodMatches12 = int(len(matches12) * GOOD_MATCH_PERCENT)
matches12 = matches12[:numGoodMatches12]
# Draw top matches
im12Matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches12, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("matches12.jpg", im12Matches)
plt.figure()
plt.imshow(im12Matches[:,:,::-1]), plt.title('Matches Between img1 & img2')
plt.show()


# Draw keypoints between img2 and img3
img2Keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img3Keypoints = cv2.drawKeypoints(img3, keypoints3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.subplot(121), plt.imshow(img2Keypoints[:,:,::-1]), plt.title('Keypoints of Image 2')
plt.subplot(122), plt.imshow(img3Keypoints[:,:,::-1]), plt.title('Keypoints of Image 3')
plt.show()

# Match features between img2 and img3
matches23 = matcher.match(descriptors2, descriptors3, None)
matches23.sort(key=lambda x: x.distance, reverse=False)
numGoodMatches23 = int(len(matches23) * GOOD_MATCH_PERCENT)
matches23 = matches23[:numGoodMatches23]
# Draw top matches
im23Matches = cv2.drawMatches(img2, keypoints2, img3, keypoints3, matches23, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("matches23.jpg", im23Matches)
plt.figure()
plt.imshow(im23Matches[:,:,::-1]), plt.title('Matches Between img2 & img3')
plt.show()


# Draw keypoints between img3 and img4
img3Keypoints = cv2.drawKeypoints(img3, keypoints3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img4Keypoints = cv2.drawKeypoints(img4, keypoints4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.subplot(121), plt.imshow(img3Keypoints[:,:,::-1]), plt.title('Keypoints of Image 3')
plt.subplot(122), plt.imshow(img4Keypoints[:,:,::-1]), plt.title('Keypoints of Image 4')
plt.show()

# Match features between img3 and img4
matches34 = matcher.match(descriptors3, descriptors4, None)
matches34.sort(key=lambda x: x.distance, reverse=False)
numGoodMatches34 = int(len(matches34) * GOOD_MATCH_PERCENT)
matches34 = matches34[:numGoodMatches34]
# Draw top matches
im34Matches = cv2.drawMatches(img3, keypoints3, img4, keypoints4, matches34, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("matches34.jpg", im34Matches)
plt.figure()
plt.imshow(im34Matches[:,:,::-1]), plt.title('Matches Between img3 & img4')
plt.show()


# Draw keypoints between img4 and img5
img4Keypoints = cv2.drawKeypoints(img4, keypoints4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5Keypoints = cv2.drawKeypoints(img5, keypoints5, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.subplot(121), plt.imshow(img4Keypoints[:,:,::-1]), plt.title('Keypoints of Image 4')
plt.subplot(122), plt.imshow(img5Keypoints[:,:,::-1]), plt.title('Keypoints of Image 5')
plt.show()

# Match features between img4 and img5
matches45 = matcher.match(descriptors4, descriptors5, None)
matches45.sort(key=lambda x: x.distance, reverse=False)
numGoodMatches45 = int(len(matches45) * GOOD_MATCH_PERCENT)
matches45 = matches45[:numGoodMatches45]
# Draw top matches
im45Matches = cv2.drawMatches(img4, keypoints4, img5, keypoints5, matches45, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("matches45.jpg", im45Matches)
plt.figure()
plt.imshow(im45Matches[:,:,::-1]), plt.title('Matches Between img4 & img5')
plt.show()

###########################################################################################################################################################################

# ~ CREATE THE STITCHED IMAGE ~


# Extract points from matches between img1 and img2
points1_12 = np.zeros((len(matches12), 2), dtype=np.float32)
points2_12 = np.zeros((len(matches12), 2), dtype=np.float32)
for i, match in enumerate(matches12):
    points1_12[i, :] = keypoints1[match.queryIdx].pt
    points2_12[i, :] = keypoints2[match.trainIdx].pt
# Find homography from img2 to img1
h12, mask = cv2.findHomography(points2_12, points1_12, cv2.RANSAC)


# Extract points from matches between img2 and img3
points3_23 = np.zeros((len(matches23), 2), dtype=np.float32)
points4_23 = np.zeros((len(matches23), 2), dtype=np.float32)
for i, match in enumerate(matches23):
    points3_23[i, :] = keypoints2[match.queryIdx].pt
    points4_23[i, :] = keypoints3[match.trainIdx].pt
# Find homography from img3 to img2
h23, mask = cv2.findHomography(points4_23, points3_23, cv2.RANSAC)


# Extract points from matches between img3 and img4
points5_34 = np.zeros((len(matches34), 2), dtype=np.float32)
points6_34 = np.zeros((len(matches34), 2), dtype=np.float32)
for i, match in enumerate(matches34):
    points5_34[i, :] = keypoints3[match.queryIdx].pt
    points6_34[i, :] = keypoints4[match.trainIdx].pt
# Find homography from img4 to img3
h34, mask = cv2.findHomography(points6_34, points5_34, cv2.RANSAC)


# Extract points from matches between img4 and img5
points7_45 = np.zeros((len(matches45), 2), dtype=np.float32)
points8_45 = np.zeros((len(matches45), 2), dtype=np.float32)
for i, match in enumerate(matches45):
    points7_45[i, :] = keypoints4[match.queryIdx].pt
    points8_45[i, :] = keypoints5[match.trainIdx].pt
# Find homography from img5 to img4
h45, mask = cv2.findHomography(points8_45, points7_45, cv2.RANSAC)


# Use homography
img1Height, img1Width, _ = img1.shape
img2Height, img2Width, _ = img2.shape
img3Height, img3Width, _ = img3.shape
img4Height, img4Width, _ = img4.shape
img5Height, img5Width, _ = img5.shape

# Calculate output size for the final stitched image
width_output = img1Width + img2Width + img3Width + img4Width + img5Width
height_output = max(img1Height, img2Height, img3Height, img4Height, img5Height)

# Warp images using their respective homographies
img2Aligned = cv2.warpPerspective(img2, h12, (width_output, height_output))
img3Aligned = cv2.warpPerspective(img3, np.dot(h12, h23), (width_output, height_output))
img4Aligned = cv2.warpPerspective(img4, np.dot(np.dot(h12, h23), h34), (width_output, height_output))
img5Aligned = cv2.warpPerspective(img5, np.dot(np.dot(np.dot(h12, h23), h34), h45), (width_output, height_output))


# Create a blank canvas for the final stitched image
stitchedImage = np.zeros((height_output, width_output, 3), dtype=np.uint8)
stitchedImage[0:img1Height, 0:img1Width] = img1

plt.figure()
plt.imshow(stitchedImage[:,:,::-1]), plt.title('The black canvas for the final stitched image')
plt.show()

# Combine the stitched image
stitchedImage12 = np.maximum(stitchedImage, img2Aligned)
stitchedImage123 = np.maximum(stitchedImage12, img3Aligned)
stitchedImage1234 = np.maximum(stitchedImage123, img4Aligned)
stitchedImage12345 = np.maximum(stitchedImage1234, img5Aligned)

# Display the final stitched image
plt.figure()
plt.imshow(stitchedImage12345[:, :, ::-1]), plt.title('All 5 images stitched')
plt.show()

# Crop the image using array slicing
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

# Display the final stitched image - cropped version (without black margins)
stitched_crop = crop_image(stitchedImage12345,23,36,4145,1303)
plt.figure()
plt.imshow(stitched_crop[:, :, ::-1]), plt.title('Final Panoramic Stitch')
plt.show()

###########################################################################################################################################################################

# ~ FINAL IMAGE PROCESSING ~

# Apply a Gaussian blur to smooth out seams and transitions
stitched_smooth = cv2.GaussianBlur(stitched_crop, (3, 3), 0)

# Convert the image to HSV color space for saturation adjustment
hsv = cv2.cvtColor(stitched_smooth, cv2.COLOR_BGR2HSV)
# Split the HSV image into its channels
h_channel, s_channel, v_channel = cv2.split(hsv)

# Increase the saturation channel
s_channel = cv2.multiply(s_channel, 2.0)
s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)

# Merge the channels back and convert to BGR
hsv_enhanced = cv2.merge((h_channel, s_channel, v_channel))
stitched_colored = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

# Display the final enhanced result
plt.figure()
plt.imshow(stitched_colored[:, :, ::-1])
plt.title("Final Panorama - enhanced saturation and blur")
plt.axis('off')
plt.show()