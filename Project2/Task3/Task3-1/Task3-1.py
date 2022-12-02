import cv2
import numpy as np

# The morphological operation Opening is erosion followed by dilation
def opening(img, kernel_size, erosion_iteration, dilation_iteration):
    # Kernel (aka structuring element) which decides the nature of operation by sliding through the image.
    # Create an elliptical/circular shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Erosion removes small white noises:
    # A pixel in the original image is 1 only if all the pixels under the kernel is 1.
    # Otherwise, it is 0, i.e. eroded.
    erosion = cv2.erode(img, kernel, erosion_iteration)
    # Dilation increases the white region in the image:
    # A pixel of the original image is 1 (white) if at least one pixel under the kernel is 1.
    # Otherwise, it's 0 (black).
    dilation = cv2.dilate(erosion, kernel, dilation_iteration)
    return dilation


# The morphological operation Closing is dilation followed by erosion
def closing(img, kernel_size, erosion_iteration, dilation_iteration):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Dilation increases the white region in the image:
    # A pixel of the original image is 1 (white) if at least one pixel under the kernel is 1.
    # Otherwise, it's 0 (black).
    dilation = cv2.dilate(img, kernel, dilation_iteration)
    # Erosion removes small white noises:
    # A pixel in the original image is 1 only if all the pixels under the kernel is 1.
    # Otherwise, it is 0, i.e. eroded.
    erosion = cv2.erode(dilation, kernel, erosion_iteration)
    return erosion

## Pre-processing: Convert the images into black and white,
## so that the round objects appear white and the rest is black.

# Read in the images in grayscale
oranges = cv2.imread("oranges.jpg", 0) # 0 reads in the images in grayscale
tree = cv2.imread("orangetree.jpg", 0)

# Binary threshold to separate and threshold the various colors,
# where 127 is the middle pixel value for a grayscaled image and 255 is white.
# Thus, in the new image, bw_img, values are 0 (i.e. black) if threshold > pixel value of grayscaled image,
# otherwise, values are 255 (i.e. white).
threshold_oranges, bw_oranges = cv2.threshold(oranges, 127, 255, cv2.THRESH_BINARY)
threshold_tree, bw_tree = cv2.threshold(tree, 127, 255, cv2.THRESH_BINARY)

# Display the Black and white oranges
cv2.imshow("Black and white oranges", bw_oranges)
# Save the image
cv2.imwrite("blackWhiteOranges.jpg", bw_oranges)

# Display the Black and white orange tree
cv2.imshow("Black and white orange tree", bw_tree)
# Save the image
cv2.imwrite("blackWhiteTree.jpg", bw_tree)


## Count the number of oranges in both images

orangesOpen = opening(bw_oranges, (136, 136), 5, 4)
sumOranges = np.count_nonzero(orangesOpen) # Counts the number of non-zero values in the array
countOranges = round(sumOranges / (136 * 136))
print("Number of oranges in oranges.jpg: ", countOranges)

# Since the orange tree image has still a lot of noise even after the pre-processing,
# we need to perform both opening and closing.
treeOpen = opening(bw_tree, (136, 136), 21, 10)
treeClose = closing(treeOpen, (4, 4), 15, 10)
sumTree = np.count_nonzero(treeClose) # Counts the number of non-zero values in the array
countTree = round(sumTree / (136 * 136))
print("Number of oranges in the orangetree.jpg: ", countTree)

cv2.waitKey(0)
cv2.destroyAllWindows()
