import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Read in the images in grayscale
image = cv2.imread("jar.jpg", 0)
image = cv2.resize(image, (600, 400)) # Resize the image to reduce compiling time

# To improve the contrast, apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
# It equalizes the image by reducing over-amplification of the contrast.
# clipLimit parameter sets the threshold for contrast limiting
# tileGridSize parameter sets the number of tiles in the row and column.
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
improvedImg = clahe.apply(image)

# Display the improved image
cv2.imshow("Improved jar.jpg", improvedImg)
# Save the image
cv2.imwrite("improvedJar.jpg", improvedImg)

height, width = improvedImg.shape

# Loops through the frequencies and put them in the corresponding lists
frequencies = []
kernelRange = []
for i in range(1, int(width/2), 5):
    imageOpen = opening(improvedImg, (i, i), 1, 1) # Perform Opening
    frequencies.append(np.sum(imageOpen))
    kernelRange.append(i)

# Loops through the different kernel ranges and put them in the corresponding lists
differentRanges = []
for i in range(1, int(width/2), 5):
    differentRanges.append(i)

# looping through the length of the frequencies and appending the differences in frequencies list
diffFrequencies = []
for i in range(1, len(frequencies)):
    diffFrequencies.append(np.abs(frequencies[i - 1] - frequencies[i]))

plt.figure()
plt.plot(differentRanges, frequencies)
plt.title("Frequencies")
plt.savefig("frequencies.png")

plt.figure()
plt.plot(diffFrequencies)
plt.title("Difference in frequencies")
plt.savefig("diffFreq.png")

plt.show()
