import cv2
import numpy as np
import matplotlib.pyplot as plt

# Adds sinusoidal periodic noise to an image
def addSinNoise(image):
    # Read the image as pixel values
    imagePixels = np.mean(cv2.imread(image), axis=2) / 255

    # Add sinusoidal periodic noise
    for k in range(imagePixels.shape[1]):
        imagePixels[:, k] += np.cos(0.3 * np.pi * k) # 0.3 is the width of the horizontal lines

    return imagePixels


## Crayons.jpg

# Original Grayscaled image
crayons = cv2.imread('crayons.jpg')
crayonsGray = cv2.cvtColor(crayons, cv2.COLOR_BGR2GRAY)
# Save the image
cv2.imwrite("crayonsGray.jpg", crayonsGray)

# The grayscaled image with sinusoidal periodic noise
crayonsSinNoise = addSinNoise('crayons.jpg')
#Show the image
plt.imshow(crayonsSinNoise, cmap='gray')
plt.axis('off')
# Save the image
plt.imsave("crayonsSinNoise.jpg", crayonsSinNoise, cmap='gray')
plt.show()

