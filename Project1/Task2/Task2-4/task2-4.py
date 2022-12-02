import cv2
import numpy as np
import matplotlib.pyplot as plt

# Transforms an BGR image to RGB image
def convertBGRtoRGB(image):

    # Read the image
    imageBGR = cv2.imread(image)

    # Convert the BGR image to RGB (since OpenCV reads the images as BGR)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    return imageRGB

# Gamma correction (aka Power Law Transform):
# Converts the low contrast image to a higher contrast and vice versa depending on the gamma value
# Gamma < 1 gets brighter
# Gamma > 1 gets darker
# Gamma = 1 will have no effect on the input image
def gammaCorr(imageRGB, gamma):
    gammaCorrImage = ((imageRGB / 255) ** gamma)

    return gammaCorrImage


## Fog.jpg

# RGB
fogRGB = convertBGRtoRGB('fog.jpg')
# Display the RGB image
cv2.imshow('fog in RGB', fogRGB)
# Save the image
cv2.imwrite("fogRGB.jpg", fogRGB)

# Convert the low contrast image to a higher contrast one
# Hence, Gamma = 3.5
gammaCorrectFog = gammaCorr(fogRGB, 3.5)

# Display the Power law pointwise transform Fog image
cv2.imshow('power law transform fog', gammaCorrectFog)
# Save the image
cv2.imwrite("fogPowerLaw.jpg", gammaCorrectFog)


## Shadows.jpg

# RGB
shadowsRGB = convertBGRtoRGB('shadows.jpg')
# Display the RGB image
cv2.imshow('shadows in RGB', shadowsRGB)
# Save the image
cv2.imwrite("shadowsRGB.jpg", shadowsRGB)

# Convert the high contrast image to a lower contrast one
# Hence, Gamma = 0.5
gammaCorrectShadows = gammaCorr(shadowsRGB, 0.5)

# Display the Power law pointwise transform Fog image
cv2.imshow('power law transform shadows', gammaCorrectShadows)
# Save the image
cv2.imwrite("shadowsPowerLaw.jpg", gammaCorrectShadows)


cv2.waitKey(0)
cv2.destroyAllWindows()
