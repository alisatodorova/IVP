import cv2
import numpy as np
import matplotlib.pyplot as plt

# Transforms an BGR image to RGB image
# def convertBGRtoRGB(image):
#
#     # Read the image
#     imageBGR = cv2.imread(image)
#
#     # Convert the BGR image to RGB (since OpenCV reads the images as BGR)
#     imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
#
#     return imageRGB


## Pink.jpg with dimenstions 1488x1248

# RGB
# pinkRGB = convertBGRtoRGB('pink.jpg')
# # Display the RGB image
# cv2.imshow('pink in RGB', pinkRGB)
# # Save the image
# cv2.imwrite("pinkRGB.jpg", pinkRGB)

# Read the image
pink = cv2.imread('pink.jpg')

# Translate the image horizontally and vertically by 200 pixels
transpose = np.float32([[1, 0, 200], [0, 1, 200]])
pinkTranslated = cv2.warpAffine(pink, transpose, (pink.shape[0], pink.shape[1]))

# Display the Translated Pink image
cv2.imshow('pink translated', pinkTranslated)
# Save the image
cv2.imwrite("pinkTranslated.jpg", pinkTranslated)


# Convert the translated image to grayscale image
pinkTranslatedGray = cv2.cvtColor(pinkTranslated, cv2.COLOR_BGR2GRAY)

# Calculate the 2D FFT of the translated image Pink
pink2DFFTTranslated = np.fft.fft2(pinkTranslatedGray)
print(pink2DFFTTranslated)
# Bring it to the center
pink2DFFTshiftedTranslated = np.fft.fftshift(pink2DFFTTranslated)

# The translated image's FT magnitude
magnitudeTranslated = 20 * np.log(np.abs(pink2DFFTshiftedTranslated))
# Display the magnitude of the Translated Pink image
cv2.imshow('magnitude of translated image', magnitudeTranslated)
# Save the image
cv2.imwrite("magnitudeTranslated.jpg", magnitudeTranslated)


# Convert the original image to grayscale image
pinkGray = cv2.cvtColor(pink, cv2.COLOR_BGR2GRAY)

# The original image's 2D FT magnitude
pink2DFFTOriginal = np.fft.fft2(pinkGray)
print(pink2DFFTOriginal)
# Bring it to the center
pink2DFFTshiftedOriginal = np.fft.fftshift(pink2DFFTOriginal)

# The original image's 2D FT magnitude
magnitudeOriginal = 20*np.log(np.abs(pink2DFFTshiftedOriginal))
# Display the magnitude of the original Pink image
cv2.imshow('magnitude of original image', magnitudeOriginal)
# Save the image
cv2.imwrite("magnitudeOriginal.jpg", magnitudeOriginal)

cv2.waitKey(0)
cv2.destroyAllWindows()
