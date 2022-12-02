import cv2
import numpy as np

# # Transforms an BGR image to RGB image
# def convertBGRtoRGB(image):
#
#     # Read the image
#     imageBGR = cv2.imread(image)
#
#     # Convert the BGR image to RGB (since OpenCV reads the images as BGR)
#     imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
#
#     return imageRGB

# Converts an image into polar coordinates
def convertToPolarCoor(image):

    # Convert the image to float
    imageFloat = image.astype(np.float32)

    # Calculate the maximum radius
    maxRadius = np.sqrt((imageFloat.shape[0] ** 2.0) + (imageFloat.shape[1] ** 2.0)) / 2

    # Convert into linear polar coordinates
    # cv2.linearPolar(InputArray src, OutputArray dst, Point2f center, double maxRadius, int flags)
    imagePolarCoor = cv2.linearPolar(imageFloat, (imageFloat.shape[0] / 2, imageFloat.shape[1] / 2), maxRadius, cv2.WARP_FILL_OUTLIERS)

    # Convert the float to a range from 0 to 255
    imagePolarCoor = imagePolarCoor.astype(np.uint8)

    return imagePolarCoor

## Yellow.jpeg
yellow = cv2.imread('yellow.jpeg')

# Convert the Yellow image into polar coordinates
yellowPolar = convertToPolarCoor(yellow)
# Display the Yellow image into polar coordinates
cv2.imshow('yellow into polar coordinates', yellowPolar)
# Save the image
cv2.imwrite("yellowPolar.jpg", yellowPolar)

## Flower.jpeg
flower = cv2.imread('flower.jpeg')
# Convert the Flower image into polar coordinates
flowerPolar = convertToPolarCoor(flower)
# Display the Flower image into polar coordinates
cv2.imshow('flower into polar coordinates', flowerPolar)
# Save the image
cv2.imwrite("flowerPolar.jpg", flowerPolar)


## In RGB

## Yellow.jpeg in RGB
# yellowRGB = convertBGRtoRGB('yellow.jpeg')
# # Display the RGB image
# cv2.imshow('yellow in RGB', yellowRGB)
# # Save the image
# cv2.imwrite("yellowRGB.jpg", yellowRGB)

# # Convert the RGB Yellow image into polar coordinates
# yellowPolar = convertToPolarCoor(yellowRGB)
# # Display the RGB Yellow image into polar coordinates
# cv2.imshow('yellow into polar coordinates', yellowPolar)
# # Save the image
# cv2.imwrite("yellowPolar.jpg", yellowPolar)

## Flower.jpeg in RGB
# flowerRGB = convertBGRtoRGB('flower.jpeg')
# # Display the RGB image
# cv2.imshow('flower in RGB', flowerRGB)
# # Save the image
# cv2.imwrite("flowerRGB.jpg", flowerRGB)
#
# # Convert the RGB Flower image into polar coordinates
# flowerPolar = convertToPolarCoor(flowerRGB)
# # Display the RGB Flower image into polar coordinates
# cv2.imshow('flower into polar coordinates', flowerPolar)
# # Save the image
# cv2.imwrite("flowerPolar.jpg", flowerPolar)

cv2.waitKey(0)
cv2.destroyAllWindows()
