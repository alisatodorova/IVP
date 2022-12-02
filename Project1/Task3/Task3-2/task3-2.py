import cv2
import numpy as np

# Cartoonifies an image by creating a black outline around the shapes
# and fills in the colored areas with a uniform color
def cartoonifyImage(image):

    # Convert the original image to grayscale image
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth the grayscale image by adding a blurring effect
    imageBlurred = cv2.medianBlur(imageGray, 23)

    # Extract the edges in the image and highlight them
    imageEdges = cv2.adaptiveThreshold(imageBlurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)

    # Remove the noise from the original image
    imageNoNoise = cv2.bilateralFilter(image,11,255,255)

    # Mask (i.e. combine) the color image with edges
    imageCartoon = cv2.bitwise_and(imageNoNoise, imageNoNoise, mask = imageEdges)

    return imageCartoon


## Yellow.jpeg

# Read in the image
yellow = cv2.imread('yellow.jpeg')
# Cartoonify the image
yellowCartoon = cartoonifyImage(yellow)
# Display the cartoonified image
cv2.imshow('yellow as a cartoon', yellowCartoon)
# Save the image
cv2.imwrite("yellowCartoon.jpg", yellowCartoon)



## Flower.jpeg

# Read in the image
flower = cv2.imread('flower.jpeg')
# Cartoonify the image
flowerCartoon = cartoonifyImage(flower)
# Display the cartoonified image
cv2.imshow('flower as a cartoon', flowerCartoon)
# Save the image
cv2.imwrite("flowerCartoon.jpg", flowerCartoon)


cv2.waitKey(0)
cv2.destroyAllWindows()
