import cv2
import numpy as np

# Transforms an BGR image to RGB image
def convertBGRtoRGB(image):

    # Read the image
    imageBGR = cv2.imread(image)

    # Convert the BGR image to RGB (since OpenCV reads the images as BGR)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    return imageRGB

# Converts the RGB image to an intensity image “I” in HSI space
def convertRGBtoI(imageRGB):

    # Divide the image by pixel value 255 to represent the image in [0 1] range
    dividedImage = np.divide(imageRGB, 255)

    # Separate the three channels from BGR image
    RImage = dividedImage[:, :, 2] #red
    GImage = dividedImage[:, :, 1] #green
    BImage = dividedImage[:, :, 0] #blue

    # Compute Intensity
    intensityImage = np.divide(RImage + GImage + BImage, 3)

    return intensityImage


#Converts the RGB image to an intensity image “V” in HSV space
def convertRGBtoV(imageRGB):

    # Divide the image by pixel value 255
    dividedImage = np.divide(imageRGB, 255)

    # Compute Value by calculating the maximum of the RGB image
    valueImage = np.max(dividedImage, axis=2)

    return valueImage


## Birds.jpg

#RGB
birdsRGB = convertBGRtoRGB('birds.jpg')
# Display the RGB image
cv2.imshow('birds in RGB', birdsRGB)
# Save the image
cv2.imwrite("birdsRGB.jpg", birdsRGB)

#Intensity image “I” in HSI space
birdsIntensity = convertRGBtoI(birdsRGB)
# Display the intensity image “I” in HSI space
cv2.imshow('intensity birds I in HSI space', birdsIntensity)
# Save the image
cv2.imwrite("birdsIntensity.jpg", birdsIntensity)

#Intensity image “V” in HSV space
birdsValue = convertRGBtoV(birdsRGB)
# Display the intensity image “V” in HSV space
cv2.imshow('intensity birds V in HSV space', birdsValue)
# Save the image
cv2.imwrite("birdsValue.jpg", birdsValue)



## Stone.jpg

#RGB
stoneRGB = convertBGRtoRGB('stone.jpg')
# Display the RGB image
cv2.imshow('stone in RGB', stoneRGB)
# Save the image
cv2.imwrite("stoneRGB.jpg", stoneRGB)

#Intensity image “I” in HSI space
stoneIntensity = convertRGBtoI(stoneRGB)
# Display the intensity image “I” in HSI space
cv2.imshow('intensity stone I in HSI space', stoneIntensity)
# Save the image
cv2.imwrite("stoneIntensity.jpg", stoneIntensity)

#Intensity image “V” in HSV space
stoneValue = convertRGBtoV(stoneRGB)
# Display the intensity image “V” in HSV space
cv2.imshow('intensity stone V in HSV space', stoneValue)
# Save the image
cv2.imwrite("stoneValue.jpg", stoneValue)

cv2.waitKey(0)
cv2.destroyAllWindows()
