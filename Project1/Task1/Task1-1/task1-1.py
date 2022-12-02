import cv2

# Transforms an RGB image to HSV image
def convertRGBtoHSV(image):

    # Read the image
    imageBGR = cv2.imread(image)

    # Convert the BGR image to RGB (since OpenCV reads the images as BGR)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to HSV
    imageHSV = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)

    return imageRGB, imageHSV



## Birds.jpg
birdsRGB, birdsHSV = convertRGBtoHSV('birds.jpg')
# Display the images
cv2.imshow('birds in RGB', birdsRGB)
cv2.imshow('birds in HSV', birdsHSV)
# Save the images
cv2.imwrite("birdsRGB.jpg", birdsRGB)
cv2.imwrite("birdsHSV.jpg", birdsHSV)


## Stone.jpg
stoneRGB, stoneHSV = convertRGBtoHSV('stone.jpg')
# Display the images
cv2.imshow('stone in RGB', stoneRGB)
cv2.imshow('stone in HSV', stoneHSV)
# Save the images
cv2.imwrite("stoneRGB.jpg", stoneRGB)
cv2.imwrite("stoneHSV.jpg", stoneHSV)


cv2.waitKey(0)
cv2.destroyAllWindows()
