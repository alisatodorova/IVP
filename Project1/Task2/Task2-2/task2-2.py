import cv2
import numpy as np

# Transforms an BGR image to RGB image
def convertBGRtoRGB(image):

    # Read the image
    imageBGR = cv2.imread(image)

    # Convert the BGR image to RGB (since OpenCV reads the images as BGR)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    return imageRGB



## Fog.jpg
# RGB
fogRGB = convertBGRtoRGB('fog.jpg')
# Display the RGB image
cv2.imshow('fog in RGB', fogRGB)
# Save the image
cv2.imwrite("fogRGB.jpg", fogRGB)

# Negative pointwise transform
negativeFog = 255 - fogRGB

# Display the Negative Fog image
cv2.imshow('negative fog', negativeFog)
# Save the image
cv2.imwrite("fogNegative.jpg", negativeFog)


## Shadows.jpg

# RGB
shadowsRGB = convertBGRtoRGB('shadows.jpg')
# Display the RGB image
cv2.imshow('shadows in RGB', shadowsRGB)
# Save the image
cv2.imwrite("shadowsRGB.jpg", shadowsRGB)

# Negative pointwise transform
negativeShadows = 255 - shadowsRGB

# Display the Negative Shadows image
cv2.imshow('negative shadows', negativeShadows)
# Save the image
cv2.imwrite("shadowsNegative.jpg", negativeShadows)

cv2.waitKey(0)
cv2.destroyAllWindows()
