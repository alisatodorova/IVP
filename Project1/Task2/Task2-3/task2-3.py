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

# Create histogram for Negative Fog.jpg
plt.title('Negative Fog.jpg histogram')
plt.xlabel('value (bins)')
plt.ylabel('pixels frequency')
negativeFogHist = plt.hist(negativeFog.ravel(),256,[0,256])
# Save the histogram
plt.savefig("negativeFogHist.jpg")


## Shadows.jpg

# Clear the plot
plt.clf()

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


# Create histogram for Negative Shadows.jpg
plt.title('Negative Shadows.jpg histogram')
plt.xlabel('value (bins)')
plt.ylabel('pixels frequency')
negativeShadowsHist = plt.hist(negativeShadows.ravel(),256,[0,256])
# Save the histogram
plt.savefig("negativeShadowsHist.jpg")

plt.show()
