import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the figure size
fig1 = plt.figure(figsize=(10,7))
fig2 = plt.figure(figsize=(10,7))

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

# Create histogram for the image
plt.title('Fog.jpg histogram')
plt.xlabel('value (bins)')
plt.ylabel('pixels frequency')
plt.hist(fogRGB.ravel(), 256, [0, 256])
# Save the histogram as an image
fig1.savefig('fogRGBHist.jpg')
plt.show()

## Shadows.jpg

# Clear the plot
plt.clf()

# RGB
shadowsRGB = convertBGRtoRGB('shadows.jpg')
# Display the RGB image
cv2.imshow('shadows in RGB', shadowsRGB)
# Save the image
cv2.imwrite("shadowsRGB.jpg", shadowsRGB)

# Create histogram for the image
plt.title('Shadows.jpg histogram')
plt.xlabel('value (bins)')
plt.ylabel('pixels frequency')
plt.hist(shadowsRGB.ravel(), 256, [0, 256])
# Save the histogram as an image
fig2.savefig('shadowsRGBHist.jpg')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
