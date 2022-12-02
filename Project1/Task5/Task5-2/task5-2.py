import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
noisyImage = cv2.imread('crayonsSinNoise.jpg')
# Convert the translated image to grayscale image
noisyImageGray = cv2.cvtColor(noisyImage, cv2.COLOR_BGR2GRAY)
# Convert the image to float
imageGray = noisyImageGray.astype(np.float32)

# Calculate the 2D FFT of the image
noisyImageGray2DFFT = np.fft.fft2(imageGray)
print(noisyImageGray2DFFT)
# Bring it to the center
noisyImageGray2DFFTshifted = np.fft.fftshift(noisyImageGray2DFFT)

# The power spectrum P(u,v) is the magnitude squared
powerSpectrum = 20 * np.log(np.abs(noisyImageGray2DFFTshifted) ** 2)

# Display the noisy image’s power spectrum in 2D
# Convert the float to a range from 0 to 255
power2D = powerSpectrum.astype(np.uint8)
cv2.imshow('power2D', power2D)
# Save the image
cv2.imwrite("power2D.jpg", power2D)


# Display the noisy image’s power spectrum in 1D
# A 1D representation is a slice of the (2D) power spectrum,
# i.e. display the middle row of P(u,v)
power1D = power2D[int(noisyImage.shape[0]/2),:]
plt.plot(power1D)
#Save the image
plt.savefig("power1D.jpg")
plt.show()

# Clear the plot for the next plot
plt.clf()

# Display the noisy image’s power spectrum in 3D
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
x,y = np.meshgrid(np.arange(power2D.shape[0]),np.arange(power2D.shape[1]))
ax.plot_surface(x,y,power2D.T,cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
#Save the image
plt.savefig("power3D.jpg")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
