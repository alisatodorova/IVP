import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Butterworth Low Pass Filters
def butterworthLPF(d0, n1, n2, n):
    k1, k2 = np.meshgrid(np.arange(-round(n2 / 2) + 1, math.floor(n2 / 2) + 1),
                         np.arange(-round(n1 / 2) + 1, math.floor(n1 / 2) + 1))
    d = np.sqrt(k1 ** 2 + k2 ** 2)
    h = 1 / (1 + (d / d0) ** (2 * n))
    return h


# Applies the filter on the image
def applyFilter(image, filterMask):

    # Calculate the 2D FFT of the image and bring it to the center
    f = np.fft.fftshift(np.fft.fft2(image))
    # Add the filter
    f1 = f * filterMask
    # Calculate the 2D FFT of the filtered image and bring it to the center
    x1 = np.fft.ifft2(np.fft.ifftshift(f1))
    # Convert the float to a range from 0 to 255
    filterAdded = x1.astype(np.uint8)

    return filterAdded


# Read the image
noisyImage = cv2.imread('crayonsSinNoise.jpg')
# Convert the translated image to grayscale image
imageGray = cv2.cvtColor(noisyImage, cv2.COLOR_BGR2GRAY)
# Display the image
cv2.imshow('noisy image', imageGray)

## Remove the periodic noise in the frequency domain

# Cutoff frequency
d0=round(imageGray.shape[0] / 50) # 50 because the noise's horizontal lines are close to each other
# Butterworth Low Pass Filters
bw_maskLPF = butterworthLPF(d0, imageGray.shape[0], imageGray.shape[1], 1)
# Apply the filter
noNoise = applyFilter(imageGray, bw_maskLPF)

# Then show the de-noised image (in space)
cv2.imshow('de-noised image', noNoise)
# Save the image
cv2.imwrite("denoised.jpg", noNoise)

## Display the 2D power spectrum of the de-noised image

# Convert the image to float
noNoiseFloat = noNoise.astype(np.float32)

# Calculate the 2D FFT of the image
noNoise2DFFT = np.fft.fft2(noNoiseFloat)
# Bring it to the center
noNoise2DFFTshifted = np.fft.fftshift(noNoise2DFFT)

# The power spectrum P(u,v) is the magnitude squared
powerSpectrum = 20 * np.log(np.abs(noNoise2DFFTshifted) ** 2)

# Convert the float to a range from 0 to 255
power2D = powerSpectrum.astype(np.uint8)
# Show the 2D power spectrum
cv2.imshow('power2D', power2D)
# Save the image
cv2.imwrite("power2D.jpg", power2D)


# Display the de-noised image’s power spectrum in 1D
# A 1D representation is a slice of the (2D) power spectrum,
# i.e. display the middle row of P(u,v)
power1D = power2D[int(noNoise.shape[0]/2),:]
plt.plot(power1D)
#Save the image
plt.savefig("power1D.jpg")
plt.show()

# Clear the plot for the next plot
plt.clf()

# Display the de-noised image’s power spectrum in 3D
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
x,y = np.meshgrid(np.arange(power2D.shape[0]),np.arange(power2D.shape[1]))
ax.plot_surface(x,y,power2D.T,cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
#Save the image
plt.savefig("power3D.jpg")
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
