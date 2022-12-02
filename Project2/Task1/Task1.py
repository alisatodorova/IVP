import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.util import random_noise

# Read the Bird image
bird = cv2.imread('bird.jpg')
# Convert the image to double
birdDouble = bird.astype(np.double)

#### Degrading the image and adding noise ####

# Values of (u, v) have to be in the interval [−1, 1]
# The length of these values needs to be equal to the dimensions of the image n1 and n2.
n2 = birdDouble.shape[0]
n1 = birdDouble.shape[1]
[k1, k2] = np.mgrid[0:n2, 0:n1]

# The spacing from [−1, 1] produces n1 points for u and n2 points for v
[u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
u = 2 * u/n2
v = 2 * v/n1

# Original image FT is F(u,v)
birdFT = np.fft.fft2(birdDouble)

# Diagonal motion blurring degradation function H(u,v) = sinc(α·u + β·v) · exp(−jπ(α·u + β·v))
# I set alpha = 0.1 and beta = 0.1, so that the blur is not too noisy, but it’s still visible.
alpha = 0.1
beta = 0.1
H = np.sinc((u * alpha + v * beta)) * np.exp(-1j * np.pi * (u * alpha + v * beta))

# Separate the three channels from BGR image
RImage = birdFT[:, :, 2]  # red
GImage = birdFT[:, :, 1]  # green
BImage = birdFT[:, :, 0]  # blue

# Apply the diagonal motion blur filter to the Bird image
# Noisy image FT is G(u, v) = F(u,v) * H(u,v)
G = birdFT
G[:, :, 2] = np.multiply(RImage, H)
G[:, :, 1] = np.multiply(GImage, H)
G[:, :, 0] = np.multiply(BImage, H)

# The diagonal motion blurred image
result = np.fft.ifft2(G)
birdDiagonalMotionBlur = np.abs(result)

# Display the image
cv2.imshow("Bird with diagonal motion blur filter", birdDiagonalMotionBlur)
# Save the image
cv2.imwrite("birdDiagonalMotionBlur.jpg", birdDiagonalMotionBlur)

# Applies additive Gaussian noise to the motion blurred image
# mean=0.01 and var=0.2 produces noise that is visible but does not completely degrade the image.
birdsMotionNoise = random_noise(np.abs(birdDiagonalMotionBlur).astype(np.uint8), 'gaussian', mean=0.01, var=0.2)
birdsMotionNoise = birdsMotionNoise.astype(np.double)

# Display the image
cv2.imshow("Bird with diagonal motion blur filter and additive Gaussian noise", birdsMotionNoise)
# Save the image
cv2.imwrite("birdBlurAndNoise.jpg", birdsMotionNoise)

#### Removing noise ####

# Apply direct inverse filtering to the image after it has undergone only motion blur.
R1 = birdFT
# F'(u, v) = G(u, v)/H(u, v)
R1[:, :, 2] = np.divide(G[:, :, 2], H)
R1[:, :, 1] = np.divide(G[:, :, 1], H)
R1[:, :, 0] = np.divide(G[:, :, 0], H)
rx1 = np.abs(np.fft.ifft2(R1))
# Display the image
cv2.imshow("Direct inverse filtering of Bird image with only motion blur filter", rx1)
# Save the image
cv2.imwrite("inverseFiltMotionBlur.jpg", rx1)

# Apply direct inverse filtering to the image after it has undergone motion blur and additive noise.
R2 = G
Fn = np.fft.fft2(birdsMotionNoise) #noisy with blur image FT
# F''(u, v) = Fn(u, v)/H(u, v)
R2[:, :, 2] = np.divide(Fn[:, :, 2], H)
R2[:, :, 1] = np.divide(Fn[:, :, 1], H)
R2[:, :, 0] = np.divide(Fn[:, :, 0], H)
rx2 = np.abs(np.fft.ifft2(R2))

# Display the image
cv2.imshow("Direct inverse filtering of Bird image with motion blur filter and additive noise", rx2)
# Save the image
cv2.imwrite("inverseFiltMotionAndNoise.jpg", rx2)


# 2D Wiener filter, i.e. MMSE filter H_w(u, v), when there is only additive noise (no motion blur)
# See report for MMSE filter formula
# Applies additive Gaussian noise to the original image
birdsAddNoise = random_noise(np.abs(bird).astype(np.uint8), 'gaussian', mean=0.01, var=0.2)
birdsAddNoise = birdsAddNoise.astype(np.double)
# Display the image
cv2.imshow("additive Gaussian noise to the original Bird image", birdsAddNoise)
# Save the image
cv2.imwrite("addGaussianNoise.jpg", birdsAddNoise)

snn = abs(np.fft.fft2(birdsAddNoise)) ** 2 # Noise power spectrum, S_n(u,v)
sxx = abs(np.fft.fft2(birdDouble)) ** 2  # Power spectrum of the original image, S_f(u,v)

# We have that H(u,v) = 1
dh1 = 1 ** 2 + snn / sxx
Hw1 = 1 / dh1

R3 = G
R3[:, :, 2] = np.multiply(Hw1[:, :, 2], G[:, :, 2])
R3[:, :, 1] = np.multiply(Hw1[:, :, 1], G[:, :, 1])
R3[:, :, 0] = np.multiply(Hw1[:, :, 0], G[:, :, 0])
rx3 = np.abs(np.fft.ifft2(R3))
# Display the image
cv2.imshow("MMSE filtering of Bird image with only additive noise", rx3)
# Save the image
cv2.imwrite("mmseFiltNoise.jpg", rx3)


# 2D Wiener filter, i.e. MMSE filter H_w(u,v), when the image undergoes motion blur and additive noise
# See report for MMSE filter formula
# Approximate the noise
approxNoise = birdDouble - birdsMotionNoise

# Noise power spectrum, S_n(u,v)
snn2_red = abs(np.fft.fft2(approxNoise[:, :, 2])) ** 2
snn2_green = abs(np.fft.fft2(approxNoise[:, :, 1])) ** 2
snn2_blue = abs(np.fft.fft2(approxNoise[:, :, 0])) ** 2

# Power spectrum of the original image, S_f(u,v)
sxx2_red = abs(np.fft.fft2(birdDouble[:, :, 2])) ** 2
sxx2_green = abs(np.fft.fft2(birdDouble[:, :, 1])) ** 2
sxx2_blue = abs(np.fft.fft2(birdDouble[:, :, 0])) ** 2

dh2_red = np.abs(H) ** 2 + snn2_red / sxx2_red
dh2_green = np.abs(H) ** 2 + snn2_green / sxx2_green
dh2_blue = np.abs(H) ** 2 + snn2_blue / sxx2_blue
Hw2_red = np.conj(H) / dh2_red
Hw2_green = np.conj(H) / dh2_green
Hw2_blue = np.conj(H) / dh2_blue

R4 = G
R4[:, :, 2] = np.multiply(Hw2_red, G[:, :, 2])
R4[:, :, 1] = np.multiply(Hw2_green, G[:, :, 1])
R4[:, :, 0] = np.multiply(Hw2_blue, G[:, :, 0])
rx4 = np.abs(np.fft.ifft2(R4))
# Display the image
cv2.imshow("MMSE filtering of Bird image with motion blur filter and additive noise", rx4)
# Save the image
cv2.imwrite("mmseFiltMotionAndNoise.jpg", rx4)


cv2.waitKey(0)
cv2.destroyAllWindows()
