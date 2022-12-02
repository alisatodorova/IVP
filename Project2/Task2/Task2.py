import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from numpy import pi
from numpy import r_

img = cv2.imread('cameraman.tif', 0)

# Define 2D DCT (discrete cosine transform)
def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.T, norm='ortho').T, norm='ortho')

# Define IDCT (inverse discrete cosine transform)
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct(a.T, norm='ortho').T, norm='ortho')

### 2.1 Watermark Insertion ###

# Perform a blockwise DCT
imsize = img.shape
dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:i+8,j:j+8] = dct2(img[i:i+8,j:j+8])

# Extract 8x8 block and look at its DCT coefficients
pos = 128

# Display entire DCT
plt.figure()
plt.imshow(dct, vmax = np.max(dct) * 0.01, vmin = 0)
plt.title("8x8 DCTs of the image")
plt.savefig("8x8DCTofImage.png")

# DCT coefficients
K = 14  #14 doesn't degrade the image quality too much
dct_thresh = dct * (abs(dct) > K) #values larger than magnitude 14
plt.figure()
plt.imshow(dct_thresh, vmax = np.max(dct) * 0.01, vmin = 0)
plt.title("DCT coefficients in 8x8 blocks")
plt.savefig("8x8DCTcoeff.png")

# Compare original image with DCT compressed image
img_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        img_dct[i:(i + 8), j:(j + 8)] = idct2(dct_thresh[i:(i + 8), j:(j + 8)])

plt.figure()
plt.axis('off')
plt.imshow(np.hstack((img, img_dct)), cmap='gray')
plt.title("Comparison between original and DCT compressed images")
plt.savefig("originalVSDCTcompressed.png")

# Create a watermark by generating a K-element pseudo-random sequence of numbers, ω1, ω2, ..., ωK
# that follows a Gaussian distribution with a mean μ = 0 and variance σ2 that you will choose.
# np.random.normal(The mean of the distribution; The standard deviation; The size or shape of your array)
w = np.random.normal(0, np.sqrt(0.002), K)

# 2D DCT of the original image
dct_img = dct2(img)

# Convert a flat array of flat indices into a tuple of coordinate arrays
index = np.dstack(np.unravel_index(np.argsort(dct_img.ravel()), imsize))[0]

# We have K+1 non-zero DCT coefficients and for c′i = ci · (1 + αωi) we have that 1 ≤ i ≤ K
index = index[-(K + 1):-1] # Negative index means we count from last to first position of the row

# K DCT coefficients ci
ci = dct_img[index[:,0], index[:,1]] # [:,0] = all rows, first column; [:,1] = all rows, second column

# Embed the watermark into K largest non-DC DCT coefficients in each block: c′i = ci · (1 + αωi), 1 ≤ i ≤ K
ci_new = ci * (1 + 0.1 * w) # α=0.1 controls the strength of the watermark

# Create the DCT of your watermarked image by replacing the K DCT coefficients ci that you kept with c′i
dct_img[index[:,0], index[:,1]] = ci_new

# # The watermarked image
watermarked_img = idct2(dct_img) # inverse DCT of the DCT with the new coefficients c′i
plt.figure()
plt.axis('off')
plt.imshow(watermarked_img, cmap='gray')
# plt.title("Watermarked image")
plt.savefig("watermarkedImage.png", transparent=True)

# Comparison between original and watermarked images
plt.figure()
plt.axis('off')
plt.imshow(np.hstack((img, watermarked_img)), cmap='gray')
plt.title("Comparison between original and watermarked images")
plt.savefig("originalVSwatermarked.png")

# Difference image (i.e. between the original and watermarked images)
difference_img = np.abs(watermarked_img - img)
plt.figure()
plt.axis('off')
plt.imshow(difference_img, cmap='gray')
plt.title("Difference image")
plt.savefig("differenceImage.png")

plt.figure()
difference_hist = plt.hist(watermarked_img)
plt.xlabel("Value")
plt.ylabel("Pixels Frequency")
plt.title("Histogram of difference image")
plt.savefig("watermarkedImageHistogram.png")


### 2.2 Watermark Detection ###

## CASE 1: Watermarked image is the mystery image
case1 = cv2.imread('watermarkedImage.png', 0)

# Compute the 2D DCT of the M×N mystery image with coefficients cˆi, i=1,...,MN.
dct_case1 = dct2(case1)

# Keep its K largest non-DC DCT coefficients, now denoted as cˆ1,cˆ2,...,cˆK.
# Use the same K<MN, found in the watermark insertion part above.
c_hat1 = dct_case1[index[:,0], index[:,1]]

# Estimate an approximation of the watermark in your mystery image: ωˆi = (cˆi − ci)/αci , 1 ≤ i ≤ K
w1 = (c_hat1 - ci) / 0.1 * ci # α=0.1 controls the strength of the watermark

# Mean ω ̄ of the watermark sequence ω1,ω2,...,ωK
mean_watermark1 = np.mean(w)
# Mean of the approximated watermark sequence ωˆ1, ωˆ2, . . . , ωˆK
mean_approxWatermark1 = np.mean(w1)

# Measure the similarity of ωˆi with ωi using the correlation coefficient gamma
# (see assignment for formula), where 1 ≤ i ≤ K, so in np.sum we have initial=1 and where=K
gamma1 = np.sum([(w1 - mean_approxWatermark1) * (w - mean_watermark1)], initial=1, where=K) / np.sqrt(np.sum([(w1 - mean_approxWatermark1)**2], initial=1, where=K) * np.sum([(w - mean_watermark1)**2],initial=1, where=K))
print(gamma1)


## CASE 2: Original image (which doesn't have a watermark) is the mystery image
case2 = cv2.imread('cameraman.tif', 0)

# Compute the 2D DCT of the M×N mystery image with coefficients cˆi, i=1,...,MN.
dct_case2 = dct2(case2)

# Keep its K largest non-DC DCT coefficients, now denoted as cˆ1,cˆ2,...,cˆK.
# Use the same K<MN, found in the watermark insertion part above.
c_hat2 = dct_case2[index[:,0], index[:,1]]

# Estimate an approximation of the watermark in your mystery image: ωˆi = (cˆi − ci)/αci , 1 ≤ i ≤ K
w2 = (c_hat2 - ci) / 0.1 * ci # α=0.1 controls the strength of the watermark

# Mean ω ̄ of the watermark sequence ω1,ω2,...,ωK
mean_watermark2 = np.mean(w)
# Mean of the approximated watermark sequence ωˆ1, ωˆ2, . . . , ωˆK
mean_approxWatermark2 = np.mean(w2)

# Measure the similarity of ωˆi with ωi using the correlation coefficient gamma
# (see assignment for formula), where 1 ≤ i ≤ K, so in np.sum we have initial=1 and where=K
gamma2 = np.sum([(w2 - mean_approxWatermark2) * (w - mean_watermark2)], initial=1, where=K) / np.sqrt(np.sum([(w2 - mean_approxWatermark2)**2], initial=1, where=K) * np.sum([(w - mean_watermark2)**2],initial=1, where=K))
print(gamma2) # Division by 0 because the original image doesn't have an invisible watermark
# Zero-size array passed to numpy.mean raises this warning (Division by 0)

plt.show()

