import cv2
import numpy as np
import glob

## Implementation of the code from https://learnopencv.com/eigenface-using-opencv-c-python/

# Creates a data matrix where the images are in rows
def createDataMatrix(images):
    numImages = len(images)
    data = np.matrix(numImages, dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten() #creates 1-dimensional images
        data[i, :] = image

    return data

# Creates a new face based on slider values
def createNewFace(*args):
    # The mean image
    output = averageFace

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars")
        weight = sliderValues[i] - MAX_SLIDER_VALUE / 2
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size
    output = cv2.resize(output, (0, 0), fx=2, fy=2)
    cv2.imshow("Result", output)


# Number of EigenFaces
NUM_EIGEN_FACES = 10

# Maximum weight
MAX_SLIDER_VALUE = 255

# Read in the images
# Grandma images
gran = []
for img in glob.glob("/Users/alisatodorova/surfdrive/Year2/Intro to IVP/Todorova_project2/venv/Task4/gran/*.JPG"):
	gran.append(cv2.imread(img))

# Man images
man = []
for img in glob.glob("/Users/alisatodorova/surfdrive/Year2/Intro to IVP/Todorova_project2/venv/Task4/man/*.JPG"):
	man.append(cv2.imread(img))

# Woman images
woman = []
for img in glob.glob("/Users/alisatodorova/surfdrive/Year2/Intro to IVP/Todorova_project2/venv/Task4/woman/*.JPG"):
	woman.append(cv2.imread(img))

# Create data matrix for PCA
data_gran = createDataMatrix(gran)
data_man = createDataMatrix(man)
data_woman = createDataMatrix(woman)

# Compute the eigenvectors from the stack of images created
mean_gran, eigenVectors_gran = cv2.PCACompute(data_gran, mean=None, maxComponents=NUM_EIGEN_FACES)
mean_man, eigenVectors_man = cv2.PCACompute(data_man, mean=None, maxComponents=NUM_EIGEN_FACES)
mean_woman, eigenVectors_woman = cv2.PCACompute(data_woman, mean=None, maxComponents=NUM_EIGEN_FACES)

# Average Faces
averageFace_gran = mean_gran.astype(np.uint8)
averageFace_man = mean_man.astype(np.uint8)
averageFace_woman = mean_woman.astype(np.uint8)

# Put the eigen faces into a list
eigenFaces_gran = []
for eigenVector in range(eigenVectors_gran.shape[0]):
	eigenFace_gran = eigenVectors_gran[eigenVector]
	eigenFaces_gran.append(eigenFace_gran)

eigenFaces_man = []
for eigenVector in range(eigenVectors_man.shape[0]):
	eigenFace_man = eigenVectors_man[eigenVector]
	eigenFaces_man.append(eigenFace_man)

eigenFaces_woman = []
for eigenVector in range(eigenVectors_woman.shape[0]):
	eigenFace_woman = eigenVectors_woman[eigenVector]
	eigenFaces_woman.append(eigenFace_woman)


# Display result at 2x size
cv2.imshow("Result Grandma", averageFace_gran)
cv2.imwrite("resultGran.jpg", averageFace_gran)

cv2.imshow("Result Man", averageFace_man)
cv2.imwrite("resultMan.jpg", averageFace_man)

cv2.imshow("Result Woman", averageFace_woman)
cv2.imwrite("resultWoman.jpg", averageFace_woman)

cv2.waitKey(0)
cv2.destroyAllWindows()
