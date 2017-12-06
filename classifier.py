# importing the header for saving the classifier as a file for extended use
from sklearn.externals import joblib as jlb

#Header for calling MNIST dataset 
from sklearn import datasets

#The Histogram of Oriented Graphics
from skimage.feature import hog

#header of Sklearn library for calling the LinearSVC
from sklearn.svm import LinearSVC

#header of Sklearn library for checking accuracy
from sklearn.metrics import accuracy_score

import numpy as np

'''We are done with all the imports and now is the time to initialise and 
create a classifier with the data from the MNIST Dataset'''

#call the complete data from the MNIST Origianl dataset
data_col = datasets.fetch_mldata("MNIST Original")
#call the total dataset feature and save it in array
data_feat = np.array(data_col.data, 'int16')
#call the total dataset labels and save it in array
data_lab = np.array(data_col.target, 'int')

# Now we will calculate the HOG features for each image from the dataset
list_hog = []
for features in data_feat:
	fd = hog(features.reshape((28,28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
	list_hog.append(fd)
hog_feat = np.array(list_hog, 'float64')

#Seperating training_features and training_labels (0.8*Total_data) from the dataset
feat = hog_feat[:55999]
lab = data_lab[:55999]

#Seperating test_features and training_labels (0.8*Total_data) from the dataset
test_feat = hog_feat[56000:]
test_lab = data_lab[56000:]

# Initializing the linear SVM classifier
clasf = LinearSVC()
#Entering the training features and labels in the classifier
clasf.fit(feat, lab)
#dumping the classifier for extended further usage.
jlb.dump(clasf,"clf_dump.pkl", compress=3)

# Testing the data using the builtin accuracy_score function
pred = clasf.predict(test_feat)
accu = accuracy_score(test_lab, pred)

#printing the accuracy
print(accu)

