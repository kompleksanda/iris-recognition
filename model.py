
import pickle
from time import time
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from loc import match

h = 64
w = 4
# Load train and test data
X = pickle.load(open("data/X_iris_svd.p", "rb"))
y = pickle.load(open("data/y_iris_svd.p", "rb"))

print (X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2018)
m, n = match(X_train, y_train, X_test, y_test, True, 4)
print(m,n)

N_COMPONENTS = 4
print("Extracting the top %d eigenfaces from %d faces" % (N_COMPONENTS, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=N_COMPONENTS, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
eigenfaces = pca.components_.reshape((N_COMPONENTS, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

param_grid = {'C': [1e3, 5e3],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(
                   SVC(kernel='rbf', class_weight='balanced'),
                   param_grid)

print(X_train_pca.shape)
print(y_train.shape)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
