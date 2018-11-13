import analysis as an 
from sklearn.svm import SVC

cv_values = [3,5,7]

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
for k in kernels:
	clf = SVC(kernel=k)
	for v in cv_values:
		print str(v) +"-fold cross validation results for SVC, kernel=" + k
		an.cross_validate(clf, v, an.process_data, an.patients)