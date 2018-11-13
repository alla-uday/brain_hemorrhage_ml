import analysis as an 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

cv_values = [3,5,7]

clf = KNeighborsRegressor(n_neighbors=5)
for v in cv_values:
	print str(v) +"-fold cross validation results for KNeighborsRegressor with n_neighbors=5"
	an.cross_validate(clf, v, an.process_data, an.patients, regression=True)

clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
for v in cv_values:
	print str(v) +"-fold cross validation results for KNeighborsClassifier with n_neighbors=5"
	an.cross_validate(clf, v, an.process_data, an.patients)

clf = RadiusNeighborsClassifier(radius=8, outlier_label=0, weights='uniform')
for v in cv_values:
	print str(v) +"-fold cross validation results for RadiusNeighborsClassifier with n_neighbors=5"
	an.cross_validate(clf, v, an.process_data, an.patients)