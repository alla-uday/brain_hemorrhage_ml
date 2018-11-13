import analysis as an 
from sklearn.neighbors.nearest_centroid import NearestCentroid

cv_values = [3,5,7]
clf = NearestCentroid()
for v in cv_values:
	print str(v) +"-fold cross validation results for NearestCentroid"
	an.cross_validate(clf, v, an.process_data, an.patients)
