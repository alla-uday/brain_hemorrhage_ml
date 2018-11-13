import analysis as an 
from sklearn.ensemble import BaggingClassifier

cv_values = [3,5,7]

clf = BaggingClassifier(n_estimators=20)
for v in cv_values:
	print str(v) +"-fold cross validation results for BaggingClassifier(n_estimators=20)"
	an.cross_validate(clf, v, an.process_data, an.patients)