import analysis as an 

from sklearn.ensemble import GradientBoostingClassifier

cv_values = [3,5,7]

clf = GradientBoostingClassifier(n_estimators=300, max_depth=6, loss='exponential')
for v in cv_values:
	print str(v) +"-fold cross validation results for GradientBoostingClassifier(n_estimators=300, max_depth=6, loss=exponential)"
	an.cross_validate(clf, v, an.process_data, an.patients)