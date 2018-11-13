import analysis as an
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)


cv_values = [3,5,7]
clf = RandomForestClassifier(n_estimators=35)
for v in cv_values:
	print str(v) +"-fold cross validation results for RandomForestClassifier with n_estimators=35"
	an.cross_validate(clf, v, an.process_data, an.patients)

clf = DecisionTreeClassifier(max_depth=None)
for v in cv_values:
	print str(v) +"-fold cross validation results for DecisionTreeClassifier"
	an.cross_validate(clf, v, an.process_data, an.patients)

clf = ExtraTreesClassifier(n_estimators=35)
for v in cv_values:
	print str(v) +"-fold cross validation results for ExtraTreesClassifier with n_estimators=35"
	an.cross_validate(clf, v, an.process_data, an.patients)

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=35)
for v in cv_values:
	print str(v) +"-fold cross validation results for AdaBoostClassifier with n_estimators=35 and max_depth=3"
	an.cross_validate(clf, v, an.process_data, an.patients)	

clf = DecisionTreeRegressor()
for v in cv_values:
	print str(v) +"-fold cross validation results for DecisionTreeRegressor"
	an.cross_validate(clf, v, an.process_data, an.patients, regression=True)