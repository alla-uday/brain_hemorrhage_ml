import analysis as an

from sklearn import linear_model

loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
cv_values = [3,5,7]

for l in loss:
	clf = linear_model.SGDClassifier(loss=l)
	for v in cv_values:
		print str(v) + '-fold cross validation results for SGDClassifier with loss=' + l
		an.cross_validate(clf, v, an.process_data, an.patients)
