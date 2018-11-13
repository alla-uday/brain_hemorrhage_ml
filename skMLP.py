import analysis as an 

from sklearn.neural_network import MLPClassifier

cv_values = [3,5,7]
types = ['identity', 'logistic', 'tanh', 'relu']

for t in types:
	clf = MLPClassifier(activation=t)
	for v in cv_values:
		print str(v) +"-fold cross validation results for MLPClassifier, activation=" + t
		an.cross_validate(clf, v, an.process_data, an.patients)

