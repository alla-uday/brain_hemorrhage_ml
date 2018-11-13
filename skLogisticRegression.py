import analysis as an

from sklearn.linear_model import LogisticRegression

cv_values = [3,5,7]
logisticRegr = LogisticRegression()
for v in cv_values:
	print str(v) +"-fold cross validation results for LogisticRegression"
	an.cross_validate(logisticRegr, v, an.process_data, an.patients)