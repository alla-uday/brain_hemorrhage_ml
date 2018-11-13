import sys
import csv
import numpy as np
from operator import itemgetter
from numpy import array
import tensorflow as tf
from sklearn import preprocessing

with open("train_data_final_50K.csv") as f:
    reader = csv.reader(f)
    data = [r for r in reader]

new_data = []
for line in data:
	test = []
	for num in line:
		test.append(float(num))
	new_data.append(test)

#sort data according to numeric order of patient ID => k is the sorted array
patient_id = {}
k = sorted(new_data, key=itemgetter(0))
for record in k:
	if patient_id.get(record[0], None): #if key already exists
		patient_id[record[0]] = patient_id[record[0]] + 1
	else:
		patient_id[record[0]] = 1

train_id = []
test_id = []
total = 0

for k in patient_id.keys():
	if total > 35000:
		test_id.append(k)
	else:
		total = total + patient_id[k]
		train_id.append(k)

print(total)
print("train", len(train_id))
print("test", len(test_id))
trainingData = []
trainingResults = []
testingData = []
testingResults = []

for d in new_data:
    if d[0] in train_id:
        trainingData.append(d[4:622])
        trainingResults.append(d[622])
    else:
        testingData.append(d[4:622])
        testingResults.append(d[622])
        
trainData = array(trainingData)
testData = array(testingData)
trainResults = array(trainingResults)
testResults = array(testingResults)



# SET PARAMETERS

learningRate = 0.5  # for gradient descent
numTrainingExamples = 35189
numTestingExamples = 14811
totalNumExamples = 50000

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 618])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# SET UP WEIGHTS/PARAMETERS FOR MODEL

# initialize weights and bias all to 0
W = tf.Variable(tf.zeros([618, 1]))
b = tf.Variable(tf.zeros([1, 1]))
sess.run(tf.global_variables_initializer())

# IMPLEMENT MODEL

# get the predicted output from the model
# currently, this is just a logistic regression model with a single linear layer
y = tf.matmul(X, W) + b

# use the cross entropy as the error for the prediction (sigmoid since binary logistic regression)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

# used for testing the data
prediction = tf.sigmoid(y)  # float between 0 and 1 that represents probability of hemorrhage
predicted_class = tf.greater(prediction, 0.5)
correct_prediction = tf.equal(predicted_class, tf.equal(y_, 1.0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # variable for whether prediction is correct (0 or 1)

# TRAIN MODEL

# use gradient descent with specified learning rate
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy)
# run the training step on training set
numCorrectTrainingExamples = 0
for i in range(0, 35189):
	# run one step of gradient descent
	sess.run(train_step, feed_dict={X: np.reshape(trainData[i], (1, 618)), y_: np.reshape(trainResults[i], (1, 1))})
	# check to see how accurate the model is so far
	train_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainData[i], (1, 618)), y_: np.reshape(trainResults[i], (1, 1))})
	numCorrectTrainingExamples += train_accuracy
	if i % 1000 == 0 and i != 0:
		print("Training step %d: training accuracy %f%%"%(i, (numCorrectTrainingExamples/i) * 100))

# TEST MODEL

# run the evaluation on the test set
numCorrectTestExamples = 0;
for i in range(0,14811):
	test_accuracy = accuracy.eval(feed_dict={X: np.reshape(testData[i], (1, 618)), y_: np.reshape(testResults[i], (1, 1))})
	numCorrectTestExamples += test_accuracy
	# check the test result accuracy
	if i % 1000 == 0 and i != numTrainingExamples:
		print("Testing step %d: testing accuracy %f%%"%(i, (numCorrectTestExamples/(i - numTrainingExamples)) * 100))

# usually around 96% for training set and 80% for test set (no feature scaling)
# usually around 93% for training set and 73% for test set (with feature scaling)
print("Final results: training accuracy of %f%%, testing accuracy of %f%%"%((numCorrectTrainingExamples/numTrainingExamples) * 100, (numCorrectTestExamples/numTestingExamples) * 100))
