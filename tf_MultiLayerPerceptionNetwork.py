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

# parameters for training/testing
learningRate = 0.5  # for gradient descent
numTrainingExamples = 35189
numTestingExamples = 14811
totalNumExamples = 50000

# parameters for the neural network
n_hidden_1 = 64 # number of features in 1st hidden layer 
n_hidden_2 = 32 # number of features in 2nd hidden layer
trainData_scaled = preprocessing.scale(trainData)
testData_scaled = preprocessing.scale(testData)
print(type(trainData_scaled))
print(trainData_scaled.dtype)
# OBTAIN DATA
sess = tf.InteractiveSession()

# SET UP NODES FOR INPUTS AND OUTPUTS

X = tf.placeholder(tf.float32, shape=[None, 618])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# IMPLEMENT MODEL

# define model
def multilayer_perceptron(X, weights, biases):
    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([618, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1]))
}

# instantiate model
prediction = multilayer_perceptron(X, weights, biases)

# define cost function and optimizer for each training step
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y_))
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# used for testing the data
predicted_class = tf.greater(tf.sigmoid(prediction), 0.5)
correct_prediction = tf.equal(predicted_class, tf.equal(y_, 1.0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # variable for whether prediction is correct (0 or 1)

# initialize the global variables
sess.run(tf.global_variables_initializer())

# TRAIN MODEL

# run the training step on training set
numCorrectTrainingExamples = 0
for i in range(1,35189):
	# run one step of gradient descent
	sess.run(train_step, feed_dict={X: np.reshape(trainData_scaled[i], (1, 618)), y_: np.reshape(trainResults[i], (1, 1))})
	# check to see how accurate the model is so far
	train_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainData_scaled[i], (1, 618)), y_: np.reshape(trainResults[i], (1, 1))})
	numCorrectTrainingExamples += train_accuracy
	if i % 1000 == 0 and i != 0:
	   print("Training step %d: training accuracy %f%%"%(i, (numCorrectTrainingExamples/i) * 100))

# TEST MODEL

# run the evaluation on the test set
numCorrectTestExamples = 0;
for i in range(1,14811):
	test_accuracy = accuracy.eval(feed_dict={X: np.reshape(testData_scaled[i], (1, 618)), y_: np.reshape(testResults[i], (1, 1))})
	numCorrectTestExamples += test_accuracy
	# check the test result accuracy
	if i % 1000 == 0 and i != numTrainingExamples:
		print("Testing step %d: testing accuracy %f%%"%(i, (numCorrectTestExamples/(i - numTrainingExamples)) * 100))

# usually around 70% for training set and 50% for test set (with and without feature scaling)
print("Final results: training accuracy of %f%%, testing accuracy of %f%%"%((numCorrectTrainingExamples/numTrainingExamples) * 100, (numCorrectTestExamples/numTestingExamples) * 100))
