import tensorflow as tf
import numpy as np
import math

file_name = "data.txt"
file = open(file_name, "r")
content = file.readlines()
features=[]
labels=[]
for line in content:
	numbers = line.split()
	x=[]
	x.append(1)
	for i in range(0,len(numbers)-1):
		x.append(numbers[i])
	a=(numbers[i+1])
	labels.append(float(a))
	features.append(x)
	del (x)
#print("Number of training examples", len(features))
weights= np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.2,0.3,0.4],dtype='float32')
w= tf.get_variable(name='w', initializer=weights, dtype='float32')
with tf.Session() as sess:
	sess.run(tf.variables_initializer([w]))
	my= np.array(features)
	x= tf.constant(my, dtype='float32')
	y_true= tf.constant(labels, dtype='float32')
	y_predicted= tf.matmul(x,tf.expand_dims(w,1))
	y_predicted=tf.sigmoid(y_predicted)
	#print(y_predicted[0].eval(session=sess))
	s= tf.reduce_sum(tf.square(y_true- y_predicted), name='s')
	me_size= tf.get_variable(name='me_size', initializer=198.0, dtype='float32')
	sess.run(tf.variables_initializer([me_size]))
	mse= tf.divide(s,2*me_size, name='mse')
	#Defining an optimiser
	opt= tf.train.GradientDescentOptimizer(learning_rate=8)
	train= opt.minimize(mse)
	for iterations in range(0, 90000):
		sess.run(train)
		print(mse.eval(session=sess))
		print(w.eval(session=sess))
	#print(w.eval(session=sess))
	print(mse.eval(session=sess))
	#print(y_predicted.eval(session=sess))