import numpy as np
import tensorflow as tf

file_name= "new_data.txt"
file = open(file_name, "r")
content= file.readlines()
features=[]
labels=[]
for line in content:
	numbers= line.split()
	x=[]
	x.append(1)
	for i in range(0,len(numbers)-1):
		x.append(numbers[i])
	a=(numbers[i+1])
	labels.append(float(a))
	features.append(x)
	del (x)
weights= np.array([1.0,1.3,2.4,3.5,2.6], dtype='float32')
with tf.Session() as sess:
	my= np.array(features)
	x= tf.constant(my, dtype='float32')
	y_true= tf.constant(labels, dtype='float32')
	w= tf.get_variable(name='w', initializer=weights, dtype='float32')
	sess.run(tf.variables_initializer([w]))
	y_predicted= tf.matmul(x,tf.expand_dims(w,1))
	print(y_predicted.eval(session=sess))
	s= tf.reduce_sum(tf.square(y_true- y_predicted), name='se')
	me_size= tf.get_variable(name='me_size', initializer=82.0, dtype='float32')
	sess.run(tf.variables_initializer([me_size]))
	mse= tf.divide(s,2*me_size, name='mse')
	print("mse")
	print(mse.eval(session=sess))
	opt= tf.train.GradientDescentOptimizer(learning_rate=0.001)
	train= opt.minimize(mse)
	for i in  range(0, 10000):
		sess.run(train)
	print(mse.eval(session=sess))
	print(weights)
	print(y_predicted.eval(session=sess))
