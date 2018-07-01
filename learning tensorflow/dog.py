import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import sklearn.preprocessing 
import cv2
import os

seed = 128
rng = np.random.RandomState(seed)

root= os.path.abspath('/home/ruchika/Documents/Summer of Science ML/dog_breed')
data_dir= os.path.join(root, 'data')
train_dir= os.path.join(data_dir, 'train')
train_data=pd.read_csv(os.path.join(data_dir, 'labels.csv'))
train_data.head()
filenames=[]
for images in os.listdir(train_dir):
	filenames.append(images)
	print(images)
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_jpeg(content, channels=3)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [224, 224])
image_batch = tf.train.batch([resized_image], batch_size=8)
trainx=[]
image_number=0
#reading images
for images in os.listdir(train_dir):
	image_path= os.path.join(train_dir, images)
	img = imread(image_path, flatten=True)
	img= cv2.resize(img, (100,100))
	image_number=image_number+1
	img = img.astype('float32')
	tt= np.reshape(img, [10000])
	trainx.append(tt)
	print(image_number)
#dividing training data in train and cv sets
split_size= int(len(trainx)*0.8)
train_x, val_x = trainx[:split_size], trainx[split_size:]
train_y, val_y = train_data.label.values[:split_size], train_data.label.values[split_size:]
print(train_data.label.values)
# print(train_x)
#Defining the neural network
input_units= 10000
hidden_layer1= 1024
hidden_layer2= 256
hidden_layer3= 32
output_units= 120
epochs= 5
learning_rate=0.001
batch_size=10
x = tf.placeholder(tf.float32, [len(filenames), input_units])
y = tf.placeholder(tf.float32, [len(filenames), output_units])
#Initialising weights and biases
print("DONE")
w1= tf.get_variable('w1', initializer= tf.random_normal([input_units, hidden_layer1], seed= seed), dtype='float32')
b1= tf.get_variable('b1', initializer= tf.random_normal([hidden_layer1], seed= seed))
w2= tf.get_variable('w2', initializer= tf.random_normal([hidden_layer1, hidden_layer2], seed= seed), dtype='float32')
b2= tf.get_variable('b2', initializer= tf.random_normal([hidden_layer2], seed= seed))
w3= tf.get_variable('w3', initializer= tf.random_normal([hidden_layer2, hidden_layer3], seed= seed), dtype='float32')
b3= tf.get_variable('b3', initializer= tf.random_normal([hidden_layer3], seed= seed))
w4= tf.get_variable('w4', initializer= tf.random_normal([hidden_layer3, output_units], seed= seed), dtype='float32')
b4= tf.get_variable('b4', initializer= tf.random_normal([output_units], seed= seed))
#Calculating hidden layers
hidden1= tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
hidden2= tf.nn.relu(tf.add(tf.matmul(hidden1, w2), b2))
hidden3= tf.nn.relu(tf.add(tf.matmul(hidden2, w3), b3))
output_layer= tf.sigmoid(tf.add(tf.matmul(hidden3, w4), b4))
cost= tf.reduce_mean(tf.square(y-output_layer))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer.minimize(cost)
# Training the neural network
y=train_y
print(y)
y_true= np.zeros([len(y), 120], dtype='float32')
for f in range(0, len(y_true)):
	h= y[f]
	if(h!=121):
		y_true[f][h-1]=1
# enc= sklearn.preprocessing.OneHotEncoder()
# y= pd.get_dummies(train_data.label.values)
# enc.fit(y)
# train_xi= tuple(train_x)
# train_xz= tf.convert_to_tensor(train_x, dtype= tf.float32)
sess= tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(y_true)
# train_z.eval(session=sess)
# print(yi)
# print(image_batch.eval(session=sess))
for epoch in range(0, 12):
	print("optimising")
	_, c=sess.run(optimizer, feed_dict={x:train_x,y:y_true })		
	print("Cost", c)
#VAlidation error
cost_v=0.0
for i in range(0, len(val_x)):
	hidden1= tf.nn.relu(tf.add(tf.matmul(val_x, w1), b1))
	hidden2= tf.nn.relu(tf.add(tf.matmul(hidden1, w2), b2))
	hidden3= tf.nn.relu(tf.add(tf.matmul(hidden2, w3), b3))
	output_layer= tf.sigmoid(tf.add(tf.matmul(hidden3, w4), b4))
	cost_v= cost_v+ tf.reduce_mean(tf.square(val_y-output_layer))
print("loss on validation set is ", cost_v)




