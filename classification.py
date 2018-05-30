import numpy 
import os
import math

def sigmoid(x):
	ans = math.exp(-x)
	return(1/(1+ans))
file_name= "data.txt"
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
	labels.append(a)
	features.append(x)
	del (x)
weights= numpy.array([2,0.3,0,5,0.06,0.07,0.8,0.9,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.8])
alpha=0.0009	
l = 30
for iter in range(0,9000):
	cost =0.0
	sumWeights=0.0
	for s in range(0, len(weights)):
			SumWeights= sumWeights+ weights[s]**2
	cost_der=numpy.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
	for i in range(0,len(features)):
		h=0.0
		for w in range(0,len(weights)):
			h= h+ weights[w]* float(features[i][w])
		h= sigmoid(h)
		print("h", h, labels[i])
		if (float(labels[i])==0):
			try:
				ans= math.log(1-h)
			except ValueError:
   		 		ans = float('inf')
			cost =cost- ans
		if(float(labels[i])==1):
			try:
				ans= math.log(h)
			except ValueError:
   		 		ans = float('inf')
			cost =cost- ans
		for j in range(0,len(weights)):
			cost_der[j]= cost_der[j]+ (h-float(labels[i]))*float(features[i][j])
	print("cost", cost/(len(features))+ l*sumWeights/(2*len(features)))
	for y in range(0,len(weights)):
		weights[y]= weights[y]-alpha*(cost_der[y]+ l*weights[y])/len(features)
	del (cost_der)
for a in range (0,len(weights)):
	print(weights[a])
h_test=0.0
test= numpy.array([47,14.3,5.6,2.3,5.1,44.6,0,0.2,18.2,1,1.7,58.8,0.4,1.1,1.5,2.8,0.7,0.2,1.6])
for w in range(0,len(weights)):
	h_test=h_test+ test[w]* weights[w] 
print("test", sigmoid(h_test))