import numpy
import os

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
	labels.append(a)
	features.append(x)
	del (x)
weights= numpy.array([2,2.3,3.4,4.5,5.6])
alpha=0.00029
l =18
for iter in range(0,90000):
	cost =0.0
	sumWeights=0
	for s in range(0, len(weights)):
			SumWeights= sumWeights+ weights[s]**2
	cost_der=numpy.array([0.0,0.0,0.0,0.0,0.0])
	for i in range(0,len(features)):
		h=0.0
		for w in range(0,len(weights)):
			h= h+ weights[w]* float(features[i][w])
		cost= cost+ (h-float(labels[i]))**2 
		for j in range(0,len(weights)):
			cost_der[j]= cost_der[j]+ (h-float(labels[i]))*float(features[i][j])
	print("cost", cost/(2*len(features))+ l*sumWeights/(2*len(features)))
	for y in range(0,len(weights)):
		weights[y]= weights[y]-alpha*(cost_der[y]+ l*weights[y])/len(features)
	del (cost_der)
for a in range (0,len(weights)):
	print(weights[a])
test=numpy.array([1,62.25,3.56,50,1.49734])
h_test=0.0
for w in range(0,len(weights)):
			h_test= h_test+ weights[w]* test[w]
print(h_test)










