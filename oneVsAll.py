import numpy
import os
import math

def sigmoid(x):
	ans = math.exp(-x)
	return 1/(1+ans)

#Number of classes = K
#read features
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
#Final Weights
final_weights=[]
#Applying one VS all
for k in range(0,K):
	#Initialising default weights for each class
	weights=[]
	for o in range(0, len(features[0])):
		weights.append(round(random.uniform(1,3),2))
	oneVsAll=[]
	oneVsAll_labels=[]
	for i in range(0, features):
		oneVsAll.append(features[i])
		if (labels[i]==k):
			oneVsAll_labels.append(1)
		else:
			oneVsAll_labels.append(0)
	for iterations in range(0, 800):
		cost=0.0
		cost_der=numpy.zeros([])
		#Applying Classification on this features set
		for j in range(0, len(oneVsAll)):
			h=0.0
			for w in range(0, len(weights)):
				h=h+weights[w]*oneVsAll[j][w]
			h=sigmoid(h)
			#calculating cost
			if (oneVsAll_labels[j]==0):
			try:
				ans= math.log(1-h)
			except ValueError:
   		 		ans = float('inf')
			cost =cost- ans
			if(oneVsAll_labels[j]==1):
				try:
					ans= math.log(h)
				except ValueError:
	   		 		ans = float('inf')
				cost =cost- ans
			#calculating derivative of cost
			for l in range(0,len(weights)):
				cost_der[l]= cost_der[l]+ (h-oneVsAll_labels[j])*float(oneVsAll[j][l])
		for we in range(0, len(weights)):
			weights[we]=weights[we]-(learning_rate*cost_der/len(features))
	final_weights.append(weights)
