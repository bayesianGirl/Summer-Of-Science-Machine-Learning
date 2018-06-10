import numpy 
import os
import math 
import random

def sigmoid(x):
	ans = math.exp(-x)
	return(1/(1+ans))

def SigmoidDerivative(x):
	return math.exp(-x)/((1+math.exp(-x))**2)

errorfile= open("error.dat", "w")
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
	labels.append(a)
	features.append(x)
	del (x)
print(len(features[0]))
L = input("Number of hidden layers") 
activation=[]
for l in range(0,L):
	act = input("Activation units in layer") 
	activation.append(act)
activation.append(1)
#Initialising 3D weight matrix randomnly
weights_total=[]
al = len(features[0])-1
for i in range(0,L):
	weights=[]
	for k in range (0,activation[i]):
		temp_weights=[]
		for j in range(0,al+1):
			temp_weights.append(round(random.uniform(1,3),2))
		weights.append(temp_weights)
		del(temp_weights)
	al= activation[i]
	weights_total.append(weights)
	del(weights)
# initialising weights to the last layer 
temp_weights=[]
weights=[]
for p in range (0,activation[len(activation)-2]+1):
		temp_weights.append(round(random.uniform(1,3),2))
weights.append(temp_weights)
weights_total.append(weights)
del(temp_weights)
del(weights)
ac=[]
learning_rate=0.6
epochs=0
while(True):
	for i in range(0, len(features)):
		# Forward Propagation
		cost=0.0
		total_units=[]
		z=[]
		ac=features[i]
		for l in range(0,L+1):
			units=[]
			z_temp=[]
			for num in range(0,activation[l]):
				h=0.0
				for w in range(0,len(weights_total[l][num])):
					#print(weights_total[l][num])
					h=h+weights_total[l][num][w]*float(ac[w])
				z_temp.append(h)
				h= sigmoid(h)
				units.append(h)
			if(l!=L):
				units.insert(0, 1)
			ac=units
			total_units.append(units)
			z.append(z_temp)
			del(units)
			del(z_temp)
		#calculating cost
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
		print("cost", cost)
		errorfile.write(str(cost))
		errorfile.write("\n")
		#Backpropagation
		#print(total_units)
		error=[]
		error.append(total_units[L][0]-float(labels[i]))
		#print(error)
		for r in range(0,L):
			s=[]
			sig_der=[]
			final_error=[]
			final_product=[]
			#print("total", total_units[r])
			sig_der.append(SigmoidDerivative(1))
			for d in range(0,len(z[L-r-1])):
					sig_der.append(SigmoidDerivative(z[L-r-1][d]))
			#print("weights",weights_total[L-r])
			if(len(error)==1):
				#print("yes")
				prod=numpy.dot(weights_total[L-r], error[0])
			else:
				error=numpy.delete(error, 0,0)
				prod=numpy.dot( error,weights_total[L-r])
			for t in range(0, len(prod)):
				try:
					length= len(prod[t])
				except TypeError:
	   		 		length=1
				if(length>1):
					for u in range(0,len(prod[t])):
						final_error.append(prod[t][u]*sig_der[t])
				if(length==1):
					final_error.append(prod[t]*sig_der[t])
			size = numpy.shape(weights_total[L-r])
			for ze in range(0, size[0]):
				for p in range(0, len(total_units[L-r-1])):
					weights_total[L-r][ze][p]= weights_total[L-r][ze][p]- learning_rate*final_error[p]*total_units[L-r-1][p]
			error=final_error
			del(sig_der)
			del(final_error)
			epochs=epochs+1
		print("Number of iterations",epochs)

