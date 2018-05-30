import numpy 
import os
import math
import random
#Number of clusters=3

def min(point1, point2, point3):
	if ((point1<point2 & point1<point3) || (point1==point2 & point1<point3)||(point1==point3 & point1<point2) ):
		return 1
	if (point2<point1 & point2<point3|| (point1==point2 & point2<point3)||(point2==point3 & point2<point1))
		return 2
	if (point3<point1 & point3<point1|| (point3==point2 & point3<point1)||(point3==point1 & point3<point2))
		return 3

#initialsing cluster centroids
centroid_1=[]
centroid_2=[]
centroid_3=[]
for i in range(0,4):
	centroid_3.append(random.randint(0,4))
	centroid_2.append(random.randint(0,4))
	centroid_1.append(random.randint(0,4))
file_name= "iris.txt"
file = open(file_name, "r")
content= file.readlines()
features=[]
for line in content:
	numbers= line.split()
	x=[]
	for i in range(0,len(numbers)-1):
		x.append(float(numbers[i]))
	features.append(x)
	del (x)
for iter in range(0,150):
	cluster1_points=[]
	cluster2_points=[]
	cluster3_points=[]
	cluster_index=[]
	for i in range (0,len(features)):
		distance_1= sum((centroid_1-features[i])**2)
		distance_2= sum((centroid_2-features[i])**2)
		distance_3= sum((centroid_3-features[i])**2)
		index=min(distance_1,distance_2,distance_3)
		if (index==1):
			cluster1_points.append(features[i])
		if (index==2):
			cluster2_points.append(features[i])
		if (index==3):
			cluster3_points.append(features[i])
		cluster_index.append(index)
	#finding centroid of each clusters
	sum1=numpy.array([0.0,0.0,0.0,0.0])
	sum2=numpy.array([0.0,0.0,0.0,0.0])
	sum3=numpy.array([0.0,0.0,0.0,0.0])
	for i1 in range(0, len(cluster1_points)):
		sum1=sum1+ cluster1_points[i1]
	sum1=sum1/len(cluster1_points)
	for i2 in range(0, len(cluster2_points)):
		sum2=sum2+ cluster2_points[i2]
	sum2=sum2/len(cluster2_points)
	for i3 in range(0, len(cluster3_points)):
		sum3=sum3+ cluster3_points[i3]
	sum3=sum3/len(cluster3_points)
	#calculating cost 
	for i1 in range(0, len(cluster1_points)):
		cost1= sum((sum1-cluster1_points[i11])**2)
	for i2 in range(0, len(cluster2_points)):
		cost2= sum((sum2-cluster2_points[i21])**2)
	for i3 in range(0, len(cluster3_points)):
		cost3= sum((sum3-cluster3_points[i3])**2)
	print("cost", cost1+cost2+cost3)
	#updating centroids
	centroid_1= sum1
	centroid_2=sum2
	centroid_3=sum3
	del(cluster1_points)
	del(cluster2_points)
	del(cluster3_points)
	del(cluster_index)
