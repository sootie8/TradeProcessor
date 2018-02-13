from __future__ import print_function

import numpy as np
import tflearn
import json
import math as m
import sys
import os
import json
from random import shuffle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

def Timesteps():
	return 60

#net = tflearn.input_data(shape=[None, Timesteps(), 2])
model = Sequential()
#model.add(LSTM(128, input_shape=(Timesteps(), 2), return_sequences = True))
model.add(LSTM(128, input_shape=(Timesteps(), 2), return_sequences = True))
model.add(LSTM(128, input_shape=(Timesteps(), 2), return_sequences = False))
model.add(Dense(units=1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="adam")

def Setup(): 
	#Load all trade history from sqlite. 
	# Run SQL query, process rows.
	data_file = open('data.json','r');
	jsonStr = data_file.read();

	rows = json.loads(jsonStr)

	print(len(rows))

	pointers = GetPointers(rows)
	shuffle(pointers)

	rows = np.asarray(rows)

	print(rows.shape)

	Round(rows, pointers)


#Store 3 values for each row, back pointer (beggining of dataset), middlepointer (end of dataset and start of prediction set -1) 
#and forward pointer (end of prediction set).
def GetPointers(rows):

	pointers = []
	backpointer = 0
	middlepointer = Timesteps() -1
	forwardpointer = Timesteps() + 9

	while forwardpointer < len(rows) -1:
		forwardpointer += 1
		middlepointer += 1
		backpointer += 1
		pointers.append((backpointer, middlepointer, forwardpointer))

	return pointers


# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        	[r.pop(id) for r in data]
    return data;



def shapeLabel(val):
	arr = np.empty([2])
	arr[0] = 1 if val <= 0.0 else 0
	arr[1] = 1 if val > 0.0 else 0
	return arr

def generateMetricArray(rows, pointers):
	inputRows = []

	for x, y, z in pointers:
		inputSet = rows[x:z+1]
		#Calculate metrics if we want to or not, shape input data.

		inputRows.append(inputSet)

	return inputRows

def getMovement(row):
	startPrice = row[Timesteps() -1][0]
	endPrice = row[len(row) -1][0]

	mov = endPrice / startPrice
	return mov

def Round(rows, pointers):
	rows = generateMetricArray(rows, pointers)

	global model

	#rows = np.asarray(rows);

	#rows.shape = (len(rows), (Timesteps() + 1) *2)
	#rows = 
	#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

	movements = list(map(lambda x: getMovement(x), rows))

	for i in range(len(rows)):
		#rows[i] = scaler.fit_transform(rows[i])
		rows[i] = preprocessing.scale(rows[i])

	labels = list(map(lambda x: x[len(x) -1:, 0], rows))

	rows = np.asarray(rows);

	print(rows.shape)

	rows = rows[:, :Timesteps()]

	labels = np.asarray(labels)
	rows = np.asarray(rows)

	print (labels.shape, rows.shape)

	print (labels)
 
	#labels, rows = EvenOut(labels, rows)

	# Start training (apply gradient descent algorithm).
	model.fit(rows[100:], labels[100:], epochs=1, batch_size=64)

	pred = model.predict(rows[:100])

	correct = 0
	actLessThanCount = 0
	avgOut = 0

	lessThanWrong = 0
	lessThanRight = 0
	greaterThanWrong = 0
	greaterThanRight = 0

	balance = 100


	for x, y, z, w in zip(pred, labels[:100], rows[:100], movements[:100]):
		x = x[0]
		y = y[0]
		z = z[len(z) -1][0]
		print("predicted:", x, "actual", y, "lastdatapoint", z)

		predLessThan = x < z

		actLessThan = y < z

		avgOut += abs(x - y)

		if actLessThan:
			actLessThanCount+= 1

		if predLessThan == actLessThan:
			if predLessThan == True:
				lessThanRight += 1
				balance = balance * (1 + (1 - w))
			else:
				greaterThanRight += 1
				balance = balance * w

			correct += 1
		else:
			if predLessThan == True:
				lessThanWrong += 1
				balance = balance * ( 1 - (w - 1))
			else:
				greaterThanWrong += 1
				balance = balance * w

	#model.save("model.tfl")
	print(correct)
	print(actLessThanCount)
	print(avgOut / 100)
	print ("balance:", balance)

	print("Less Than Wrong", lessThanWrong, "Less Than Right", lessThanRight, "Greater Than Wrong", greaterThanWrong, "Greater Than Right", greaterThanRight)
	model.save('model.keras');
	sys.exit()
#start the program

Setup()

