from __future__ import print_function

import numpy as np
import tflearn
import json
import psycopg2
import math as m
import talib
import MySQLdb
from random import shuffle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

con = MySQLdb.Connection(
    host='192.168.0.11',
    user='new',
    passwd='jh1995',
    port=3306,
    db='TradeDB'
)

def Timesteps():
	return 60

#net = tflearn.input_data(shape=[None, Timesteps(), 2])
model = Sequential()
model.add(LSTM(128, input_shape=(Timesteps(), 2), return_sequences = True))
model.add(LSTM(128, input_shape=(Timesteps(), 2), return_sequences = False))
model.add(Dense(units=1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="adam")

def Setup(): 
	#Load all trade history from sqlite. 
	# Run SQL query, process rows.
	print('before query')
	cur = con.cursor()
	sql = 'SELECT close, volume FROM tenminutetrades'
	cur.execute(sql)
	rows = cur.fetchall()

	pointers = GetPointers(rows)
	shuffle(pointers)

	rows = np.asarray(rows)

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



def EvenOut(labels, rows):
	lessCount = len(filter(lambda x: x[0] == 1, labels))
	moreCount = len(filter(lambda x: x[0] == 0, labels))

	maxOfEach = min(lessCount, moreCount)

	lcount = 0
	hcount = 0

	less = filter(lambda xy: xy[0][0] == 1,zip(labels, rows))
	more = filter(lambda xy: xy[0][0] == 0,zip(labels, rows))
	less = less[:maxOfEach]
	more = more[:maxOfEach]

	newcomb = less + more

	shuffle(newcomb)
	nlabels, nrows  = zip(*newcomb)
	return nlabels, nrows

def Round(rows, pointers):
	rows = generateMetricArray(rows, pointers)

	global model

	#rows = np.asarray(rows);

	#rows.shape = (len(rows), (Timesteps() + 1) *2)
	#rows = preprocessing.scale(rows)
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

	for i in xrange(len(rows)):
		rows[i] = scaler.fit_transform(rows[i])

	labels = map(lambda x: x[len(x) -1:, 0], rows)

	rows = np.asarray(rows);

	print(rows.shape)

	rows = rows[:, :Timesteps()]

	labels = np.asarray(labels)
	rows = np.asarray(rows)

	print (labels.shape, rows.shape)
 
	#labels, rows = EvenOut(labels, rows)

	# Start training (apply gradient descent algorithm).
	model.fit(rows[100:], labels[100:], epochs=5, batch_size=16)

	pred = model.predict(rows[:100])

	correct = 0
	actLessThanCount = 0
	avgOut = 0


	for x, y, z in zip(pred, labels[:100], rows[:100]):
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
			correct += 1

	#model.save("model.tfl")
	print(correct)
	print(actLessThanCount)
	print(avgOut / 100)

#start the program

Setup()

