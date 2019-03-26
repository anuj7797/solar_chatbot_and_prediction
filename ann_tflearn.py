import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from random import shuffle


def ann(train_x, train_y, test_x, test_y):
	
	train_x = np.array(train_x, dtype="float64").reshape(-1, 8, 1, 1)
	test_x = np.array(test_x, dtype="float64").reshape(-1, 8, 1, 1)
	train_y = np.array(train_y, dtype="float64").reshape(-1, 1)
	test_y = np.array(test_y, dtype="float64").reshape(-1, 1)
	
	net = input_data(shape=[None, 8, 1, 1], name='input')

	net = fully_connected(net, 1024, activation='relu')
	# net = dropout(net, 0.8)

	# net = fully_connected(net, 1024, activation='relu')
	# net = dropout(net, 0.8)
	
	# net = fully_connected(net, 2048, activation='relu')
	
	# net = fully_connected(net, 1024, activation='relu')
	
	# net = fully_connected(net, 512, activation='relu')
	# net = dropout(net, 0.8)
	
	# net = fully_connected(net, 512, activation='relu')
	# net = dropout(net, 0.8)

	net = fully_connected(net, 1, activation='relu')

	net = regression(net, optimizer='adam', learning_rate=0.003, loss='mean_square', name='targets')
	model = tflearn.DNN(net)

	print("Model Training")
	model.fit({'input': train_x}, {'targets': train_y}, n_epoch=1000,
			  validation_set=({'input': test_x}, {'targets': test_y}),
			  snapshot_epoch=True, show_metric=True, run_id='ann')
	print("Model Trained")
