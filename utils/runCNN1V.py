import os
import numpy as np
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D, Dropout
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class runCNN1V:
	def __init__(self, x_train, y_train, ckptpath="model.h5", retrain=True, n_classes=2):
		'''
		===================================
		Settings
		===================================
		'''
		
		## Batch size
		batch_size = 256
		
		## Epochs
		epochs = 60
		
		## Loss
		# - f1_loss / "binary_crossentropy"  / "categorical_crossentropy"
		loss = "binary_crossentropy"
		
		## Metrics
		metrics = ["accuracy"]

		## Optimizer
		# - Learning rate (Default: -1)
		lr = 0.0001

		# - Choose optimizer
		optimizer = "adam"

		# - List optimizers (Default parameters)
		doptimizers = {
			"sgd": keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
			"rmsprop": keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),
			"adagrad": keras.optimizers.Adagrad(learning_rate=0.01),
			"adadelta": keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
			"adam": keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
			"adamax": keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
			"nadam": keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
		}
		oboptimizer = doptimizers[optimizer]
		if lr>=0: oboptimizer.learning_rate.assign(lr)
		
		## Callback
		callback_list = []
		
		# - Early stopping
		'''
		callback_list.append(keras.callbacks.EarlyStopping(
			monitor = "accuracy",
			patience = 3
		))
		'''
		
		# - Model checkpoint
		callback_list.append(keras.callbacks.ModelCheckpoint(
			#monitor = "val_loss",
			#save_best_only = True,
			save_weights_only = True,
			filepath = ckptpath
		))

		# x size
		xdim = len(x_train[0] if x_train else [])

		# print("* Input Vector Size: {}".format(xdim))

		# Model Layers
		self.model = Sequential()
		self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(xdim, 1)))
		self.model.add(Dropout(0.5))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Flatten())
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dense(n_classes, activation='softmax'))
		
		self.model.compile(loss=loss, optimizer=oboptimizer, metrics=metrics)
	
		# multi-input training

		# Parallelism
		# - setting max CPU cores
		max_cpu_cores = multiprocessing.cpu_count()
		# - configuring tensorflow protocol to choice between CPU|GPU, burning the hardware!!!
		config = tf.ConfigProto(intra_op_parallelism_threads=max_cpu_cores,
								inter_op_parallelism_threads=max_cpu_cores, 
								allow_soft_placement=True,
								device_count = {'GPU': 1, 'CPU': max_cpu_cores})
		# - only for shared GPU's: allocates the GPU memory dynamically instead of all at time, 
		config.gpu_options.allow_growth=True 
		sess = tf.Session(graph=tf.get_default_graph(),config=config) 
		K.set_session(sess)

		# TRAINING

		# - Just Load Weights?
		if not retrain:
			if not os.path.exists(ckptpath):
				exit("Section detection model is not trained yer!")
			self.model.load_weights(ckptpath)

			# so, stop here, don't train
			return

		# transform X & Y
		x_train = np.array(x_train).reshape((-1, xdim, 1))
		y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)		

		# - Start Training
		history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callback_list)

	def Predict(self, x_valid, y_valid):
		xdim = len(x_valid[0] if x_valid else [])
		x_valid = np.array(x_valid).reshape((-1, xdim, 1))

		# Make predictions for training and test data
		yrpred_valid = self.model.predict(x_valid)
		
		ypred = []
		for yp in yrpred_valid:
			ypred.append(np.argmax(yp))
		
		ypred = np.array(ypred, dtype=np.int)
		
		nprec = np.sum(y_valid==ypred)
		ncorrs = sum(ypred)
		
		return yrpred_valid, nprec, ncorrs
