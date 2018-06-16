# This script inspired by https://github.com/tensorflow/models/tree/master/official/mnist/mnist.py

import tensorflow as tf
import numpy as np
import os
import sys
from multiprocessing import Pool

# Imports from local directory

from image_manipulators import simardElasticDeformation, composeImages
# Below is some information about the default values of these functions:
#   simardElasticDeformation(image, sigma, alpha, gf_args, mc_args):
#	  sigma = 4, alpha = 24
#	  gf_args = {'order': 0, 'mode': 'reflect', 'cval': 0.0, 'truncate': 4.0}
#	  mc_args = {'order': 1, 'mode': 'constant', 'cval' = 0.0, 'prefilter': False}
#
#   composeImages(image1, image2): # Note that the following values are hard-coded.
#     replacement_threshold = np.random.uniform(0.3, 0.45)
#	  opacity 				= np.random.uniform(0.2, 0.6)

LEARNING_RATE = 1e-4
TRAIN_DIGITS = True
TRAIN_OPERATORS = False # Use submission 14 operator classifier
COMPUTE_PREDICTIONS = True

CHARACTER_TYPES = ['digits', 'operators']

# Most of the network's hyper-parameters are specified in the following dictionaries.
digit_params = {
	'model_dir': '/tmp/digit_classifier_16',
	'eval_index_file': './eval_indexes/digit_eval_indexes.npy',
	'digit_images_file': './images_and_labels/digits/digit_images.npy',
	'digit_labels_file': './images_and_labels/digits/digit_labels.npy',
	'mnist_images_file': './images_and_labels/mnist/mnist_images.npy',
	'mnist_labels_file': './images_and_labels/mnist/mnist_labels.npy',
# Network parameters
	'conv1': (32, 5),
	'conv2': (64, 5),
	'max_pool': (2, 2),
	'fc1': 1024,
	'fc2': 512,
	'fc3': 512,
	'logits': 10,
	'drop': 0.4,
# Training parameters
	'epochs': 600,
	'epochs_between_evals': 10,
	'batch_size': 500,
	'percent_to_deform': 80,
	'percent_to_compose': 30,
	'percent_as_eval': 10}
operator_params = {
	'model_dir': '/tmp/operator_classifier_14',
	'eval_index_file': './eval_indexes/operator_eval_indexes.npy',
	'operator_images_file': './images_and_labels/operators/operator_images.npy',
	'operator_labels_file': './images_and_labels/operators/operator_labels.npy',
# Network parameters
	'conv1': (20, 5),
	'conv2': (40, 5),
	'max_pool': (2, 2),
	'fc1': 100,
	'fc2': None,
	'fc3': None,
	'logits': 3,
	'drop': 0.4,
# Training parameters
	'epochs': 100,
	'epochs_between_evals': 25,
	'batch_size': 500,
	'percent_to_deform': 50,
	'percent_as_eval': 30}

class Model():

	def __init__(self, params):
		"""
		`params`: Dictionary specifying the features of the network. Looks for the following:
			'conv1': 		tuple, 	(num_filters, kernel_size)
			'conv2': 		tuple, 	(num_filters, kernel_size)
			'max_pool': 	tuple, 	(pool_size, strides)
			'fc1': 			int, 	num_units
			'fc2': 			int, 	num_units
			'fc3':			int, 	num_units
			'logits':		int,	num_units
			'drop': 		float, 	rate
			'data_format': 	string, 'channels_first' or 'channels_last'
		"""

		required_params = ['conv1', 'conv2', 'max_pool', 'fc1', 'fc2', 'fc3', 'logits', 'drop', 'data_format']
		check_required_parameters(required_params, params)

		data_format = params['data_format']
		if data_format == 'channels_first':
			# Channels first should improve performance when training on GPU
			self._input_shape = [-1, 1, 24, 24]
		else:
			# Channels last should improve performance when training on CPU
			assert data_format == 'channels_last'
			self._input_shape = [-1, 24, 24, 1]

		self._use_fc2 = (True if params['fc2'] is not None else False)
		self._use_fc3 = (True if params['fc3'] is not None else False)

		self.conv1		= tf.layers.Conv2D(*params['conv1'], name='conv1', padding='same', data_format=data_format, activation=tf.nn.relu)
		self.conv2 		= tf.layers.Conv2D(*params['conv2'], name='conv2', padding='same', data_format=data_format, activation=tf.nn.relu)
		self.max_pool 	= tf.layers.MaxPooling2D(*params['max_pool'], padding='same', data_format=data_format)
		self.fc1 		= tf.layers.Dense(params['fc1'], name='fc1', activation=tf.nn.relu)
		if self._use_fc2: self.fc2 		= tf.layers.Dense(params['fc2'], name='fc2', activation=tf.nn.relu)
		if self._use_fc3: self.fc3		= tf.layers.Dense(params['fc3'], name='fc3', activation=tf.nn.relu)
		self.logits		= tf.layers.Dense(params['logits'], name='logits', activation=tf.nn.relu)
		self.drop 		= tf.layers.Dropout(params['drop'])

	def __call__(self, inputs, training):

		y = tf.reshape(inputs, self._input_shape)
		y = self.conv1(y)
		y = self.max_pool(y)
		y = self.conv2(y)
		y = self.max_pool(y)
		y = tf.layers.flatten(y)
		y = self.fc1(y)
		y = self.drop(y, training=training)
		if self._use_fc2: 
			y = self.fc2(y)
			y = self.drop(y, training=training)
		if self._use_fc3: 
			y = self.fc3(y)
			y = self.drop(y, training=training)

		return self.logits(y)

def model_fn(features, labels, mode, params):

	PREDICT = tf.estimator.ModeKeys.PREDICT
	TRAIN 	= tf.estimator.ModeKeys.TRAIN
	EVAL 	= tf.estimator.ModeKeys.EVAL

	model = Model(params) # Instantiate the model
	images = features

	if mode == PREDICT:
		logits = model(images, training=False) # Call the model
		predictions = {
			'classes': tf.argmax(logits, axis=1),
			'probabilities': tf.nn.softmax(logits)
		}

		return tf.estimator.EstimatorSpec(
			mode=PREDICT,
			predictions=predictions)

	if mode == TRAIN:
		logits = model(images, training=True)
		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

		# Include the batch normalization parameters in the train op
		# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# with tf.control_dependencies(update_ops):
		# 	train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

		return tf.estimator.EstimatorSpec(
			mode=TRAIN,
			loss=loss,
			#train_op=train_op
			train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

	assert mode == EVAL
	logits = model(images, training=False)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	return tf.estimator.EstimatorSpec(
		mode=EVAL, 
		loss=loss,
		eval_metric_ops={
			'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
		})

def train_input_fn(features, labels, character_type, params):
	"""
	`features`: np.array of images
	`labels`: np.array of labels
	`params`: dictionary with keys 'percent_to_compose', 'percent_to_deform', 'batch_size', 'epochs_between_evals'
		Note: composition is only performed if `character_type` == 'digits'
	`character_type`: string, either 'digits' or 'operators'
	"""
	training_images = features
	training_labels = labels

	assert character_type in ('digits', 'operators')
	if character_type == 'digits':
		required_params = ['percent_to_compose', 'percent_to_deform', 'batch_size']
	else: 
		required_params = ['percent_to_deform', 'batch_size']
	check_required_parameters(required_params, params)

	if character_type == 'digits':
		num_images = len(training_images) # alternatively, training_images.shape[0]

		# Compose specified percent of training images.
		percent_to_compose = params['percent_to_compose']
		assert percent_to_compose is None or 0 < percent_to_compose <= 50
		if percent_to_compose is not None:
			num_to_compose = int(percent_to_compose / 100 * num_images)
			random_indexes = np.random.choice(num_images, size=num_to_compose * 2, replace=False)
			# Note: we select twice as many indexes since each composition requires two images.
			print('Composing %g percent of training images with other images in the set.' % percent_to_compose)
			for i, (index_1, index_2) in enumerate(random_indexes.reshape((-1, 2))):
				training_images[index_1] = composeImages(training_images[index_1], training_images[index_2])
			print('Composed a total of %g images.' % i)
	else: # character_type == 'operators'
		num_images = len(training_images)

	# Deform specified percent of training images.
	percent_to_deform = params['percent_to_deform']
	assert percent_to_deform is None or 0 < percent_to_deform <= 100
	if percent_to_deform == 100:
		print('Deforming all images in the training set.')
		# Utilize `multiprocessing.Pool` to speed up deformation.
		with Pool(os.cpu_count()) as p:
			training_images = np.asarray(p.map(simardElasticDeformation, training_images))
		print('Done deforming.')
	elif percent_to_deform is not None:
		num_to_deform = int(percent_to_deform / 100 * num_images)
		random_indexes = np.random.choice(num_images, size=num_to_deform, replace=False)
		print('Deforming %g percent of images in the training set.' % percent_to_deform)
		with Pool(os.cpu_count()) as p:
			training_images[random_indexes] = np.asarray(p.map(simardElasticDeformation, training_images[random_indexes]))
		print('Done deforming.')

	# Pack up the training images and labels into a tf.data.Dataset.
	batch_size = params['batch_size']
	epochs_between_evals = params['epochs_between_evals']
	ds = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
	ds = ds.cache().shuffle(buffer_size=num_images).batch(batch_size)
	ds = ds.repeat(epochs_between_evals)
	return ds

def eval_input_fn(features, labels, batch_size):
	ds = tf.data.Dataset.from_tensor_slices((features, labels))
	ds = ds.batch(batch_size).make_one_shot_iterator().get_next()
	return ds

def train_and_evaluate(classifier, data, params):
	"""
	`classifier`: tf.estimator.Estimator
	`data`: {
		'character_type': str,  # 'digits' or 'operators'
		'train': dict,          #  keys == 'images', 'labels'
		'eval':  dict}          #  keys == 'images', 'labels'
	`params`: {
		...
		'epochs': int,
		'epochs_between_evals': int,
		'batch_size': int,
		...}
	"""
	assert isinstance(classifier, tf.estimator.Estimator)
	character_type = data['character_type']
	assert character_type in CHARACTER_TYPES
	required_params = ['epochs', 'epochs_between_evals', 'batch_size']
	check_required_parameters(required_params, params)
	
	epochs = params['epochs']
	epochs_between_evals = params['epochs_between_evals']
	batch_size = params['batch_size']

	for i in range(epochs // epochs_between_evals):
		print('\nEntering %s training epoch %s.\n' % (character_type, i * epochs_between_evals))
		classifier.train(
			input_fn=lambda: train_input_fn(
				data['train']['images'].copy(), # Ensure that training images are not deformed repeatedly.
				data['train']['labels'], 
				character_type=character_type, 
				params=params))

		eval_results = classifier.evaluate(
			input_fn=lambda: eval_input_fn(
				data['eval']['images'], 
				data['eval']['labels'], 
				batch_size))
		print('\nEvaluation results:\n%s\n' % eval_results)

	return eval_results

def compute_predictions(classifier, images, character_type):
	assert character_type in CHARACTER_TYPES
	assert isinstance(classifier, tf.estimator.Estimator)
	predictions = []
	classifier_predictions = classifier.predict(
		tf.estimator.inputs.numpy_input_fn(images, shuffle=False))
	for i, result in enumerate(classifier_predictions):
		predictions.append(result['probabilities'])
	predictions = np.asarray(predictions)
	np.save('./predictions/' + character_type + '_predictions', predictions) 
	# Load with digit_images when viewing predictions.
	return

def check_required_parameters(required_params, given_params):
	difference = set(required_params) - set(given_params)
	if difference != set():
		raise ValueError('Missing required parameters: {}'.format(difference))
	else: return

def partition_indexes(num_images, percent_as_eval):
	assert 0 < percent_as_eval < 100
	num_as_eval = int(percent_as_eval / 100 * num_images)
	print('Generating list of evaluation indexes.')
	eval_indexes = np.random.choice(num_images, size=num_as_eval, replace=False)
	train_indexes = np.asarray([i for i in range(num_images) if i not in eval_indexes])
	return eval_indexes, train_indexes

def load_or_generate_indexes(num_images, percent_as_eval, filename):
	num_as_eval = int(percent_as_eval / 100 * num_images)
	try:
		eval_indexes = np.load(filename)
		try: assert len(eval_indexes) == num_as_eval
		except AssertionError: 
			print('Expected %s evaluation indexes, but %s were' \
				'found in local file `%s`.' % (num_as_eval, len(eval_indexes), filename))
			raise
		print('Loaded evaluation indexes from local file.')
		train_indexes = np.asarray([i for i in range(num_images) if i not in eval_indexes])
	except(FileNotFoundError):
		eval_indexes, train_indexes = partition_indexes(num_images, percent_as_eval)
		print('Saving evaluation indexes to local file `%s`.' % filename)
		np.save(filename, eval_indexes)

	return eval_indexes, train_indexes

def main(argv):

	# If training on GPU, set data_format to channels_first. Otherwise, 'channels_last'.
	data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

	final_eval_results = {}

	if TRAIN_DIGITS:
		# Load digit classifier.
		digit_params['data_format'] = data_format # Append the data_format at run-time.
		digit_classifier = tf.estimator.Estimator(
			model_dir=digit_params['model_dir'],
			model_fn=model_fn,
			params=digit_params)

		# Load digit data; partition into train/eval sets.
		digit_images = np.load(digit_params['digit_images_file'])
		digit_labels = np.load(digit_params['digit_labels_file'])
		mnist_images = np.load(digit_params['mnist_images_file'])
		mnist_labels = np.load(digit_params['mnist_labels_file'])
		assert len(digit_images) == len(digit_labels)
		assert len(mnist_images) == len(mnist_labels)

		eval_indexes, train_indexes = load_or_generate_indexes(
			len(digit_images), # num_images
			digit_params['percent_as_eval'],
			digit_params['eval_index_file'])

		eval_digit_images = digit_images[eval_indexes]
		eval_digit_labels = digit_labels[eval_indexes]

		# Adjoin resized MNIST images and labels to our training set.
		train_digit_images = np.concatenate((digit_images[train_indexes], mnist_images))
		train_digit_labels = np.concatenate((digit_labels[train_indexes], mnist_labels))

		# Increase the size of the training set by duplication.
		# This allows the network to train with more deformed images in each epoch.
		# train_digit_images = np.concatenate([digit_images[train_indexes]] * 3)
		# train_digit_labels = np.concatenate([digit_labels[train_indexes]] * 3)

		# Train the digit classifier
		digit_data = {
			'character_type': 'digits',
			'train': {'images': train_digit_images, 'labels': train_digit_labels},
			'eval':  {'images': eval_digit_images,  'labels': eval_digit_labels}
		}
		eval_results = train_and_evaluate(digit_classifier, digit_data, digit_params)
		final_eval_results['digits'] = eval_results

		if COMPUTE_PREDICTIONS: compute_predictions(
			digit_classifier, digit_images, digit_data['character_type'])

		# Clean up the workspace.
		del eval_digit_images, train_digit_images, mnist_images, digit_images, digit_classifier

	if TRAIN_OPERATORS:
		# Load operator classifier.
		operator_params['data_format'] = data_format # Append data_format at run-time.
		operator_classifier = tf.estimator.Estimator(
			model_dir=operator_params['model_dir'],
			model_fn=model_fn,
			params=operator_params)

		# Load operator data.
		operator_images = np.load(operator_params['operator_images_file'])
		operator_labels = np.load(operator_params['operator_labels_file'])
		assert len(operator_images) == len(operator_labels)

		eval_indexes, train_indexes = load_or_generate_indexes(
			len(operator_images), # num_images
			operator_params['percent_as_eval'],
			operator_params['eval_index_file'])

		eval_op_images = operator_images[eval_indexes]
		eval_op_labels = operator_labels[eval_indexes]
		train_op_images = operator_images[train_indexes]
		train_op_labels = operator_labels[train_indexes]

		# Train the operator classifier.
		operator_data = {
			'character_type': 'operators',
			'train': {'images': train_op_images, 'labels': train_op_labels},
			'eval':  {'images': eval_op_images,  'labels': eval_op_labels}
		}
		eval_results = train_and_evaluate(operator_classifier, operator_data, operator_params)
		final_eval_results['operators'] = eval_results

		if COMPUTE_PREDICTIONS: compute_predictions(
			operator_classifier, operator_images, operator_data['character_type'])

		# Clean up the workspace.
		del eval_op_images, train_op_images, operator_images, operator_classifier

	for key in final_eval_results:
		print('\n%s results:\n%s\n' % (key, final_eval_results[key]))

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	main(argv=sys.argv) # Currently not using commandline arguments.
