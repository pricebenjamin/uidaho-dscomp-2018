from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from competition_trainer import model_fn, digit_params, operator_params


def verifyImage(i, digits, ops):
	a, b, c = digits
	ops = tuple(ops)
	if ops == ('+', '='):
		return a + b == c
	elif ops == ('-', '='):
		return a - b == c
	elif ops == ('=', '+'):
		return a == b + c
	elif ops == ('=', '-'):
		return a == b - c
	else:
		print('Verify image %g: Invalid operators %s. Returning `False`.' % (i, ops))
		return False

def main(argv):

	data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

	digit_params['data_format'] = data_format
	digit_classifier 	= tf.estimator.Estimator(
		model_fn=model_fn, model_dir='/tmp/digit_classifier_16', params=digit_params)

	operator_params['data_format'] = data_format
	operator_classifier = tf.estimator.Estimator(
		model_fn=model_fn, model_dir='/tmp/operator_classifier_14', params=operator_params)

	predicted_digits 	= []
	predicted_operators = []
	operator_strings 	= ['+', '-', '=']
	num_tests = 20000

	TEST_DIGITS_FILENAME 	= './images_and_labels/digits/test_digits.npy'
	TEST_OP_FILENAME		= './images_and_labels/operators/test_operators.npy'

	test_digits = np.load(TEST_DIGITS_FILENAME)
	test_op 	= np.load(TEST_OP_FILENAME)

	assert test_digits.shape 	== (3 * num_tests, 24, 24)
	assert test_op.shape 		== (2 * num_tests, 24, 24)

	digit_input_fn = tf.estimator.inputs.numpy_input_fn(
		test_digits, shuffle=False)
	predict = digit_classifier.predict(digit_input_fn)
	for result in predict:
		predicted_digits.append(result['classes'])

	op_input_fn = tf.estimator.inputs.numpy_input_fn(
		test_op, shuffle=False)
	predict = operator_classifier.predict(op_input_fn)
	for result in predict:
		op_str = operator_strings[result['classes']]
		predicted_operators.append(op_str)

	predicted_digits = np.reshape(
		np.array(predicted_digits), (num_tests, 3))
	predicted_operators = np.reshape(
		np.array(predicted_operators), (num_tests, 2))

	predicted_images = \
		list(zip(predicted_digits, predicted_operators))

	import csv
	output_file = open('./submissions/ben_price_dscomp_submission_16.csv', 'w', newline='')
	csv_file = csv.writer(output_file, delimiter=',')
	csv_file.writerow(['index', 'label'])
	for i, (digits, ops) in enumerate(predicted_images):
		csv_file.writerow([i, int(verifyImage(i, digits, ops))])

if __name__ == "__main__":
	tf.app.run()