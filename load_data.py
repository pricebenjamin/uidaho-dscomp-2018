### Open / fetch CSV files

import urllib.request
import shutil
import os

download_url_prefix         = 'https://dscomp.ibest.uidaho.edu/uploads/'
TRAIN_CSV_FILENAME 			= 'train.csv'
TRAIN_LABELS_CSV_FILENAME 	= 'train_labels.csv'
TEST_CSV_FILENAME 			= 'test.csv'
filenames = [
	TRAIN_CSV_FILENAME,
	TRAIN_LABELS_CSV_FILENAME,
	TEST_CSV_FILENAME
]
open_files = {}

## Look for the following folder structure:
# ./images_and_labels
#     |- /digits
#     |    |- digit_images.npy 
#     |    |- digit_labels.npy
#     |- /operators
#     |    |- operator_images.npy
#     |    |- operator_labels.npy
#     |- /mnist
#     |    |- mnist_images.npy
#     |    |- mnist_labels.npy

root = "images_and_labels"
digit_dir = "digits"
operator_dir = "operators"

csv_dir = './csv_files'
if not os.path.exists(csv_dir):
	print('Creating directory `%s`.' % csv_dir)
	os.mkdir(csv_dir)

for filename in filenames:
	# Check if files exist in current directory; if so, open them.
	try: 
		print('Looking for `%s` in `%s`.' % (filename, csv_dir))
		open_file = open(csv_dir + filename, newline='')
	# If files don't exist, downlaod, save, then open them.
	except(FileNotFoundError):
		download_url = download_url_prefix + filename
		print('File not found. Downloading from')
		print('    %s' % download_url)
		# The following code was taken from 
		# https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
		with urllib.request.urlopen(download_url) as response, \
		open(csv_dir + filename, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)
		print('    Done.')
		open_file = open(csv_dir + filename, newline='')
	open_files[filename] = open_file


### Read `train.csv` file into an np.array.

import csv
import numpy as np

train_csv_file = csv.reader(
	open_files[TRAIN_CSV_FILENAME], delimiter=',')
train_images = []
print('Reading data from `%s`.' % TRAIN_CSV_FILENAME)
for i, line in enumerate(train_csv_file):
	if i == 0: continue # First line contains headers.
	if i % 10000 == 0: print('    Reading line %g.' % i)
	image = []
	for j, grey_value in enumerate(line):
		if j == 0: continue # First column contains row number.
		# Convert `grey_value` (string) to a float, then store in `image`.
		image.append(float(grey_value))
	train_images.append(image)

# Convert `train_images` into an np.array with proper dtype.
train_images = np.asarray(train_images, dtype=np.float32)

# Clean up the workspace.
del train_csv_file
del open_files[TRAIN_CSV_FILENAME]

### Read the `train_labels.csv` file into an np.array.

train_labels_csv_file = csv.reader(
	open_files[TRAIN_LABELS_CSV_FILENAME], delimiter=',')
train_labels = []
print('Reading data from `%s`.' % TRAIN_LABELS_CSV_FILENAME)
for i, line in enumerate(train_labels_csv_file):
	if i == 0: continue # First line contains headers.
	for j, label in enumerate(line):
		if j == 0: continue # First column contains row number.
		train_labels.append(int(label))

# Convert `train_labels` into an np.array with proper dtype.
train_labels = np.asarray(train_labels, dtype=np.int32)

# Clean up the workspace.
del train_labels_csv_file
del open_files[TRAIN_LABELS_CSV_FILENAME]


### Separate images and labels of digits from images and labels of operators.

op_labels = [10, 11, 12]
print('Computing operator indices...')
op_indices = [i for i, label in enumerate(train_labels) if label in op_labels]

assert len(train_images) == len(train_labels)

print('Extracting images and labels of operators from the training data...')
op_images = np.asarray([image for i, image in enumerate(train_images) if i in op_indices])
op_labels = np.asarray([label - 10 for i, label in enumerate(train_labels) if i in op_indices])
# We subtract 10 from the operator labels in order to properly compute loss
# when training the operator model. This shifts the labels from [10, 11, 12] to [0, 1, 2].
assert len(op_images) == len(op_labels)

print('Extracting images and labels of digits from the training data...')
digit_images = np.asarray([image for i, image in enumerate(train_images) if i not in op_indices])
digit_labels = np.asarray([label for i, label in enumerate(train_labels) if i not in op_indices])
assert len(digit_images) == len(digit_labels)

del train_images, train_labels, op_indices


### Save (pickle) and unload the data.

DIGIT_IMAGES_FILENAME 		= 'digit_images.npy'
DIGIT_LABELS_FILENAME 		= 'digit_labels.npy'
OP_IMAGES_FILENAME			= 'operator_images.npy'
OP_LABELS_FILENAME			= 'operator_labels.npy'

arrays_to_pickle = [
	(DIGIT_IMAGES_FILENAME, digit_images),
	(DIGIT_LABELS_FILENAME, digit_labels),
	(OP_IMAGES_FILENAME, op_images),
	(OP_LABELS_FILENAME, op_labels)
]

for filename, arr in arrays_to_pickle:
	print('Saving `%s` into current directory.' % filename)
	np.save(filename, arr, allow_pickle=True)

del digit_images, digit_labels, op_images, op_labels, arr

### Read the `test.csv` file into an np.array.

test_csv_file = csv.reader(
	open_files[TEST_CSV_FILENAME], delimiter=',')
test_images = []
print('Reading data from `%s`.' % TEST_CSV_FILENAME)
for i, line in enumerate(test_csv_file):
	if i == 0: continue # First line contains headers.
	if i % 2500 == 0: print('    Reading line %g.' % i)
	image = []
	for j, grey_value in enumerate(line):
		if j == 0: continue # First column contains row number.
		image.append(float(grey_value))
	test_images.append(image)

# Convert `test_images` into an np.array.
test_images = np.asarray(test_images, dtype=np.float32)

TEST_IMAGES_FILENAME = 'test_images.npy'
print('Saving `%s` to current directory.' % TEST_IMAGES_FILENAME)
np.save(TEST_IMAGES_FILENAME, test_images, allow_pickle=True)

# Clean up the workspace.
del test_csv_file
del open_files[TEST_CSV_FILENAME]


print('Splitting each test image into five separate images...')
test_images_as_lists = []
NUM_SUBIMAGES = 5
# Currently, each image is a list of 2880 greyscale values.
# We wish to reshape this list into five subimages where each
# subimages is an np.array of shape (24,24).
for i, image in enumerate(test_images):
	if i % 2500 == 0: print('    Reshaping image %g.' % i)
	# Reshape image into a list of 120 lines of 24 values
	image = np.reshape(image, (120, 24))
	# Split the image into 5 subimages
	subimages = dict()
	for j, line in enumerate(image):
		key = j % NUM_SUBIMAGES
		if key not in subimages:
			# Initialize the subimage.
			subimages[key] = list()
			subimages[key].append(line)
		else:
			# Append the line to its corresponding subimage.
			subimages[key].append(line)
	for key in subimages:
		# Convert the list of lines into an array.
		subimages[key] = np.asarray(subimages[key])
	test_images_as_lists.append(np.asarray([subimages[key] for key in subimages]))

del test_images, subimages

test_images_as_lists = np.asarray(test_images_as_lists)
# test_images_as_lists.shape == (20000, 5, 24, 24)

print('Separating test digit images from operator images.')
test_digits = np.ndarray((60000, 24, 24), dtype=np.float32)
test_op		= np.ndarray((40000, 24, 24), dtype=np.float32)
for i, image_list in enumerate(test_images_as_lists):
	test_digits[3 * i + 0] = image_list[0]
	test_digits[3 * i + 1] = image_list[2]
	test_digits[3 * i + 2] = image_list[4]
	test_op[2 * i + 0] = image_list[1]
	test_op[2 * i + 1] = image_list[3]

del test_images_as_lists, image_list

### Save (pickle) and unload the data.

TEST_DIGITS_FILENAME 	= 'test_digits.npy'
TEST_OPERATORS_FILENAME = 'test_operators.npy'

arrays_to_pickle = [
	(TEST_DIGITS_FILENAME, test_digits),
	(TEST_OPERATORS_FILENAME, test_op)
]

for filename, arr in arrays_to_pickle:
	print('Saving `%s` into current directory.' % filename)
	np.save(filename, arr, allow_pickle=True)

del test_digits, test_op, arr