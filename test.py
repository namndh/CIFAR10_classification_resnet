import torch 
import torchvision
import pickle 
import constants
import numpy as np
from collections import defaultdict
from random import shuffle

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def diff(first, second):
	# second = set(second)
	return [item for item in first if item not in second]

labels = [1,4,7,9,10,3,2,7,8,9]
label_correct = dict()
label_total = dict()
for _class in classes:
	label_correct[_class] = 0
for label in labels:
	label_correct[classes[label-1]] += 1

print(label_correct)

data = ['a','b','c','d']
labels = [1,2,3,4]
print(zip(data, labels))

test_dataset = [('abc', 1), ('bcd', 2)]
strings, ints = zip(*test_dataset)
strings = list(strings)
ints = list(ints)
print('{}.{}'.format(type(strings), type(ints)))
print(strings)


train_val_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
train_val_set = list(train_val_set)
train_val_set_sorted = sorted(train_val_set, key=lambda tup:tup[1])
# _, labels = list(zip(*train_val_set))
# unique_labels = np.unique(labels)
# labels_and_num = dict()

# print(unique_labels)

# # shuffle(train_val_set_sorted)
train_set = train_val_set_sorted[0:4000] + train_val_set_sorted[5000:9000] + train_val_set_sorted[10000:14000] + train_val_set_sorted[15000:19000] + train_val_set_sorted[20000:24000] + train_val_set_sorted[25000:29000] + train_val_set_sorted[30000:34000] + train_val_set_sorted[35000:39000] + train_val_set_sorted[40000:44000] + train_val_set_sorted[45000:49000]
val_set = train_val_set_sorted[4000:5000] + train_val_set_sorted[9000:10000] + train_val_set_sorted[14000:15000] + train_val_set_sorted[19000:20000] + train_val_set_sorted[24000:25000] + train_val_set_sorted[29000:30000] + train_val_set_sorted[34000:35000] + train_val_set_sorted[39000:40000] + train_val_set_sorted[44000:45000] + train_val_set_sorted[49000:50000]
shuffle(train_set)
shuffle(val_set)
with open(constants.TRAIN_SET, 'wb') as train_set_bin:
	pickle.dump(train_set, train_set_bin)
with open(constants.VAL_SET, 'wb') as val_set_bin:
	pickle.dump(val_set, val_set_bin)
print(len(val_set))
print(len(train_set))


