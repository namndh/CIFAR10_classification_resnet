import torch 
import torchvision
import pickle 
import constants
import numpy as np

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

train_val_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
train_val_set = list(train_val_set)
train_val_set_sorted = sorted(train_val_set, key=lambda tup:tup[1])
# shuffle(train_val_set_sorted)
train_set = train_val_set_sorted[0:4000] + train_val_set_sorted[5000:9000] + train_val_set_sorted[10000:14000] + train_val_set_sorted[15000:19000]


_, labels = list(zip(*train_val_set))
unique_labels = np.unique(labels)
print(unique_labels)
