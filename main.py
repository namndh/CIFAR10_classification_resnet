import os
import argparse
import torch 
import torchvision 
import torchvision.transforms as transforms 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data
import time
import pickle 

import models
import utils
import constants

parser = argparse.ArgumentParser(description='CIFAR10 Classifier')
parser.add_argument('--init_lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--predict', action='store_true', help='predict input images')
parser.add_argument('--predict_option', default=0, type=int, choices=[0, 1], help='0: predict with best acc model on validate set -- 1: predict with convergence model on train set')
parser.add_argument('--inspect', action='store_true', help='inspect the model')
parser.add_argument('--depth', default=18, choices = [18, 34, 50, 101, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
parser.add_argument('--optim', default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=250, type=int, help='Number of epochs in training')
parser.add_argument('--drop_out', default=0.5, type=float)
parser.add_argument('--check_after', default=1, type=int, help='Validate the model after how many epoch')
parser.add_argument('--train_ratio', default=0.8, type=float, help='ration of train and validate set')
args = parser.parse_args()
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Available device is:{}'.format(device))
print(torch.cuda.current_device)

start_epoch = 0
save_acc = 0
save_loss = 0

network_depth = [18, 34, 50, 101, 152]

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def exp_lr_schedule(args, optimizer, epoch):
	init_lr = args.init_lr
	lr_decay_epoch = 50
	weight_decay = args.weight_decay
	lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		param_group['weight_decay'] = weight_decay

	return optimizer, lr

def save_convergence_model(save_loss, epoch_loss, model, epoch):
	save_loss = epoch_loss
	print('Saving convergence model at epoch {} with loss {}'.format(epoch, epoch_loss))
	state = {
		'model' : model.state_dict(),
		'loss'	: save_loss,
		'epoch'	: epoch
	}
	if not os.path.isdir('./checkpoint'):
		os.path.mkdir('./checkpoint')
	torch.save(state, './checkpoint/convergence.t7')

def save_best_acc_model(save_acc, model, epoch):
	print('Saving best acc model at epoch {} with acc in validation set: {}'.format(epoch, save_acc))
	state = {
		'model'	: model.state_dict(),
		'acc'	: save_acc,
		'epoch' : epoch,
	}

	if not os.path.isdir('./checkpoint'):
		os.path.mkdir('./checkpoint')
	torch.save(state, './checkpoint/best_acc_model.t7')

def train_validate(epoch, optimizer, model, criterion, train_loader, validate_loader):
	optimizer, lr = exp_lr_schedule(args, optimizer, epoch)
	print("================================================\n")
	print('=> Training in epoch {} at LR {:.3}'.format(epoch + 1, lr))
	model.train()
	train_loss = 0
	train_correct = 0
	total = 0

	for idx, (images, labels) in enumerate(train_loader):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)

		total += labels.size(0)
		train_correct += predicted.eq(labels).sum().item()

	epoch_loss = train_loss/(idx+1)
	epoch_correct = train_correct/total

	global save_loss
	if epoch == 0:
		save_convergence_model(save_loss, epoch_loss, model, epoch)
	else:
		if epoch_loss < save_loss:
			save_convergence_model(save_loss, epoch_loss, model, epoch)

	print('Loss : {} || Correct : {}'.format(epoch_loss, epoch_correct))


	if (epoch + 1) % args.check_after == 0:
		print('==============================================\n')
		print('==> Validate in epoch {} at LR {:.3}'.format(epoch + 1, lr))
		model.eval()
		validate_loss = 0
		validate_acc = 0
		validate_correct = 0
		total = 0
		for idx, (images, labels) in enumerate(validate_loader):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)

			validate_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			validate_correct += predicted.eq(labels).sum().item()

		validate_acc = validate_correct/total * 100
		print('Accuracy : {}%'.format(validate_acc))

		global save_acc
		if validate_acc > save_acc:
			save_best_acc_model(save_acc, validate_acc, model, epoch)

def predict(model, test_loader, convergence):
	assert os.path.isdir('./checkpoint'), 'Error: model is not availabel!'
	if convergence:
		checkpoint = torch.load('./checkpoint/convergence.t7')
		model.load_state_dict(checkpoint['model'])
		loss = checkpoint['loss']
		epoch = checkpoint['epoch']
		print('Model used to predict converges at epoch {} and loss {:.3}'.format(epoch, loss))
	else:
		checkpoint = torch.load('./checkpoint/best_acc_model.t7')
		model.load_state_dict(checkpoint['model'])
		acc = checkpoint['acc']
		epoch = checkpoint['epoch']		
		print('Model used to predict has best acc {:.3}% on validate set at epoch {}'.format(acc, epoch))

	torch.set_grad_enabled(False)
	model.eval()
	c= 0
	test_correct = 0
	total = 0
	class_correct = dict()
	class_total = dict()
	for _class in classes:
		class_correct[_class] = 0
		class_total[_class] = 0

	for idx, (images, labels) in enumerate(test_loader):
		images, labels = images.to(device), labels.to(device)
		# print(len(labels))
		outputs = model(images)

		_, predicted = outputs.max(1)
		test_correct += predicted.eq(labels).sum().item()
		total += labels.size(0)
		c = (predicted == labels).squeeze()
		for i in range(len(labels)):
			label = labels[i]
			class_correct[classes[label]] += c[i].item()
			class_total[classes[label]] += 1

	print('Accuracy of model in predicting: {}%'.format(test_correct/total*100))

	for i in range(len(classes)):
		print('Accuracy of {} : {:.3}%'.format(classes[i], 100*class_correct[classes[i]]/class_total[classes[i]]))


if args.depth in network_depth:
	if args.depth == 18:
		model = models.myResNet18()
		model.to(device)
	if args.depth == 34:
		model = models.myResNet34()
		model.to(device)
	if args.depth == 50:
		model = models.myResNet50()
		model.to(device)
	if args.depth == 101:
		model = models.myResNet101()
		model.to(device)
	if args.depth == 152:
		model = models.myResNet152()
		model.to(device)
criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
	optimizer = optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, momentum=0.9)
elif args.optim == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)



if args.train:
	# print(args)

	train_set = utils.myDataSet(train=True, root_dir=constants.DATA_PATH, transform=transform_train)
	val_set	= utils.myDataSet(train=False, root_dir=constants.DATA_PATH, transform=transform_test)


	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, batch_sampler=None)
	val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, batch_sampler=None)
	for epoch in range(start_epoch + args.num_epochs):
		train_validate(epoch, optimizer, model, criterion, train_loader, val_loader)

if args.predict:
	test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, batch_sampler=None)
	if args.predict_option == 0:
		conv = False
	else:
		conv = True
	predict(model, test_loader, conv)

if args.inspect:

	checkpoint = torch.load('./checkpoint/convergence.t7')
	loss = checkpoint['loss']
	epoch = checkpoint['epoch']
	print('Model used to predict converges at epoch {} and loss {:.3}'.format(epoch, loss))

	checkpoint = torch.load('./checkpoint/best_acc_model.t7')
	acc = checkpoint['acc']
	epoch = checkpoint['epoch']		
	print('Model used to predict has best acc {:3} on validate set at epoch {}'.format(acc, epoch))

# if args.resume:
# 	print('==> Resume model from last checkpoint ...')
# 	assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
# 	checkpoint = torch.load