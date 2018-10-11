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

import model
import utils
import constants


parser = argparse.ArgumentParser(description='CIFAR10 Classifier')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCH_NUM = 0
best_acc = 0
print(torch.cuda.current_device())
transform_train = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_val_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_set, validatation_set = utils.train_val_split(train_val_set)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, sampler=None, shuffle=True, batch_sampler=None)
validation_loader = torch.utils.data.DataLoader(dataset=validatation_set, batch_size=128, sampler=None, shuffle=True, batch_sampler=None)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, sampler=None, shuffle=True, batch_sampler=None)


classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = ResNet34()
# net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.t7')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     EPOCH_NUM = checkpoint['epoch']


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)




# #print(net)

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

#         # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#         #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# def test(epoch):
# 	global best_acc 
# 	net.eval()
# 	test_loss = 0
# 	correct = 0
# 	total = 0
# 	with torch.no_grad():
# 		for batch_idx, (inputs, targets) in enumerate(testloader):
# 			inputs, targets = inputs.to(device), targets.to(device)
# 			outputs = net(inputs)
# 			loss = criterion(outputs, targets)

# 			test_loss += loss.item()
# 			_, predicted = outputs.max(1)
# 			total += targets.size(0)
# 			correct += predicted.eq(targets).sum().item()

# 	print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))		
# 			# progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
# 			# 	% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
# 	acc = 100.*correct/total 
# 	if acc > best_acc:
# 		print('Saving...')
# 		state = {
# 			'net':net.state_dict(),
# 			'acc':acc, 
# 			'epoch':epoch,
# 		}

# 		if not os.path.isdir('checkpoint'):
# 			os.mkdir('checkpoint')
# 		torch.save(state, './checkpoint/ckpt.t7')
# 		best_acc = acc


# if EPOCH_NUM >= 150:
#     EPOCH_MAX = EPOCH_NUM + 100
# else:
#     EPOCH_MAX = EPOCH_NUM + 150

# for epoch in range(EPOCH_NUM, EPOCH_MAX):
# 	train(epoch)
# 	test(epoch)


