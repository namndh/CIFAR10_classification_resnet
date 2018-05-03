import torch 
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import time 
from models import Net
import constants

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='/home/t3min4l/workspace/pytorch-tutorial/data', train=True, download=True, transform=transformer)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=constants.BATCH_SIZE , sampler=None, shuffle=True, batch_sampler=None)

testset = torchvision.datasets.CIFAR10(root='/home/t3min4l/workspace/pytorch-tutorial/data', train=False, download=True, transform=transformer)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, batch_sampler=None)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

padding = nn.ConstantPad2d((3,3,3,3), 0)

for i, data in enumerate(trainloader, 0):
    images, labels = data
    print(labels)
    if i > 1:
        break

def gpu_train(trainloader):
    models = Net()
    models.cuda()
    print(models)
    criterion_CEL = nn.CrossEntropyLoss()
    optimizer_sgd = optim.SGD(models.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    for epoch in range(constants.EPOCHS):
        for iteration in range(constants.ITERATIONS):
            running_loss = 0 
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                images = padding(images)
                if (images.shape[2], images.shape[3]) == (38,38):
                    images, labels = images.cuda(), Variable(labels.cuda())

                    optimizer_sgd.zero_grad()
                    outputs = models(images)
                    loss = criterion_CEL(outputs, labels)
                    loss.backward()
                    optimizer_sgd.step()

                    running_loss += loss.data[0]
                    if i % 5000 == 4999:
                        print("iteration {} in batch {} in epoch {} loss:{}".format(i+1, iteration+1, epoch+1, running_loss/5000))
                else:
                    continue
    train_time = time.time() - start_time
    start_time = time.time()

    models.eval()
    correct = 0
    total = 0 
    for data in testloader:
        images, labels = data
        images = padding(images)
        outputs = models(images.cuda())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy over the network on the 10k test images training on GPU(1060 6gb):{}%'.format(100*correct/total))

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for data in testloader:
        images, labels = data
        images = padding(images)
        outputs = models(images.cuda())

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted.cpu() == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    elapsed_time = time.time() - start_time
    print("GPU Training:\n")
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print("Elapsed time with GPU trained network:{} and evaluating time is:{}".format(train_time, elapsed_time))

    
def cpu_train(trainloader):
    models = Net()
    print(models)
    criterion_CEL = nn.CrossEntropyLoss()
    optimizer_sgd = optim.SGD(models.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    for epoch in range(constants.EPOCHS):
        for iteration in range(constants.ITERATIONS):
            running_loss = 0 
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                images = padding(images)
                images, labels = images, Variable(labels)

                optimizer_sgd.zero_grad()
                outputs = models(images)
                if i > 9998:
                    print(labels)
                    print(outputs)
                loss = criterion_CEL(outputs, labels)
                loss.backward()
                optimizer_sgd.step()

                running_loss += loss.data[0]
                if i % 5000:
                    print("iteration {} in batch {} in epoch {} loss:{}".format(i+1, iteration+1, epoch+1, running_loss/5000))
    train_time = time.time() - start_time
    start_time = time.time()

    models.eval()
    correct = 0
    total = 0 
    for data in testloader:
        images, labels = data
        outputs = models(Variable(images))

        _, predicted = torch.max()
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy over the network on the 10k test images training on GPU(1060 6gb):{}%'.format(100*correct/total))

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for data in testloader:
        images, labels = data
        outputs = models(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
            
    elapsed_time = time.time() - start_time
    print("GPU Training:\n")
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print("Elapsed time with GPU trained network:{} and evaluating time is:{}".format(train_time, elapsed_time))


# gpu_train(trainloader)
cpu_train(trainloader)
