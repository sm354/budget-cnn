import argparse
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt

from models.ResNet import *
from utils import format_time
import os
import subprocess

class my_dataset(Dataset):
	def __init__(self, x, y, transform):
		super(my_dataset,self).__init__()
		self.t=transform
		self.x=x
		self.y=y
	def __getitem__(self, index):
		x,y=self.x[index],self.y[index]
		x=self.t(x)
		return x,y
	def __len__(self):
		return self.x.shape[0]

def get_cifar10(data_dir):
	t = transforms.Compose([transforms.ToTensor(),])
	ts = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=t)
	loader = DataLoader(ts, batch_size=50000, shuffle=False, num_workers=2)
	x_train, y_train = next(iter(loader))
	assert x_train.shape==(50000,3,32,32)
	assert y_train.shape==(50000,)

	ts = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=t)
	loader = DataLoader(ts, batch_size=10000, shuffle=False, num_workers=2)
	x_test, y_test = next(iter(loader))
	assert x_test.shape==(10000,3,32,32)
	assert y_test.shape==(10000,)
	return (x_train,y_train,x_test,y_test)

def get_cifar10_sets(data, sizes):
	# Normalisation reference : https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
	train_set, test_set = [],[]

	train_transform = [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)), ]
	test_transform = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)), ]
	if sizes==None:
		train_set.append(my_dataset(data[0],data[1],transforms.Compose(train_transform)))
		test_set.append(my_dataset(data[2],data[3],transforms.Compose(test_transform)))
	else:
		for s in sizes:
			dummy_l1=train_transform[0:1]+[transforms.Resize(size=(s,s)),transforms.RandomCrop(s,padding=s//8)]+train_transform[2:]
			dummy_l2=test_transform[0:1]+[transforms.Resize(size=(s,s))]+test_transform[1:]
			print(dummy_l1)
			print(dummy_l2)
			train_set.append(my_dataset(data[0],data[1],transforms.Compose(dummy_l1)))
			test_set.append(my_dataset(data[2],data[3],transforms.Compose(dummy_l2)))
			
		assert len(train_set)==len(sizes)
		assert len(test_set)==len(sizes)
	return train_set, test_set

def get_cifar100(data_dir):
	t = transforms.Compose([transforms.ToTensor(),])
	ts = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=t)
	loader = DataLoader(ts, batch_size=50000, shuffle=False, num_workers=2)
	x_train, y_train = next(iter(loader))
	assert x_train.shape==(50000,3,32,32)
	assert y_train.shape==(50000,)

	ts = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=t)
	loader = DataLoader(ts, batch_size=10000, shuffle=False, num_workers=2)
	x_test, y_test = next(iter(loader))
	assert x_test.shape==(10000,3,32,32)
	assert y_test.shape==(10000,)
	return (x_train,y_train,x_test,y_test)

def get_cifar100_sets(data, sizes):
	train_set, test_set = [],[]

	train_transform = [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)), ]
	test_transform = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)), ]
	if sizes==None:
		train_set.append(my_dataset(data[0],data[1],transforms.Compose(train_transform)))
		test_set.append(my_dataset(data[2],data[3],transforms.Compose(test_transform)))
	else:
		for s in sizes:
			dummy_l1=train_transform[0:1]+[transforms.Resize(size=(s,s)),transforms.RandomCrop(s,padding=s//8)]+train_transform[2:]
			dummy_l2=test_transform[0:1]+[transforms.Resize(size=(s,s))]+test_transform[1:]
			train_set.append(my_dataset(data[0],data[1],transforms.Compose(dummy_l1)))
			test_set.append(my_dataset(data[2],data[3],transforms.Compose(dummy_l2)))
			
		assert len(train_set)==len(sizes)
		assert len(test_set)==len(sizes)
	return train_set, test_set

def get_fashionmnist(data_dir):
	t = transforms.Compose([transforms.ToTensor(),])
	ts = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=t)
	loader = DataLoader(ts, batch_size=60000, shuffle=False, num_workers=2)
	x_train, y_train = next(iter(loader))
	assert x_train.shape==(60000,1,28,28)
	assert y_train.shape==(60000,)

	ts = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=t)
	loader = DataLoader(ts, batch_size=10000, shuffle=False, num_workers=2)
	x_test, y_test = next(iter(loader))
	assert x_test.shape==(10000,1,28,28)
	assert y_test.shape==(10000,)
	return (x_train,y_train,x_test,y_test)

def get_fashionmnist_sets(data, sizes):
	train_set, test_set = [],[]

	train_transform = [transforms.ToPILImage(), transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,),(1.,)), ]
	test_transform = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.1307,),(1.,)), ]
	if sizes==None:
		train_set.append(my_dataset(data[0],data[1],transforms.Compose(train_transform)))
		test_set.append(my_dataset(data[2],data[3],transforms.Compose(test_transform)))
	else:
		for s in sizes:
			dummy_l1=train_transform[0:1]+[transforms.Resize(size=(s,s)),transforms.RandomCrop(s,padding=s//8)]+train_transform[2:]
			dummy_l2=test_transform[0:1]+[transforms.Resize(size=(s,s))]+test_transform[1:]
			train_set.append(my_dataset(data[0],data[1],transforms.Compose(dummy_l1)))
			test_set.append(my_dataset(data[2],data[3],transforms.Compose(dummy_l2)))
			
		assert len(train_set)==len(sizes)
		assert len(test_set)==len(sizes)
	return train_set, test_set

def get_tinyImageNet(data_dir):
    if not os.path.exists(f"{data_dir}/train_x.npy"):
        subprocess.run(["bash", f"{data_dir}/run.sh", data_dir])
    x_train, y_train, x_test, y_test = np.load(f"{data_dir}/train_x.npy"), np.load(f"{data_dir}/train_y.npy"), np.load(f"{data_dir}/test_x.npy"), np.load(f"{data_dir}/test_y.npy")
    x_train, y_train, x_test, y_test = torch.from_numpy(x_train.transpose(0,3,1,2)), torch.from_numpy(y_train), torch.from_numpy(x_test.transpose(0,3,1,2)), torch.from_numpy(y_test)
    y_train, y_test = y_train.long(), y_test.long()
    return (x_train,y_train,x_test,y_test)

def get_tinyImageNet_sets(data, sizes):
    train_set, test_set = [],[]

    train_transform = [transforms.ToPILImage(), transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485,),(1.,)), ]
    test_transform = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.485,),(1.,)), ]
    if sizes==None:
        train_set.append(my_dataset(data[0],data[1],transforms.Compose(train_transform)))
        test_set.append(my_dataset(data[2],data[3],transforms.Compose(test_transform)))
    else:
        for s in sizes:
            dummy_l1=train_transform[0:1]+[transforms.Resize(size=(s,s)),transforms.RandomCrop(s,padding=s//8)]+train_transform[2:]
            dummy_l2=test_transform[0:1]+[transforms.Resize(size=(s,s))]+test_transform[1:]
            train_set.append(my_dataset(data[0],data[1],transforms.Compose(dummy_l1)))
            test_set.append(my_dataset(data[2],data[3],transforms.Compose(dummy_l2)))
            
        assert len(train_set)==len(sizes)
        assert len(test_set)==len(sizes)
    return train_set, test_set
def train_model(model, train_loader, device, loss_fn, optimizer):
	model.train()
	l=0
	total_samples, correct_predictions = 0, 0
	for _, (X,Y) in enumerate(train_loader):
		X,Y=X.to(device),Y.to(device)
		optimizer.zero_grad() # remove history
		Y_ = model(X)
		Y_predicted = Y_.argmax(dim=1)
		loss = loss_fn(Y_, Y)
		loss.backward() # create computational graph i.e. find gradients
		optimizer.step() # update weights/biases
		l+=loss.item()
		correct_predictions += Y_predicted.eq(Y).sum().item()
		total_samples += Y_predicted.size(0)
	a=(correct_predictions/total_samples)*100. 
	l/=_
	return (l,a)

def test_model(model, test_loader, device, loss_fn):
	model.eval()
	l=0
	total_samples, correct_predictions = 0, 0
	with torch.no_grad():
		for _, (X,Y) in enumerate(test_loader):
			X=X.to(device)
			Y=Y.to(device)
			Y_ = model(X)
			Y_predicted = Y_.argmax(dim=1)
			loss = loss_fn(Y_, Y)
			l+=loss.item()
			correct_predictions += Y_predicted.eq(Y).sum().item()
			total_samples += Y_predicted.size(0)
	a=(correct_predictions/total_samples)*100. 
	l/=_
	return (l,a)

if __name__ == "__main__":
	# torch.manual_seed(0)

	parser = argparse.ArgumentParser(description='Training Pre-Activation ResNets')
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--data_dir', default='./', type=str)
	parser.add_argument('--n', default=2, help='number of (per) residual blocks', type=int)
	parser.add_argument('--exp_name', default=False, help='experiment name')
	parser.add_argument('--algo', default=False, help='no argument means no algorithmic training')
	parser.add_argument('--save', default=True, help='give argument if saves are not to be made')
	parser.add_argument('--backup_save_here', type=str, help='save models and numpy stats here')
	parser.add_argument('--save_plots_here', default=False, help='save final learning plots here')
	parser.add_argument('--dataset', type=str)
	args=parser.parse_args()


	device='cuda:0' if torch.cuda.is_available() else 'cpu'
	print("####---- GPU Details ----####\n",torch.cuda.get_device_properties(0),"\n\n")


	####---- ResNet ----####
	io=64 if args.algo==False else 16
	img_c=1 if args.dataset=='fashionMNIST' else 3
	if args.dataset=='cifar100':
		num_c=100
	elif args.dataset=='tinyimagenet':
		num_c=200
	else:
		num_c=10
	model=ResNet(n=args.n, r=num_c, io=io, img_channels=img_c)
	model=model.to(device)
	print("####---- Model Loaded | Layers = "+str(4*2*args.n+2)+"----####\n","\n\n")


	####---- DataSet ----####
	'''
	Get X_train, Y_train, X_test, Y_test as tensors (in 0-1) without any data augmentation i.e. as original
	Get train_set, test_set using them : when algorithmic training these will be lists

	Then give these to data loaders (after giving creating dataset) which will handle augmentation, and algorithmic training

	'''
	steps = [int(0.33*args.num_epochs), int(0.67*args.num_epochs)] if args.algo!=False else None
	if args.algo!=False:
		if args.dataset=='fashionMNIST':
			sizes=[7,14,28]
		elif ((args.dataset=='cifar10') or (args.dataset=='cifar100')):
			sizes=[8,16,32]
		elif args.dataset=='tinyimagenet':
			sizes=[16,32,64]
	else:
		sizes=None

	if args.dataset=='cifar10':
		data = get_cifar10(data_dir=args.data_dir) # data = (X_train, Y_train, X_test, Y_test)
		train_set, test_set = get_cifar10_sets(data=data, sizes=sizes)
	elif args.dataset=='cifar100':
		data = get_cifar100(data_dir=args.data_dir) # data = (X_train, Y_train, X_test, Y_test)
		train_set, test_set = get_cifar100_sets(data=data, sizes=sizes)
	elif args.dataset=='fashionMNIST':
		data = get_fashionmnist(data_dir=args.data_dir) # data = (X_train, Y_train, X_test, Y_test)
		train_set, test_set = get_fashionmnist_sets(data=data, sizes=sizes)
	elif args.dataset=='tinyimagenet':
		data = get_tinyImageNet(data_dir=args.data_dir) # data = (X_train, Y_train, X_test, Y_test)
		train_set, test_set = get_tinyImageNet_sets(data=data, sizes=sizes)
	
	print("####---- %s Dataset Prepared ----####\n"%args.dataset,"\n\n")


	####---- Begin Training ----####
	train_loader = DataLoader(train_set.pop(0), batch_size=128, shuffle=True, num_workers=2)
	test_loader = DataLoader(test_set.pop(0), batch_size=128, shuffle=False, num_workers=2)
	loss_fn = nn.CrossEntropyLoss()
	# optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

	print("####---- Beginning Training ----####\n")
	print('number of model parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
	print('train image size:', next(iter(train_loader))[0].shape, '| test image size', next(iter(test_loader))[0].shape)
	train_acc, test_acc = [],[]
	exp_time = 0
	for ep in range(args.num_epochs):
		if( (args.algo!=False) and (ep in steps) ):
			print("####---- At epoch %u Stepping up the Images and Model ----####\n"%(ep))
			# save the smaller models also (to later visualize)
			model_name = args.exp_name + '_Ours_%u'%(steps.index(ep))
			if args.save==True:
				torch.save(model.state_dict(),args.backup_save_here+model_name+'.pth')

			# change data loader and step_up model
			train_loader = DataLoader(train_set.pop(0), batch_size=128, shuffle=True, num_workers=2)
			test_loader = DataLoader(test_set.pop(0), batch_size=128, shuffle=False, num_workers=2)
			model.Step_ResNet()
			model=model.to(device)

			print('number of model parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
			print('train image size:', next(iter(train_loader))[0].shape, '| test image size:x', next(iter(test_loader))[0].shape)

			# optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
			optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

		if( args.num_epochs-7>0 and ep==args.num_epochs-7 ):
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
		if( args.num_epochs-2>0 and ep==args.num_epochs-2 ):
			optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

		s = time.time()
		l1, a = train_model(model, train_loader, device, loss_fn, optimizer)
		e = time.time()
		exp_time+=(e-s)
		train_acc.append(a)

		s = time.time()
		l2, a = test_model(model, test_loader, device, loss_fn)
		e = time.time()
		exp_time+=(e-s)
		test_acc.append(a)

		print("Epoch",ep,"trainAcc",train_acc[-1],"testAcc",test_acc[-1],"losses",l1,l2)

	# save the model and the accuracy lists, plots
	print("####---- Training Completed; Saving STATS ----####\n")
	print('TIME TAKEN:',exp_time, format_time(exp_time))
	name_save = args.exp_name + '_Ours' if args.algo!=False else args.exp_name
	if args.save==True:
		torch.save(model.state_dict(),args.backup_save_here+name_save+'_final.pth')
	np.save(args.backup_save_here+name_save+'_accs.npy', np.array([train_acc,test_acc]))

	plt.figure()
	plt.title(name_save+'| Acc=%.2f'%(test_acc[-1])+'| time=%s'%(format_time(exp_time)))
	plt.xlabel('epochs')
	plt.plot(train_acc,label='train_acc')
	plt.plot(test_acc,label='test_acc')
	if steps is not None:
		for x in steps:
			plt.axvline(x=x, ls='-.', c='orange', linewidth=0.9)
	plt.grid()
	plt.legend()
	# if args.save==True:
	plt.savefig(args.save_plots_here+name_save+'.png')
	
