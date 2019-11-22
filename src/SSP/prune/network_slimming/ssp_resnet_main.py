from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import ray
import resnet.models as models
import random,time
from time import sleep
import copy 
import datetime
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from filelock import FileLock


def generate_train_loader(batch_size,kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
        transform=transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
    batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

def generate_test_loader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),batch_size=test_batch_size, shuffle=True)
    return test_loader

@ray.remote
class ParameterServer():
    def __init__(self,args,test_loader):
        self.model = models.__dict__["resnet"](dataset="cifar10",depth=args.depth)
        self.stalness_table = [0] * args.num_workers
        self.stalness_limit = args.stalness_limit 
        self.global_step = 0
        self.lr = args.lr
        self.eva_model = models.__dict__["resnet"](dataset="cifar10",depth=args.depth)
        self.optimizer = optim.SGD(self.model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
        self.test_loader = test_loader
        self.model.cpu()
        self.eva_model.cpu()
        self.ps_writer = SummaryWriter(args.tb_path+'/ps')
        self.save_path = args.save
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(args.resume, checkpoint['epoch'], best_prec1))                
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def apply_gradients(self, gradients, wk_idx):
        print("applying gradients from the ",wk_idx, " worker")
        for idx, p in enumerate(self.model.parameters()):
            p.data -= self.lr * gradients[idx]
        self.stalness_table[wk_idx] += 1
        self.global_step += 1
        print("finished applying gradients from the ",wk_idx, " worker")
        if self.global_step % 100 == 0:
            print("global_step: ",self.global_step," and prepare evaluate")
            self.evaluate()
            self.save_ckpt({
                'global_step':global_step,
                'state_dict':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict()
            },filepath=self.save_path)
            
        
    def pull_weights(self):
        return self.model.state_dict()
    
    def pull_optimizer_state(self):
        return self.optimizer.state_dict()

    def check_stalness(self,wk_idx):
        min_iter = min(self.stalness_table)
        return self.stalness_table[wk_idx] - min_iter < self.stalness_limit
        
    def get_stalness(self):
        return min(self.stalness_table)
    
    def save_ckpt(self,state,filepath):
        torch.save(state,os.path.join(filepath,'checkpoint.pth.tar'))
        
    def evaluate(self):
        print("going to evaluate")
        test_loss = 0.
        correct = 0.
        print("pulled weights")
        self.eva_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        print("loaded weights")
        print("length of the test_loader dataset is : ",len(self.test_loader.dataset))
        self.eva_model.eval()
        for data,target in self.test_loader:
            data, target = Variable(data),Variable(target)
            output = self.eva_model(data)
            batch_loss = F.cross_entropy(output, target, size_average=False).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        len_testset = len(self.test_loader.dataset)
        test_loss /= len_testset 
        accuracy = correct / len(self.test_loader.dataset)
        # log 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len_testset,accuracy))

        self.ps_writer.add_scalar('Accuracy/eval', accuracy, self.global_step)
        self.ps_writer.add_scalar('Loss/eval',test_loss , self.global_step)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, 
            correct, 
            len(data),
            100. * correct / len(data)))

@ray.remote(num_gpus=1)
def worker_task(args,ps,worker_index, train_loader):
    # Initialize the model.
    if args.debug: print(worker_index, " worker is going to sleep ",worker_index*5000)
    time.sleep(worker_index * 5000)
    
    model = models.__dict__["resnet"](dataset="cifar10",depth=args.depth)
    local_step = 0
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.cuda:
        starttime = datetime.datetime.now()
        model.cuda()
        endtime = datetime.datetime.now()
        time_cost = (endtime - starttime).seconds
        if args.debug: print("move model to gpu takes: ", time_cost, "seconds")
    if args.resume:
        checkpoint = torch.load(args.resume)
        local_step = checkpoint['step'] / 2
        optimizer.load_state_dict(checkpoint['optimizer'])


    wk_writer = SummaryWriter(os.path.join(args.tb_path,'wk_',str(worker_index)))
    
    for epoch in range(args.start_epoch,args.epochs):
        avg_loss = 0.
        train_acc = 0.
        for batch_idx,(data,target) in enumerate(train_loader):
            if args.cuda:
                starttime = datetime.datetime.now()
                data,target = data.cuda(),target.cuda()
                mid = datetime.datetime.now()
                if args.debug: print("move data to gpu takes: ", (mid - starttime).seconds, "seconds")
                model.cuda()
                endtime = datetime.datetime.now()
                time_cost = (endtime - starttime).seconds
                if args.debug: print("move model to gpu takes: ", time_cost, "seconds")
                
            while(local_step - ray.get(ps.get_stalness.remote()) > args.stalness_limit):
                print(worker_index," works too fast")
                sleep(1)
            # Get the current weights from the parameter server.
            if args.debug: print("the ",worker_index," pulls wei from ps.")
            init_wei = ray.get(ps.pull_weights.remote())
            model.load_state_dict(init_wei)
            if args.debug: print("the ",worker_index," loaded the latest wei from ps.")
            # Compute an update and push it to the parameter server.        
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            if args.debug: print(worker_index,' is generating output')
            output = model(data)
            if args.debug: print(worker_index,' generated output done and going to calculate loss')
            loss = F.cross_entropy(output,target)
            avg_loss += loss
            pred = output.data.max(1,keepdim=True)[1]
            batch_acc = pred.eq(target.data.view_as(pred)).cpu().sum()
            train_acc += batch_acc
            if args.debug: print(worker_index,' calculated loss and going to bp')
            loss.backward()
            if args.debug: print(worker_index,' bp done')
            # starttime = datetime.datetime.now()
            model.cpu()
            # endtime = datetime.datetime.now()
            # time_cost = (endtime - starttime).seconds
            # print("move model to cpu takes: ", time_cost, "seconds")
            grad = [p.grad for p in model.parameters()]
            if args.debug: print(worker_index,' got the grad list')
            local_step += 1
            ps.apply_gradients.remote(grad,worker_index)
            if args.debug: print(worker_index,' sended the grad to ps and going to move next step')
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('The {} worker, Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                worker_index, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
                wk_writer.add_scalar("Loss/worker_train",loss,local_step)
                wk_writer.add_scalar("Accuracy/worker_train",batch_acc,local_step)
        print("The {} worker finished its {} epoch with loss: {} and accuracy: {}".format(
            worker_index,
            epoch,
            avg_loss / len(train_loader),
            train_acc / float(len(train_loader)
        )))

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()

    parser = argparse.ArgumentParser(description='Distributed SSP CIFAR-10 Restnet train with network slimming')
    parser.add_argument('--ray-master',type=str,default='127.0.0.1')
    parser.add_argument('--redis-port',type=str,default='6379')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--save', default='./logs', type=str)
    parser.add_argument('--depth', default=164, type=int)
    parser.add_argument('--tb-path', default='./logs', type=str)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--num-workers',type=int,default=1)
    parser.add_argument('--stalness-limit',type=int,default=5)
    parser.add_argument('--debug',action='store_true',default=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args.ray_master)
    ray.init(address=args.ray_master+':'+args.redis_port)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


    test_loader = generate_test_loader(args.test_batch_size)
    train_loaders = [generate_train_loader(args.batch_size,kwargs) for _ in range(args.num_workers)]

    resume_from_ckpt = args.resume if (args.resume and os.path.isfile(args.resume)) else None

    ps = ParameterServer.remote(args,test_loader)
    
    worker_tasks = [worker_task.remote(args,ps,idx,train_loaders[idx]) for idx in range(args.num_workers)]
    
    