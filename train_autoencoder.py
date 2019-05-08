import os
import time

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import argparse
from photo_wct import PhotoWCT
import pdb

parser = argparse.ArgumentParser(description='Train encoder decoder')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
args = parser.parse_args()

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

num_epochs = 1000
batch_size = 16

traindir = 'face/'
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))


dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = p_wct.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss().cuda()
for epoch in range(num_epochs):

    model.train()
    total_time = time.time()
    correct = 0
    reconstruction_loss = 0
    for i, data in enumerate(dataloader):
        img = data[0]
        img = Variable(img).cuda()
        #y1, y2, y3, y4 = model(img)
        y1, y2, y3, y4 = model(img)
        #loss = criterion(y1, img) + criterion(y2, img) + criterion(y3, img) + criterion(y4, img)
        loss1 = criterion(y1, img)
        loss2 = criterion(y2, img)
        loss3 = criterion(y3, img)
        loss4 = criterion(y4, img)
        loss = loss1 + loss2 + loss3 + loss4
        loss = 100 * loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reconstruction_loss += loss.item()
        if i % 10 == 0:
            print(str(i) + ':--reconstruction loss: ' + str(loss.item()))

    print("Epoch {} complete\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
    torch.save(model.state_dict(), './PhotoWCTModels/photo_wct_new.pth')
