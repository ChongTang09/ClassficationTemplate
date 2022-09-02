import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

import funcs
from models import AlexNet

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# root_path = 'D:/Shelly\'s/Spectograms_77_24_Xethrue/'
# subfolders = ['activity_spectogram_77GHz', 'Spectrograms_24GHz', 'spectogram_Xethru']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root', default='D:/Shelly\'s/Spectograms_77_24_Xethrue/', type=str,
                    help='root folder of data')
parser.add_argument('--subfolder', default='activity_spectogram_77GHz', type=str,
                    help='Images folder')
parser.add_argument('--epoch', default=100, type=int,
                    help='number of training epoches')
parser.add_argument('--batch_size', default=128, type=int,
                    help='number of batch size')
parser.add_argument('--kfolds', default=5, type=int,
                    help='number of K folders')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--save_name', default='k_cross_model.pt', type=str,
                    help='the name of saved model')
parser.add_argument('--spatial', default=True, type=bool,
                    help='True is use SpatialGate')

if __name__ == '__main__':
    args = parser.parse_args()
    print ("args", args)

    if not os.path.isdir(args.root+args.subfolder):
        raise NameError("The directory does not exist. Please check inputs of --root and --subfolder args.")

    num_epochs=args.epoch
    batch_size=args.batch_size
    k=args.kfolds
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    foldperf={}

    classes = [x[0].replace('\\', '/').split('/')[-1] for x in os.walk(args.root+args.subfolder)][1:]
    
    images, labels = funcs.load_data(args.root, args.subfolder, classes)

    indices = torch.randperm(images.size()[0])
    images=images[indices]
    labels=labels[indices]

    dataset = TensorDataset(images, labels)

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        model = AlexNet(num_classes=len(classes), spatial=args.spatial)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

        for epoch in range(num_epochs):
            train_loss, train_correct=funcs.train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct=funcs.valid_epoch(model,device,test_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        foldperf['fold{}'.format(fold+1)] = history  

    torch.save(model, args.save_name)

    testl_f,tl_f,testa_f,ta_f=[],[],[],[]
    k=args.kfolds
    for f in range(1,k+1):

        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))     
