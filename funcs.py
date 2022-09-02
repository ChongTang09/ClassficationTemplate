import torch
import numpy as np

from glob import glob
from tqdm import tqdm

from PIL import Image
from torchvision import transforms

from sklearn.metrics import confusion_matrix

def read_imgs_to_tensors(image_path, img_cls, resize_shape=[224, 224], num_channel=3):
    img_tensors = torch.zeros(len(glob(image_path+'/*.png')), num_channel, resize_shape[0], resize_shape[1])
    img_labels = torch.LongTensor([img_cls]*len(glob(image_path+'/*.png')))
    
    resize = transforms.Resize(resize_shape)
    to_tensor = transforms.ToTensor()

    for idx, img in enumerate(tqdm(glob(image_path+'/*.png'))):
        i = Image.open(img).convert('RGB') if num_channel == 3 else Image.open(img).convert('L')

        i = resize(i)
        i = to_tensor(i)
        
        img_tensors[idx, :, :, :] = i

    return img_tensors, img_labels

def classes2ids(classes):
    mp = {}

    for idx, cls in enumerate(classes):
        mp[cls] = idx

    return mp

def load_data(root_path, subfolder, classes):
    class_ids = classes2ids(classes)
    print('Loading {}...'.format(classes[0]))
    images, labels = read_imgs_to_tensors(root_path+subfolder+'/'+classes[0], class_ids[classes[0]])
    for cls in classes[1:]:
        print('Loading {}...'.format(cls))
        imgs, lbs = read_imgs_to_tensors(root_path+subfolder+'/'+cls, class_ids[cls])
        images = torch.cat((images, imgs), dim=0)
        labels = torch.cat((labels, lbs), dim=0)

    print('Loading Completed! \nSize of images: {}\nSize of labels: {}'.format(images.shape, labels.shape))

    return images, labels

def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    gt, pred = [], []
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()

        gt = gt + labels.cpu().detach().tolist()
        pred = pred + predictions.cpu().detach().tolist()
        
    cm = confusion_matrix(gt, pred)
    cm = cm/np.sum(cm, axis=1)

    return valid_loss,val_correct, cm