# Run Command
```
python train.py --epoch 10 --batch_size 8 --kfolds 5 --lr 1e-3 --spatial True --pretrain True --root YOUR_DATA_ROOT_PATH --subfolder YOUR_DATA_FOLDER --save_name YOUR_MODEL_SAVED_NAME
```

# Args
```
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
                    help='True is using SpatialGate')
parser.add_argument('--pretrain', default=True, type=bool,
                    help='True is loading pretrained model')
```

# Required Pkgs
See requirement.txt

# Reference
Some codes in model.py are created by:
https://github.com/Jongchan/attention-module/tree/c06383c514ab0032d044cc6fcd8c8207ea222ea7
