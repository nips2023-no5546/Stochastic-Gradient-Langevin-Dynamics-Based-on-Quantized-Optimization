#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Service Function
#
# Base URL     : https://wikidocs.net/55580
###########################################################################
_description = '''\
====================================================
Read_Data.py
                    Written by ******** @ 2021-03-10
====================================================
for Reading of data set for Torch testbench  
'''
#=============================================================
# Definitions
#=============================================================
import os
import wget
import json
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

from torch.utils.data import TensorDataset  # 텐서 데이터 셋
#from torch.utils.data import DataLoader     # 데이터 로더

import my_debug as DBG

# For fix codes the MNIST data loading
'''
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
'''

g_datadir     = 'Torch_Data/'
def bar_custom(current, total, width=80):
    progress = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    return progress
class MNIST_set:
    def __init__(self, batch_size=100, bdownload=True):
        self.batchsize   = batch_size
        self.datadir     = g_datadir
        self.mnist_train = dsets.MNIST(root=self.datadir,  # 다운로드 경로 지정
                         train      =True,                  # True를 지정 하면 훈련 데이터로 다운로드
                         transform  =transforms.ToTensor(), # 텐서로 변환
                         download   =bdownload)

        self.mnist_test  = dsets.MNIST(root=self.datadir,  # 다운로드 경로 지정
                         train      =False,                 # False를 지정 하면 테스트 데이터로 다운로드
                         transform  =transforms.ToTensor(), # 텐서로 변환
                         download   =bdownload)
        self.inputCH     = 1
        self.datashape   = self.mnist_test.data.shape[1:2]
        self.outputCH    = len(self.mnist_train.classes)


    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset    = self.mnist_train if bTrain else self.mnist_test
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                    batch_size =self.batchsize,
                    shuffle    =bsuffle,
                    drop_last  =bdrop_last)
        return loadingData

    def Test_Function(self, model, _device, ClassChk=False, bTrain=False):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            # For MNIST : Data loading on CPU or GPU
            if bTrain:
                _total, _correct = 0, 0
                LoadingData = self.data_loader(bTrain=bTrain, bsuffle=False)
                # for _data in LoadingData:
                for _X, _Y in LoadingData:
                    # _X, _Y = _data
                    _X, _Y = _X.to(_device), _Y.to(_device)
                    _prediction = model(_X)
                    _value, _predicted = torch.max(_prediction.data, 1)
                    _total += _Y.size(0)
                    _correct += (_predicted == _Y).sum().item()

                _accuracy = _correct / _total
            else:
                _X = self.mnist_test.data.view(len(self.mnist_test), 1, 28, 28).float()
                _Y = self.mnist_test.targets
                _X, _Y = _X.to(_device), _Y.to(_device)

                _prediction = model(_X)
                _correct_chk= torch.argmax(_prediction, 1) == _Y
                _score      = _correct_chk.float().mean()
                _total      = len(_correct_chk)
                _correct    = _correct_chk.sum()
                _accuracy   = _score.item()

            return _total, _correct, _accuracy

class CIFAR10_set:
    # batch size = 4 추천
    def __init__(self, batch_size=4, bdownload=True):
        self.batchsize  = batch_size
        self.datadir    = g_datadir
        self.classes    = ('plane', 'car', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck')

        self.transform  = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset   = dsets.CIFAR10(root=self.datadir,
                                        train=True,
                                        transform=self.transform,
                                        download=bdownload)
        self.testset    = dsets.CIFAR10(root=self.datadir,
                                        train=False,
                                        transform=self.transform,
                                        download=bdownload)
        self.inputCH     = 3
        self.datashape   = self.trainset.data.shape
        self.outputCH    = len(self.trainset.classes)

    # Train 경우 shuffle 은 True, Test 경우 False 추천
    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset = self.trainset if bTrain else self.testset
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                                                  batch_size=self.batchsize,
                                                  shuffle=bsuffle,
                                                  num_workers=2,
                                                  drop_last=bdrop_last)
        return loadingData

    def Test_Function(self, model, _device, ClassChk=False, bTrain=False ):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            _total, _correct = 0, 0
            LoadingData = self.data_loader(bTrain=bTrain, bsuffle=False)
            #for _data in LoadingData:
            for _X, _Y in LoadingData:
                #_X, _Y = _data
                _X, _Y = _X.to(_device), _Y.to(_device)
                _prediction = model(_X)
                _value, _predicted = torch.max(_prediction.data, 1)
                _total += _Y.size(0)
                _correct += (_predicted == _Y).sum().item()

            _accuracy = _correct / _total

            return _total, _correct, _accuracy

class CIFAR100_set:
    # batch size = 16 추천
    def __init__(self, batch_size=16, bdownload=True):
        self.batchsize  = batch_size
        self.datadir    = g_datadir
        self.classes    = sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',        # aquatic mammals
                           'aquarium' 'fish', 'flatfish', 'ray', 'shark', 'trout',      # fish
                           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',       # flowers
                           'bottles', 'bowls', 'cans', 'cups', 'plates',                # food containers
                           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',  # fruit and vegetables
                           'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television', # household electrical devices
                           'bed', 'chair', 'couch', 'table', 'wardrobe',                # household furniture
                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',    # insects
                           'bear', 'leopard', 'lion', 'tiger', 'wolf',                  # large carnivores
                           'bridge', 'castle', 'house', 'road', 'skyscraper',           # large man-made outdoor things
                           'cloud', 'forest', 'mountain', 'plain', 'sea',               # large natural outdoor scenes
                           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',     # large omnivores and herbivores
                           'fox', 'porcupine', 'possum', 'raccoon', 'skunk',            # medium-sized mammals
                           'crab', 'lobster', 'snail', 'spider', 'worm',                # non-insect invertebrates
                           'baby', 'boy', 'girl', 'man', 'woman',                       # people
                           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',        # reptiles
                           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',           # small mammals
                           'maple', 'oak', 'palm', 'pine', 'willow',                    # trees
                           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',     # vehicles 1
                           'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'       # vehicles 2
                          ])

        self.transform  = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset   = dsets.CIFAR100(root=self.datadir,
                                        train=True,
                                        transform=self.transform,
                                        download=bdownload)
        self.testset    = dsets.CIFAR100(root=self.datadir,
                                        train=False,
                                        transform=self.transform,
                                        download=bdownload)
        self.inputCH     = 3
        self.datashape   = self.trainset.data.shape
        self.outputCH    = len(self.trainset.classes)

    # Train 경우 shuffle True, Test 경우 False 추천
    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset = self.trainset if bTrain else self.testset
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                                                  batch_size=self.batchsize,
                                                  shuffle=bsuffle,
                                                  num_workers=2,
                                                  drop_last=bdrop_last)
        return loadingData

    def Test_Function(self, model, _device, ClassChk=False, bTrain=False ):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            _total, _correct = 0, 0
            LoadingData = self.data_loader(bTrain=bTrain, bsuffle=False)
            #for _data in LoadingData:
            for _X, _Y in LoadingData:
                #_X, _Y = _data
                _X, _Y = _X.to(_device), _Y.to(_device)
                _prediction = model(_X)
                _value, _predicted = torch.max(_prediction.data, 1)
                _total += _Y.size(0)
                _correct += (_predicted == _Y).sum().item()

            _accuracy = _correct / _total

            return _total, _correct, _accuracy

class ImageNet_set:
    # batch size = 64 추천
    def __init__(self, batch_size=64, bdownload=False):
        self.batchsize  = batch_size
        self.datadir    = g_datadir

        # For ImageNet processing
        self.ImageNetDatadir = os.path.join(os.getcwd(), self.datadir)
        self.classes    = self.get_ImageNet_Label()
        self.download_imagenet()

        self.transform  = transforms.Compose(
                        [transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.trainset   = dsets.ImageNet(root=self.datadir,
                                        train=True,
                                        transform=self.transform,
                                        split='val',
                                        download=bdownload)
        self.testset    = dsets.ImageNet(root=self.datadir,
                                        train=False,
                                        transform=self.transform,
                                        split='val',
                                        download=bdownload)
        self.inputCH     = 3
        self.datashape   = self.trainset.data.shape
        self.outputCH    = len(self.trainset.classes)

    # Train 경우 shuffle True, Test 경우 False 추천
    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset = self.trainset if bTrain else self.testset
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                                                  batch_size=self.batchsize,
                                                  shuffle=bsuffle,
                                                  num_workers=2,
                                                  drop_last=bdrop_last)
        return loadingData

    def Test_Function(self, model, _device, ClassChk=False, bTrain=False ):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            _total, _correct = 0, 0
            LoadingData = self.data_loader(bTrain=bTrain, bsuffle=False)
            #for _data in LoadingData:
            for _X, _Y in LoadingData:
                #_X, _Y = _data
                _X, _Y = _X.to(_device), _Y.to(_device)
                _prediction = model(_X)
                _value, _predicted = torch.max(_prediction.data, 1)
                _total += _Y.size(0)
                _correct += (_predicted == _Y).sum().item()

            _accuracy = _correct / _total

            return _total, _correct, _accuracy

    def get_ImageNet_Label(self):
        _fname          = 'imagenet-simple-labels.json'
        full_filename   = os.path.join(self.ImageNetDatadir, _fname)
        try:
            with open(full_filename) as f:
                labels = json.load(f)
        except Exception as e:
            DBG.dbg("ImageNet Label data open is Fail!!")
            DBG.dbg("Program terminated")
            exit()

        return labels

    def download_imagenet(self):
        """
        download_imagenet validation set
        :param img_dir: root for download imagenet
        :return:
        """
        # make url
        val_url = 'http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar'
        devkit_url = 'http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_devkit_t12.tar.gz'

        _ImageNetDataFileName   = "ILSVRC2012_img_val.tar"
        _ImageNetdevkitFileName = "ILSVRC2012_devkit_t12.tar.gz"
        root = self.ImageNetDatadir

        if os.path.isfile(_ImageNetDataFileName) and os.path.isfile(_ImageNetdevkitFileName):
            print("There exists %s and %s in %s " %(_ImageNetDataFileName, _ImageNetdevkitFileName, root))
        else:
            print("Download...")
            os.makedirs(root, exist_ok=True)
            try:
                wget.download(url=val_url, out=root, bar=bar_custom)
                print('')
                wget.download(url=devkit_url, out=root, bar=bar_custom)
                print('')
                print('Downloading is success!!')
            except Exception as e:
                DBG.dbg("Downloading the ImageNet Data set %s is fail !!" %(_ImageNetDataFileName))
                DBG.dbg(e)
                DBG.dbg("Program termination")
                exit()


class FashionMNIST_set:
    # batch size = 4 추천
    def __init__(self, batch_size=4, bdownload=True):
        self.batchsize  = batch_size
        self.datadir    = g_datadir
        self.classes    = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat",
                           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

        self.transform  = transforms.Compose(
                        [transforms.ToTensor()])

        self.trainset   = dsets.FashionMNIST(root=self.datadir,
                                        train=True,
                                        transform=self.transform,
                                        download=bdownload)
        self.testset    = dsets.FashionMNIST(root=self.datadir,
                                        train=False,
                                        transform=self.transform,
                                        download=bdownload)
        self.inputCH     = 1
        self.datashape   = self.trainset.data.shape
        self.outputCH    = len(self.trainset.classes)

    # Train 경우 shuffle 은 True, Test 경우 False 추천
    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset = self.trainset if bTrain else self.testset
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                                                  batch_size=self.batchsize,
                                                  shuffle=bsuffle,
                                                  num_workers=2,
                                                  drop_last=bdrop_last)
        return loadingData

    def Test_Function(self, model, _device, ClassChk=False, bTrain=False ):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            _total, _correct = 0, 0
            LoadingData = self.data_loader(bTrain=bTrain, bsuffle=False)
            #for _data in LoadingData:
            for _X, _Y in LoadingData:
                #_X, _Y = _data
                _X, _Y = _X.to(_device), _Y.to(_device)
                _prediction = model(_X)
                _value, _predicted = torch.max(_prediction.data, 1)
                _total += _Y.size(0)
                _correct += (_predicted == _Y).sum().item()

            _accuracy = _correct / _total

            return _total, _correct, _accuracy




#=============================================================
# Test Processing
#=============================================================
if __name__ == "__main__":
    # =================================================================
    # Parsing the Argument
    # =================================================================
    import argparse
    import textwrap

    def _ArgumentParse(_intro_msg):
        parser = argparse.ArgumentParser(
            prog='test pytorch_inference',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(_intro_msg))

        parser.add_argument('-d', '--dataset',
                            help="Name of Data SET '(MNIST)', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'ImageNet'",
                            type=str, default='MNIST')
        parser.add_argument('-t', '--training',
                            help="[0] test [(1)] training",
                            type=int, default=1)
        args = parser.parse_args()
        args.training = True if args.training == 1 else False
        return args
    #=============================================================
    # Test Processing
    #=============================================================
    _args = _ArgumentParse(_description)

    if _args.dataset == 'MNIST':
        # For fix codes the MNIST data loading
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        Dset        = MNIST_set(batch_size=100, bdownload=True)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    elif _args.dataset == 'CIFAR10':
        Dset        = CIFAR10_set(batch_size=4, bdownload=True)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    elif _args.dataset == 'CIFAR100':
        Dset        = CIFAR100_set(batch_size=16, bdownload=True)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    elif _args.dataset == 'FashionMNIST':
        Dset        = FashionMNIST_set(batch_size=100, bdownload=True)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    elif _args.dataset == 'ImageNet':
        Dset        = ImageNet_set(batch_size=64, bdownload=False)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    else:
        LoadingData = []
        print("Data set is not depicted. It is error!!!")
        exit()

    print("Total number of batch : ", len(LoadingData))

    # K : The index of Batch : max K = number of batch
    K = 0
    for X, Y in LoadingData:
        if K == 0:
            print("Data <dim> : ", X.shape)
            print("Label<dim> : ", Y.shape)
        K+= 1

    #=============================================================
    # Final Stage
    #=============================================================

    print("Process Finished!!")