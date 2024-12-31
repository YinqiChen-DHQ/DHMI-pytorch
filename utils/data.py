import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json

def config_dataset(config):
    root_dataset = config["root_dataset_path"]

    if "cifar" in config["dataset"]:
        config["topK"] = 1000
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20
    elif config["dataset"] == "food101":
        config["topK"] = -1
        config["n_class"] = 101
    elif config["dataset"] == "FGVC_Aircraft":
        config["topK"] = -1
        config["n_class"] = 100

    elif config["dataset"] in ["nabirds", "nabirds_new"]:
        config['topK'] = -1
        config['n_class'] = 555
    elif config["dataset"] in ["car_imgs", "car_imgs_new"]:
    # elif config["dataset"] == "car_imgs":
        config['topK'] = -1
        config['n_class'] = 196

    config["data_path"] = root_dataset+"/images/" + config["dataset"] + "/"
    if config["dataset"] == "cifar-10":
        config["data_path"] = root_dataset+"/images/cifar/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = root_dataset+"/images/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = root_dataset+"/images/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = root_dataset+"/images/COCO_2014/"
    if config["dataset"] == "imagenet":
        config["data_path"] = root_dataset + "/images/imagenet/"
    if config['dataset'] == 'nabirds':
        config['data_path'] = root_dataset + "/images/nabirds/"
    if config['dataset'] == 'car_imgs':
        config['data_path'] = root_dataset + "/images/car_imgs/"
    if config['dataset'] == 'food101':
        config['data_path'] = root_dataset + "/images/food101/"
    if config['dataset'] == 'FGVC_Aircraft':
        config['data_path'] = root_dataset + "/images/FGVC_Aircraft/"


    # if config["dataset"] == "voc2012":
    #     config["data_path"] = "./dataset/"


    config["data"] = {
        "train_set": {"list_path": root_dataset+"/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": root_dataset+"/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": root_dataset+"/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    if config["dataset"] in ["car_imgs", "nabirds"]:
        config["data"] = {
            "train_set": {"list_path": root_dataset + "/data/" + config["dataset"] + "/train.txt","batch_size": config["batch_size"]},
            "database": {"list_path": root_dataset + "/data/" + config["dataset"] + "/train.txt","batch_size": config["batch_size"]},
            "test": {"list_path": root_dataset + "/data/" + config["dataset"] + "/test.txt","batch_size": config["batch_size"]}}
    if config["dataset"] in ["car_imgs_new", "nabirds_new"]:

        config["data"] = {
            "train_set": {"list_path": root_dataset + "/data/" + config["dataset"] + "/train.txt","batch_size": config["batch_size"]},
            "database": {"list_path": root_dataset + "/data/" + config["dataset"] + "/test.txt","batch_size": config["batch_size"]},
            "test": {"list_path": root_dataset + "/data/" + config["dataset"] + "/valid.txt","batch_size": config["batch_size"]}}



    return config

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/dataset/cifar/'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

