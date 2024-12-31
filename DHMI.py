from utils.tools_for_L_to_S import *
from utils.data import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_config():
    config = {
        "lambda": 0.01,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 2e-5, "weight_decay": 10 ** -5}},
        "info": "[DHMI_Ncenter]",
        "shuff": "shuff",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "root_dataset_path": "H:/Project_python/Image hash/image hash supervised/dataset",
        "net": ResNet_DHMI,
        "net_name":"ResNet",#"AlexNet"
        "dataset": "imagenet",
        "epoch": 300,
        "Init_epoch": 5,
        "test_map": 5,
        "device": torch.device("cuda:0"),
        "bit_list": [16,32,48,64],#16,32,64
    }
    config = config_dataset(config)
    return config


class CenterLoss(torch.nn.Module):
    def __init__(self, config, n_class,bits):
        super(CenterLoss, self).__init__()
        # self.init_bits = init_bits
        self.bits = bits
        self.n_class = n_class
        self.criterion_MSE = torch.nn.MSELoss(size_average=True).to(config["device"])
        self.relu = nn.ReLU()
        self.alpha = 0.05
        self.beta = 0.01
        self.gama = 0.001
        self.theta = 0
        self.hash_targets = self.get_hash_targets(self.n_class, self.bits).to(config["device"])
        self.one_hot = Variable(torch.ones((1, self.n_class)).type(torch.FloatTensor).to(config["device"]))
        self.I = Variable(torch.eye(self.n_class).type(torch.FloatTensor).to(config["device"]))

        self.criterion_BCE_n = torch.nn.BCELoss().to(config["device"])

    def forward(self,x_L, code):
        loss1 = self.relu((code.mm(code.t()) - self.bits * self.I))
        loss1 = loss1.pow(2).sum() / (self.n_class* self.n_class)

        # loss2 = (code - self.hash_targets).pow(2).sum() / (self.n_class* self.n_class)
        loss2 = self.criterion_BCE_n(0.5 * (code + 1), 0.5 * (self.hash_targets + 1))

        re = (torch.sign(code) - code).pow(2).sum() /self.n_class


        u_L_cov = torch.mm(x_L, x_L.T)
        u_S_cov = torch.mm(code, code.T)
        L_ij = torch.softmax(u_L_cov, dim=1)
        S_ij = torch.softmax(u_S_cov, dim=1)

        AB = torch.mm(L_ij, S_ij)
        BA = torch.mm(S_ij, L_ij)
        loss3 = self.criterion_MSE(AB, BA)

        loss =loss1 + self.theta*loss2 +self.beta * re + self.gama * loss3
        return  loss

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

class PretrainedLoss(torch.nn.Module):
    def __init__(self, config, n_class,bits):
        super(PretrainedLoss, self).__init__()
        # self.init_bits = init_bits
        self.bits = bits
        self.n_class = n_class
        self.criterion_MSE = torch.nn.MSELoss(size_average=True).to(config["device"])
        self.relu = nn.ReLU()
        self.alpha = 0.05
        self.beta = 0.01
        self.gama = 0.001
        self.theta = 1
        self.hash_targets = self.get_hash_targets(self.n_class, self.bits).to(config["device"])
        self.one_hot = Variable(torch.ones((1, self.n_class)).type(torch.FloatTensor).to(config["device"]))
        self.I = Variable(torch.eye(self.n_class).type(torch.FloatTensor).to(config["device"]))

        self.criterion_BCE_n = torch.nn.BCELoss().to(config["device"])

    def forward(self,x_L, code):
        loss1 = self.relu((code.mm(code.t()) - self.bits * self.I))
        loss1 = loss1.pow(2).sum() / (self.n_class* self.n_class)

        # loss2 = (code - self.hash_targets).pow(2).sum() / (self.n_class* self.n_class)
        loss2 = self.criterion_BCE_n(0.5 * (code + 1), 0.5 * (self.hash_targets + 1))
        # loss_b = self.one_hot.mm(code).pow(2).sum() / self.n_class

        re = (torch.sign(code) - code).pow(2).sum() /self.n_class


        u_L_cov = torch.mm(x_L, x_L.T)
        u_S_cov = torch.mm(code, code.T)
        # F_ij = torch.softmax(u_F_cov, dim=1)
        L_ij = torch.softmax(u_L_cov, dim=1)
        S_ij = torch.softmax(u_S_cov, dim=1)

        AB = torch.mm(L_ij, S_ij)
        BA = torch.mm(S_ij, L_ij)
        loss3 = self.criterion_MSE(AB, BA)

        loss =self.theta*loss2 #loss1 + self.alpha * 0 + self.beta * 0 + self.gama * loss3+
        return  loss#0.3*center_loss_L+

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


##########train center############
def get_center(bit):
    center_bit = bit
    if os.path.exists("center/" + config["dataset"] + '__N_code_' + str(center_bit) + '.pkl'):
        with open("center/" + config["dataset"] + '_label_code_' + str(center_bit) + '.pkl', 'rb') as f:
            label_code = pickle.load(f)
        print("load:", "center/" + config["dataset"] + '_label_code_' + str(center_bit) + '.pkl')
    else:
        print("training centers...")
        label_model = Label_net(config["n_class"], center_bit).to(config["device"])
        optimizer_label = optim.RMSprop(label_model.parameters(), lr= 2e-5, weight_decay=10 ** -5)  # weight_decay =
        cL = CenterLoss(config, config["n_class"], center_bit)
        pL = PretrainedLoss(config, config["n_class"], center_bit)

        H_K = hadamard(2048)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:config["n_class"]]).float().to(config["device"])
        labels = Variable(hash_targets)
        label_model.train()
        pre_trained = 1
        if pre_trained:
            for i in range(10000):
                x_L, code = label_model(labels)
                loss = pL(x_L, code)
                optimizer_label.zero_grad()
                loss.backward()
                optimizer_label.step()

        for i in range(5000):
            # scheduler_l.step()
            x_L, code = label_model(labels)
            loss = cL(x_L, code)
            optimizer_label.zero_grad()
            loss.backward()
            optimizer_label.step()
            if i == 1:
                print("\b\b\b\b\b\b\b init_center_loss:%.3f" % (loss))
        print("\b\b\b\b\b\b\b final_center_loss:%.3f" % (loss))


        label_model.eval()
        _, code = label_model(labels)
        label_code = torch.sign(code)
        with open("center/" + config["dataset"] + '_label_code_' + str(center_bit) + '.pkl', 'wb') as f:
            pickle.dump(label_code, f)
        with open("center/" + config["dataset"] + '_label_code_' + str(center_bit) + '.pkl', 'rb') as f:
            label_code = pickle.load(f)
        print("load:", "center/" + config["dataset"] + '_label_code_' + str(center_bit) + '.pkl')
        # print(label_code-cL.hash_targets)
    label_code_S = label_code.detach()

    return label_code_S

class DHMILoss(torch.nn.Module):
    def __init__(self, config, bits, init_bits,label_code_S,label_code_L,pretrained=0):
        super(DHMILoss, self).__init__()
        self.init_bits = init_bits
        self.bits = bits
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}

        if pretrained:
            self.hash_targets = label_code_L.to(config["device"])
            self.multi_label_random_center = torch.randint(2, (self.init_bits,)).float().to(config["device"])

            self.hash_targets_S = label_code_S.to(config["device"])
            self.multi_label_random_center_S = torch.randint(2, (bits,)).float().to(config["device"])
        else:

            self.hash_targets = self.get_hash_targets(config["n_class"], self.init_bits).to(config["device"])
            self.multi_label_random_center = torch.randint(2, (self.init_bits,)).float().to(config["device"])

            self.hash_targets_S = self.get_hash_targets(config["n_class"], bits).to(config["device"])
            self.multi_label_random_center_S = torch.randint(2, (bits,)).float().to(config["device"])

        self.criterion_BCE = torch.nn.BCELoss(reduction='none').to(config["device"])
        self.criterion_BCE_n = torch.nn.BCELoss().to(config["device"])
        self.criterion_L1 =torch.nn.L1Loss(size_average=True).to(config["device"])

        self.criterion = torch.nn.MSELoss(size_average=True).to(config["device"])


    def forward(self,u, u_L,u_S, y, ind, config):
        hash_center = self.label2center(y)
        hash_center_S = self.label2center_S(y)

        center_loss_cL = self.criterion_BCE(0.5 * (u_L + 1), 0.5 * (hash_center + 1))
        center_loss_cS = self.criterion_BCE(0.5 * (u_S + 1), 0.5 * (hash_center_S + 1))
        center_loss_L = torch.mean(center_loss_cL * torch.exp(-center_loss_cL/self.bits))
        center_loss_S = torch.mean(center_loss_cS * torch.exp(-center_loss_cS/self.bits))

        u_L_cov = torch.mm(u_L, u_L.T)
        u_S_cov = torch.mm(u_S, u_S.T)
        L_ij = torch.softmax(u_L_cov, dim=1)
        S_ij = torch.softmax(u_S_cov, dim=1)

        AB = torch.mm(L_ij, S_ij)
        BA = torch.mm(S_ij, L_ij)
        loss_eigen_L_S = self.criterion(AB, BA)
        loss_KL = loss_eigen_L_S#+loss_eigen_U_L#+loss_eigen_L_S#loss_eigen +loss_value

        Q_loss_L = (u_L.abs() - 1).pow(2).mean()
        Q_loss_S = (u_S.abs() - 1).pow(2).mean()
        return  0.5*center_loss_L+center_loss_S+loss_KL+ config["lambda"] * (Q_loss_L+Q_loss_S)

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def label2center_S(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets_S[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets_S
            random_center = self.multi_label_random_center_S.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center
    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


def train_val(config, bit):
    #init_bits = 4096
    init_bits = 2048
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit,init_bits=init_bits).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    Best_mAP = 0
    map_idx = [i * 10 for i in range(1, 16)]
    loss_idx = [i+1 for i in range(config["epoch"])]
    map_rec = []
    los_rec = []
    time_rec = []

    label_code_S = get_center(bit)
    label_code_L = get_center(init_bits)

    criterion = DHMILoss(config, bit, init_bits,label_code_S,label_code_L)

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        start_time = time.time()
        net.train()


        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u,u_L,u_S = net(image)

            loss = criterion(u,u_L,u_S, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        config["current_loss"] = train_loss
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        finish_time = time.time()
        cos_time = finish_time -start_time
        print("time cost: %.3f" % (finish_time - start_time))
        time_rec = time_rec + [cos_time]

        # if (epoch == config["epoch"] - 1):
        time_data = {
            "index": loss_idx,
            "loss": time_rec
        }
        os.makedirs(os.path.dirname(config["time_path"]), exist_ok=True)
        with open(config["time_path"], 'w') as f:
            f.write(json.dumps(time_data))


        los_rec = los_rec + [train_loss]

        loss_data = {
            "index": loss_idx,
            "loss": los_rec
        }
        os.makedirs(os.path.dirname(config["loss_path"]), exist_ok=True)
        with open(config["loss_path"], 'w') as f:
            f.write(json.dumps(loss_data))
        # if :
        if ((epoch + 1) % config["test_map"] == 0)&((epoch + 1)>=config["Init_epoch"]):
            Best_mAP,mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            map_rec = map_rec + [mAP]
            map_data = {
                "index": map_idx,
                "mAP": map_rec
            }
            os.makedirs(os.path.dirname(config["map_path"]), exist_ok=True)
            with open(config["map_path"], 'w') as f:
                f.write(json.dumps(map_data))

if __name__ == "__main__":
    config = get_config()
    print(config)
    config["bit_list"] = [64] #,3216,
    data_list = ["imagenet"]#,"imagenet","coco","cifar10-1","nuswide_21",,"coco","cifar10-1"
    for dataset in data_list:
        config['dataset'] = dataset
        config = config_dataset(config)
        for bit in config["bit_list"]:
            config[
                "pr_curve_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/pr_{config['info']}/"
            config[
                "loss_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/loss_{config['info']}/{config['dataset']}_{bit}.json"
            config[
                "map_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/map_{config['info']}/{config['dataset']}_{bit}.json"
            config[
                "time_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/time_{config['info']}/{config['dataset']}_{bit}.json"
            config[
                "save_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/ckp_{config['info']}/"
            train_val(config, bit)
