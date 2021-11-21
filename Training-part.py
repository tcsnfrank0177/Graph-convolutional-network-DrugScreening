# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from sch import SchNetModel
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher
import torch.nn.functional as F


def print_res(label, res, op):
    size = len(res)
    for i in range(size):
        line = "%s,%s\n" % (label[i], res[i])
        op.writelines(line)


def binary_acc(y_pred, y_test):
    acc = 0
    tot = 0

    for i in range(len(y_pred)):
        tot += 1
        pred_l = y_pred[i][0] >= y_pred[i][1]
        test_l = y_test[i][0] >= y_test[i][1]
        if pred_l == test_l:
            acc += 1

    acc = acc/tot*100
    # acc = th.round(acc * 100)
    return acc


def loss_sq(y_pred, y_test):
    diff = y_pred-y_test
    diff_sq = diff**2
    loss = diff_sq.sum()

    # acc = acc/tot
    # # acc = th.round(acc * 100)
    return loss


def predict(res):

    ans = []
    preds = F.softmax(res)
    for pred in preds:
        if pred[0] > pred[1]:
            ans.append(0)
        else:
            ans.append(1)
        return th.tensor(ans)


def train(model="sch", epochs=80, device=th.device("cpu"), dataset='', save='./'):
    print("start")
    train_dir = "./"
    train_file = dataset+"_train.csv"
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_dataset.mode = "Train"
    alchemy_dataset.transform = None
    alchemy_dataset.file_path = train_file
    alchemy_dataset._load()

    test_dataset = TencentAlchemyDataset()
    test_dir = train_dir
    test_file = dataset+"_valid.csv"
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    alchemy_loader = DataLoader(
        dataset=alchemy_dataset,
        batch_size=50,
        collate_fn=batcher(),
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=50,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )

    if model == "sch":
        model = SchNetModel(norm=False, output_dim=2)
    print(model)
    # if model.name in ["MGCN", "SchNet"]:
    #     model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)
    # print("test_dataset.mean= %s" % (alchemy_dataset.mean))
    # print("test_dataset.std= %s" % (alchemy_dataset.std))

    # loss_fn = nn.MSELoss()
    # MAE_fn = nn.L1Loss()
    # BceLoss = nn.CrossEntropyLoss()

    # BceLoss = nn.BCELoss()
    # BceLoss = loss_sq()
    # BceLoss = nn.BCEWithLogitsLoss
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        w_loss, w_acc = 0, 0
        model.train()
        if epoch % 20 == 0:
            res_op = open(save+'/Trainres_'+str(epoch)+'.csv', 'w')
        for idx, batch in enumerate(alchemy_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            # res = predict(res)
            # print(f'res={res}')

            # print(f'batch.label={batch.label}')
            # print(f'Res: {res.cpu().detach().numpy()}')
            # print(f'Label: {batch.label.cpu().detach().numpy()}')
            # loss = BceLoss(F.sigmoid(res), F.sigmoid(batch.label))
            loss = loss_sq(res, batch.label)
            # print(loss)
            acc = binary_acc(res, batch.label)
            # loss = loss_fn(res, batch.label)
            # mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            loss.backward()
            # acc.backward()
            optimizer.step()

            # w_mae += mae.detach().item()
            w_loss += loss.detach().item()
            w_acc += acc
            l = batch.label.cpu().detach().numpy()
            r = res.cpu().detach().numpy()
            if epoch % 20 == 0:
                print_res(l, r, res_op)

        # w_mae /= idx + 1
        w_loss /= idx + 1
        w_acc /= idx+1

        print("Epoch {:2d}, loss: {:.7f}, ACC: {:.7f}".format(
            epoch, w_loss,  acc))

        val_loss, val_acc = 0, 0
        if epoch % 20 == 0:
            valid_op = open(save+'/Validres_'+str(epoch)+'.csv', 'w')
        for jdx, batch in enumerate(test_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            # loss = BceLoss(F.sigmoid(res), F.sigmoid(batch.label))
            loss = loss_sq(res, batch.label)
            # loss = loss_fn(res, batch.label)
            # mae = MAE_fn(res, batch.label)
            acc = binary_acc(res, batch.label)
            # optimizer.zero_grad()
            # mae.backward()
            # optimizer.step()

            # val_mae += mae.detach().item()
            val_loss += loss.detach().item()
            val_acc += acc
            l = batch.label.cpu().detach().numpy()
            r = res.cpu().detach().numpy()
            if epoch % 20 == 0:
                print_res(l, r, valid_op)
        # val_mae /= jdx + 1
        val_loss /= jdx + 1
        val_acc /= jdx + 1
        print("Epoch {:2d}, val_loss: {:.7f},  val_ACC: {:.7f}".format(
            epoch, val_loss, val_acc))

        if epoch % 80 == 0:
            th.save(model.state_dict(), save+"/model_"+str(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=10000)
    parser.add_argument("--dataset", help="dataset to train", default="")
    parser.add_argument("--save", help="folder to save", default="")
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ["sch"]
    # dataset_split("delaney.csv")
    train(args.model, int(args.epochs), device, args.dataset, args.save)
