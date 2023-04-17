import sys
import os
import csv
from evm_cfg_builder.cfg import CFG
from CFGFeature import CFGFeature
from load_data import GraphData,GraphData_pca,OpCodeData,collate_graph_batch,collate_opcode_batch, PatternData, MLPData, gblData, collate_dgl
from torch.utils.data import DataLoader
import torch.utils.data
import torch.nn.functional as F
from parser1 import parameter_parser
from model.gcn_origin import GCN_normal
from model.dgl_gcn import dgl_GCN_normal
from model.gru import GRU_normal
from model.pattern import Pattern_normal
from model.MLP import Out_normal
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader
import json
import time

# 读取字节码列表
def read_bytecode():
    info_data_train = []
    info_data_test = []
    # if args.mode == 'test':
    #     path = "./compile/bytecode_dict_reentrancy_test.json"
    #     # path = "./compile/bytecode_dict.json"
    # elif args.mode == 'train':
    #     path = "./compile/bytecode_dict_reentrancy_train.json"
    path_test = "./compile/bytecode_dict_reentrancy_test.json"
    path_train = "./compile/bytecode_dict.json"
    # path_train = "./compile/bytecode_dict_reentrancy_train.json"
    with open(path_train, "r") as f:
        for line in f:
            try:
                info_data_train.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))

    with open(path_test, "r") as f:
        for line in f:
            try:
                info_data_test.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    return info_data_train,info_data_test

def create_w2v_word(cfg_feature_list_train:list):
    # create the w2v corpus 生成w2v的词汇表
    with open('compile/word.txt','a+') as writers:
        for i in range(len(cfg_feature_list_train)):
            for j in range(len(cfg_feature_list_train[i].allInstructions)):
                writers.write(str(cfg_feature_list_train[i].instructions[j]) + " ")
    writers.write("\n")

if __name__ == '__main__':
    args = parameter_parser()
    args.filters = list(map(int, args.filters.split(',')))
    args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))
    for arg in vars(args):
        print(arg, getattr(args, arg))

    cfg_feature_list_train = []
    cfg_feature_list_test = []
    bytecode_dict_list_train, bytecode_dict_list_test = read_bytecode()
    # print(bytecode_dict_list)
    for i in range(len(bytecode_dict_list_train)):
        bytecode = bytecode_dict_list_train[i]
        cfg = CFG(bytecode['runtime_bytecode'])
        cfg_feature = CFGFeature(key=bytecode['name'],cfg=cfg,label=bytecode['label'],pattern=bytecode['pattern'])
        # print(str(cfg_feature.basicBlock[0].instructions[0]).split(" "))
        cfg_feature_list_train.append(cfg_feature)
    # 测试集数据
    for i in range(len(bytecode_dict_list_test)):
        bytecode = bytecode_dict_list_test[i]
        cfg = CFG(bytecode['runtime_bytecode'])
        cfg_feature = CFGFeature(key=bytecode['name'],cfg=cfg,label=bytecode['label'],pattern=bytecode['pattern'])
        # print(str(cfg_feature.basicBlock[0].instructions[0]).split(" "))
        cfg_feature_list_test.append(cfg_feature)
    # print(cfg_feature_list_train)
    # create_w2v_word(cfg_feature_list_train)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # rnd_state = np.random.RandomState(args.seed)

    if args.model == 'all':
        datareader_graph_gbl_train = gblData(cfg_feature_list_train)
        datareader_graph_gbl_test = gblData(cfg_feature_list_test)
        datareader_graph_train = GraphData_pca(cfg_feature_list_train)
        datareader_graph_test = GraphData_pca(cfg_feature_list_test)
        datareader_opcode_train = OpCodeData(cfg_feature_list_train)
        datareader_opcode_test = OpCodeData(cfg_feature_list_test)
        datareader_pattern_train = PatternData(cfg_feature_list_train)
        datareader_pattern_test = PatternData(cfg_feature_list_test)
    elif args.model == 'dgl':
        # datareader_graph = GraphData(cfg_feature_list_train)
        datareader_graph_gbl_train = gblData(cfg_feature_list_train)
        datareader_graph_gbl_test = gblData(cfg_feature_list_test)
    elif args.model == 'gcn':
        datareader_graph_train = GraphData_pca(cfg_feature_list_train)
        datareader_graph_test = GraphData_pca(cfg_feature_list_test)
    elif args.model == 'gru':
        datareader_opcode_train = OpCodeData(cfg_feature_list_train)
        datareader_opcode_test = OpCodeData(cfg_feature_list_test)
    elif args.model == 'pattern':
        datareader_pattern_train = PatternData(cfg_feature_list_train)
        datareader_pattern_test = PatternData(cfg_feature_list_test)

    # 模型定义
    dgl_gcn_model = dgl_GCN_normal(input_dim=300, hidden_dim=64, output_dim=2)
    gcn_model = GCN_normal(input_dim=args.node_input_dim,hidden_dim=64,output_dim=2)
    gru_model = GRU_normal(input_dim=300,hidden_dim=64,output_dim=2)
    pattern_model = Pattern_normal(input_dim=64,hidden_dim=64,output_dim=2)
    output_model = Out_normal(input_dim=64, output_dim= 2)

    # loader_graph = DataLoader(dataset=datareader_graph, batch_size=25, shuffle=True, drop_last=True,collate_fn=collate_graph_batch)
    # 生成DataLoader，如果不是all，则不需要生成其他模型的DataLoader
    if args.model == 'all':
        loader_graph_gbl_train = GraphDataLoader(dataset=datareader_graph_gbl_train,batch_size=len(cfg_feature_list_train),collate_fn=collate_dgl)
        loader_graph_gbl_test = GraphDataLoader(dataset=datareader_graph_gbl_test, batch_size=len(cfg_feature_list_test) ,collate_fn=collate_dgl)

        loader_graph_train = DataLoader(dataset=datareader_graph_train,batch_size=len(cfg_feature_list_train))
        loader_graph_test = DataLoader(dataset=datareader_graph_test, batch_size=len(cfg_feature_list_test))

        loader_opcode_train = DataLoader(dataset=datareader_opcode_train,batch_size=len(cfg_feature_list_train),collate_fn=collate_opcode_batch)
        loader_opcode_test = DataLoader(dataset=datareader_opcode_test, batch_size=len(cfg_feature_list_test), collate_fn=collate_opcode_batch)

        loader_pattern_train = DataLoader(dataset=datareader_pattern_train,batch_size=len(cfg_feature_list_train))
        loader_pattern_test = DataLoader(dataset=datareader_pattern_test, batch_size=len(cfg_feature_list_test))

        out_feature_train_list = []
        out_feature_test_list = []
        # get the feature after dgl_gcn
        dgl_gcn_model.model = torch.load('model/pth/dgl_GCN_origin_D2.pth')
        # dgl_gcn_model.model = torch.load('dgl_GCN_origin.pth')
        gcn_feature_train = dgl_gcn_model.get_hidden(loader_graph_gbl_train)
        out_feature_train_list.append(gcn_feature_train[0].detach())
        gcn_feature_test = dgl_gcn_model.get_hidden(loader_graph_gbl_test)
        out_feature_test_list.append(gcn_feature_test[0].detach())

        # get the feature after gcn
        # gcn_model.model = torch.load('GCN_origin.pth')
        # gcn_feature_train = gcn_model.get_hidden(loader_graph_train)
        # out_feature_train_list.append(gcn_feature_train[0].detach())
        # gcn_feature_test = gcn_model.get_hidden(loader_graph_test)
        # out_feature_test_list.append(gcn_feature_test[0].detach())

        gru_model.model = torch.load('model/pth/gru_D2.pth')
        # gru_model.model = torch.load('gru.pth')
        gru_feature_train = gru_model.get_hidden(loader_opcode_train)
        out_feature_train_list.append(gru_feature_train[0].detach())
        gru_feature_test = gru_model.get_hidden(loader_opcode_test)
        out_feature_test_list.append(gru_feature_test[0].detach())

        pattern_model.model = torch.load('model/pth/pattern_D2.pth')
        # pattern_model.model = torch.load('pattern.pth')
        pattern_feature_train = pattern_model.get_hidden(loader_pattern_train)
        out_feature_train_list.append(pattern_feature_train[0].detach())  # pattern的特征
        out_feature_train_list.append(pattern_feature_train[1].detach())  # 标签
        pattern_feature_test = pattern_model.get_hidden(loader_pattern_test)
        out_feature_test_list.append(pattern_feature_test[0].detach())  # pattern的特征
        out_feature_test_list.append(pattern_feature_test[1].detach())  # 标签

        datareader_out_train = MLPData(out_feature_train_list)
        datareader_out_test = MLPData(out_feature_test_list)
        loader_out_train = DataLoader(dataset=datareader_out_train, batch_size=25, shuffle=True, drop_last=True)
        loader_out_test = DataLoader(dataset=datareader_out_test, batch_size=len(cfg_feature_list_train))
    else:
        if args.model == 'dgl':
            # gbl数据集，使用for g, labels in dataloader：
            loader_graph_gbl_train = GraphDataLoader(dataset=datareader_graph_gbl_train, batch_size=25, shuffle=True, drop_last= True, collate_fn=collate_dgl)
            loader_graph_gbl_test = GraphDataLoader(dataset=datareader_graph_gbl_test, batch_size=len(cfg_feature_list_train), collate_fn=collate_dgl)
        elif args.model == 'gcn':
            loader_graph_train = DataLoader(dataset=datareader_graph_train, batch_size=25, shuffle=True, drop_last=True)
            loader_graph_test = DataLoader(dataset=datareader_graph_test, batch_size=len(cfg_feature_list_train))
        elif args.model == 'gru':
            loader_opcode_train = DataLoader(dataset=datareader_opcode_train, batch_size=25, shuffle=True, drop_last=True,collate_fn=collate_opcode_batch)
            loader_opcode_test = DataLoader(dataset=datareader_opcode_test, batch_size=len(cfg_feature_list_train), collate_fn=collate_opcode_batch)
        elif args.model == 'pattern':
            loader_pattern_train = DataLoader(dataset=datareader_pattern_train, batch_size=25,shuffle=True,drop_last=True)
            loader_pattern_test = DataLoader(dataset=datareader_pattern_test, batch_size=len(cfg_feature_list_train))



    # gcn_model = GCN_modify(args.node_input_dim,2,filters=args.filters,n_hidden=args.n_hidden,dropout=args.dropout)
    # 训练模型
    loss_list = []
    if args.mode == 'train' and args.model != 'all':
        for i in range(args.epochs):
            if args.model == 'dgl':
                loss = dgl_gcn_model.train(loader_graph_gbl_train,i)
                loss_list.append(loss)
            elif args.model == 'gcn':
                loss = gcn_model.train(loader_graph_train,i)
                loss_list.append(loss)
            elif args.model == 'gru':
                loss = gru_model.train(loader_opcode_train,i)
                loss_list.append(loss)
            elif args.model == 'pattern':
                loss = pattern_model.train(loader_pattern_train,i)
                loss_list.append(loss)
        x = [i for i in range(args.epochs)]
        plt.plot(x, loss_list)
        plt.show()
        print(args.model, "loss")

    elif args.mode == 'train' and args.model == 'all':
        for i in range(args.epochs):
            loss = output_model.train(loader_out_train,i)
            loss_list.append(loss)
        x = [i for i in range(args.epochs)]
        plt.plot(x, loss_list)
        plt.show()
        print(args.model, "loss")

    # test
    if args.model == 'gcn':
        gcn_model.model = torch.load('model/pth/GCN_origin.pth')
        gcn_model.test(loader_graph_test, i)
    elif args.model == 'dgl':
        dgl_gcn_model.model = torch.load('model/pth/dgl_GCN_origin_D2.pth')
        dgl_gcn_model.test(loader_graph_gbl_test, i)
    elif args.model == 'gru':
        gru_model.model = torch.load('model/pth/gru_D2.pth')
        gru_model.test(loader_opcode_test,i)
    elif args.model == 'pattern':
        pattern_model.model = torch.load('model/pth/pattern_D2.pth')
        pattern_model.test(loader_pattern_test, i)
    elif args.model == 'all':
        output_model.model = torch.load('model/pth/output_D2.pth')
        output_model.test(loader_out_test, i)




    # def visual(feat):
    #     # t-SNE的最终结果的降维与可视化
    #     ts = manifold.TSNE(n_components=3, init='pca', random_state=0)
    #     # ts = manifold.TSNE(n_components=2, init='pca')
    #
    #     x_ts = ts.fit_transform(feat)
    #
    #     print(x_ts.shape)  # [num, 3]
    #
    #     x_min, x_max = x_ts.min(0), x_ts.max(0)
    #
    #     x_final = (x_ts - x_min) / (x_max - x_min)
    #
    #     return x_final
    #
    #
    # # 设置散点形状
    # # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # # 设置散点颜色
    # colors = ['#ee4545', '#5dde60']
    # # 图例名称
    # Label_Com = ['yes','no']
    # # 设置字体格式
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'bold',
    #          'size': 32,
    #          }
    #
    #
    # def plotlabels(S_lowDWeights, Trure_labels):
    #     True_labels = Trure_labels.reshape((-1, 1))
    #     S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    #     # S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    #     S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'z': S_data[:, 2], 'label': S_data[:, 3]})
    #     print(S_data)
    #     print(S_data.shape)  # [num, 3]
    #     ax = plt.axes(projection='3d')
    #
    #     for index in range(2):  # 假设总共有三个类别，类别的表示为0,1,2
    #         X = S_data.loc[S_data['label'] == index]['x']
    #         Y = S_data.loc[S_data['label'] == index]['y']
    #         Z = S_data.loc[S_data['label'] == index]['z']
    #         # plt.scatter(X, Y, cmap='brg', s=100, c=colors[index], edgecolors=colors[index],alpha=0.65)
    #         ax.scatter3D(X, Y, Z, cmap='brg', s=100,c=colors[index],edgecolors=colors[index],alpha=0.65)
    #
    #         # ax.xticks([])  # 去掉横坐标值
    #         # ax.yticks([])  # 去掉纵坐标值
    #
    #     # plt.title(name, fontsize=32, fontweight='normal', pad=20)
    #
    #
    # # feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
    # a_list = []
    # label_list = []
    # for batch_idx, data in enumerate(loader_graph):
    #     print(data[0][1])
    #     # a = data[1].view([data[0].size()[0],-1])
    #     # a = a[:,:150000]
    #     output,a = gcn_model.model.forward(data[0].to("cuda"),data[1].to("cuda"))
    #     a = a.view([a.size()[0], -1])
    #     a = a[:,:150000].cpu().detach()
    #     for i in range(50):
    #         label_test= data[4]
    # label_test = label_test.numpy()
    # lz = torch.rand(128,1024)
    # feat = a
    #
    # # label_test = np.array(label_test1 + label_test2 + label_test3)
    # # print(label_test)
    # # print(label_test.shape)
    #
    # # fig = plt.figure(figsize=(10, 10))
    #
    # plotlabels(visual(feat), label_test)
    #
    # plt.show()