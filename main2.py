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
    if args.mode == 'test':
        # path = "./compile/bytecode_dict_reentrancy_test.json"
        path = "./compile/bytecode_dict.json"
    elif args.mode == 'train':
        path = "./compile/bytecode_dict_reentrancy_train.json"
        # path = "./compile/bytecode_dict.json"
    with open(path, "r") as f:
        for line in f:
            try:
                info_data_train.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    return info_data_train

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
    bytecode_dict_list_train= read_bytecode()
    # print(bytecode_dict_list)
    for i in range(len(bytecode_dict_list_train)):
        bytecode = bytecode_dict_list_train[i]
        cfg = CFG(bytecode['runtime_bytecode'])
        cfg_feature = CFGFeature(key=bytecode['name'],cfg=cfg,label=bytecode['label'],pattern=bytecode['pattern'])
        # print(str(cfg_feature.basicBlock[0].instructions[0]).split(" "))
        cfg_feature_list_train.append(cfg_feature)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rnd_state = np.random.RandomState(args.seed)

    if args.model == 'all':
        datareader_graph_gbl_train = gblData(cfg_feature_list_train)
        datareader_graph_train = GraphData_pca(cfg_feature_list_train)
        datareader_opcode_train = OpCodeData(cfg_feature_list_train)
        datareader_pattern_train = PatternData(cfg_feature_list_train)
    elif args.model == 'dgl':
        # datareader_graph = GraphData(cfg_feature_list_train)
        datareader_graph_gbl_train = gblData(cfg_feature_list_train)
    elif args.model == 'gcn':
        datareader_graph_train = GraphData_pca(cfg_feature_list_train)
    elif args.model == 'gru':
        datareader_opcode_train = OpCodeData(cfg_feature_list_train)
    elif args.model == 'pattern':
        datareader_pattern_train = PatternData(cfg_feature_list_train)

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
        loader_graph_train = DataLoader(dataset=datareader_graph_train,batch_size=len(cfg_feature_list_train))
        loader_opcode_train = DataLoader(dataset=datareader_opcode_train,batch_size=len(cfg_feature_list_train),collate_fn=collate_opcode_batch)
        loader_pattern_train = DataLoader(dataset=datareader_pattern_train,batch_size=len(cfg_feature_list_train))

        out_feature_train_list = []
        # get the feature after dgl_gcn
        # dgl_gcn_model.model = torch.load('dgl_GCN_origin.pth')
        # gcn_feature_train = dgl_gcn_model.get_hidden(loader_graph_gbl_train)
        # out_feature_train_list.append(gcn_feature_train[0].detach())
        # gcn_feature_test = dgl_gcn_model.get_hidden(loader_graph_gbl_test)
        # out_feature_test_list.append(gcn_feature_test[0].detach())

        # get the feature after gcn
        gcn_model.model = torch.load('model/pth/GCN_origin.pth')
        gcn_feature_train = gcn_model.get_hidden(loader_graph_train)
        out_feature_train_list.append(gcn_feature_train[0].detach())

        gru_model.model = torch.load('model/pth/gru.pth')
        gru_feature_train = gru_model.get_hidden(loader_opcode_train)
        out_feature_train_list.append(gru_feature_train[0].detach())

        pattern_model.model = torch.load('model/pth/pattern.pth')
        pattern_feature_train = pattern_model.get_hidden(loader_pattern_train)
        out_feature_train_list.append(pattern_feature_train[0].detach())  # pattern的特征
        out_feature_train_list.append(pattern_feature_train[1].detach())  # 标签

        datareader_out_train = MLPData(out_feature_train_list)
        loader_out_train = DataLoader(dataset=datareader_out_train, batch_size=25, shuffle=True, drop_last=True)
    elif args.mode == 'train':
        if args.model == 'dgl':
            # gbl数据集，使用for g, labels in dataloader：
            loader_graph_gbl_train = GraphDataLoader(dataset=datareader_graph_gbl_train, batch_size=25, shuffle=True, drop_last= True, collate_fn=collate_dgl)

        elif args.model == 'gcn':
            loader_graph_train = DataLoader(dataset=datareader_graph_train, batch_size=25, shuffle=True, drop_last=True)

        elif args.model == 'gru':
            loader_opcode_train = DataLoader(dataset=datareader_opcode_train, batch_size=25, shuffle=True, drop_last=True,collate_fn=collate_opcode_batch)

        elif args.model == 'pattern':
            loader_pattern_train = DataLoader(dataset=datareader_pattern_train, batch_size=25,shuffle=True,drop_last=True)
    elif args.mode == 'test':
        if args.model == 'dgl':
            loader_graph_gbl_train = GraphDataLoader(dataset=datareader_graph_gbl_train,
                                                     batch_size=len(cfg_feature_list_train), collate_fn=collate_dgl)
        elif args.model == 'gcn':
            loader_graph_train = DataLoader(dataset=datareader_graph_train, batch_size=len(cfg_feature_list_train))
        elif args.model == 'gru':
            loader_opcode_train = DataLoader(dataset=datareader_opcode_train, batch_size=len(cfg_feature_list_train),
                                             collate_fn=collate_opcode_batch)
        elif args.model == 'pattern':
            loader_pattern_train = DataLoader(dataset=datareader_pattern_train, batch_size=len(cfg_feature_list_train))


    # gcn_model = GCN_modify(args.node_input_dim,2,filters=args.filters,n_hidden=args.n_hidden,dropout=args.dropout)
    # 训练模型
    if args.mode == 'train' and args.model != 'all':
        loss_list = []
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
        plt.plot(x,loss_list)
        plt.show()

    elif args.mode == 'train' and args.model == 'all':
        for i in range(args.epochs):
            output_model.train(loader_out_train,i)

    # test
    if args.model == 'gcn':
        print("GCN Result:\n")
        gcn_model.model = torch.load('model/pth/GCN_origin.pth')
        gcn_model.test(loader_graph_train, i)
    elif args.model == 'dgl':
        print("DGL Result:\n")
        dgl_gcn_model.model = torch.load('model/pth/dgl_GCN_origin.pth')
        dgl_gcn_model.test(loader_graph_gbl_train, i)
    elif args.model == 'gru':
        print("GRU Result:\n")
        gru_model.model = torch.load('model/pth/gru.pth')
        gru_model.test(loader_opcode_train,i)
    elif args.model == 'pattern':
        print("Pattern Result:\n")
        pattern_model.model = torch.load('model/pth/pattern_D2.pth')
        pattern_model.test(loader_pattern_train, i)
    elif args.model == 'all':
        print("ALL Result:\n")
        output_model.model = torch.load('model/pth/output.pth')
        output_model.test(loader_out_train, i)
