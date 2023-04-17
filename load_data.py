import torch
import copy
import torch.utils
import numpy as np
from numpy import linalg as la
from sklearn.decomposition import PCA
import dgl
from dgl.data import DGLDataset
import math
import scipy.sparse as sparse
import scipy



# gbl数据处理
class gblData(DGLDataset):

    def __init__(self, data_list):
        super().__init__(name='gcn')
        self.data_list = data_list
        grap_list = []
        label_list = []
        for i in range(len(self.data_list)):
            node_features = torch.from_numpy(self.data_list[i].block_feature).float()
            edges_src = self.data_list[i].edge_src
            edges_dst = self.data_list[i].edge_dst
            graph = dgl.graph((edges_src, edges_dst), num_nodes=self.data_list[i].basicBlock_len)
            graph.ndata['feat'] = node_features
            label_list.append(self.data_list[i].label)
            graph= dgl.add_self_loop(graph)
            grap_list.append(graph)
        self.graph = grap_list
        self.label = label_list

    # def process(self):
    #     grap_list = []
    #     label_list = []
    #     for i in range(len(self.data_list)):
    #         node_features = torch.from_numpy(self.data_list[i].block_feature)
    #         edges_src = self.data_list[i].edge_src
    #         edges_dst = self.data_list[i].edge_dst
    #         graph = dgl.graph((edges_src,edges_dst),num_nodes=self.data_list[i].basicBlock_len)
    #         graph.ndata['feat'] = node_features
    #         label_list.append(self.data_list[i].label)
    #         grap_list.append(graph)
    #     self.graph = grap_list
    #     self.label = label_list

    def __getitem__(self, idx):
        # 将原始数据处理为图、标签和数据集划分的掩码
        return  self.graph[idx], self.label[idx]


    def __len__(self):
        return len(self.data_list)


def collate_dgl(samples):
    # 输入`samples`是一个列表# 每个元素都是一个(图, 标签)
    # 生成graoh，labels两个列表
    graphs, labels = map(list, zip(*samples))  # map函数将第二个参数（一般是数组）中的每一个项，处理为第一个参数的类型。

    # DGL提供了一个dgl.batch()方法，生成batch_graphs.
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

# 图数据获取-pca处理
class GraphData_pca(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.adj_list = []
        self.degree_invert_list = []
        self.block_feature_list = []

        B = len(data_list)
        N_nodes = [self.data[b].basicBlock_len for b in range(B)]
        n_comps = math.floor(sum(N_nodes)/len(N_nodes))

        C = data_list[0].block_feature[0].shape[0]  # 自定义的特征维度
        N_nodes_max = int(np.max(N_nodes))  # 获取最大的节点长度
        x = torch.zeros(N_nodes_max, C)  # 节点特征矩阵
        A = torch.zeros(N_nodes_max, N_nodes_max)  # 邻接矩阵
        print(n_comps)
        # pca = PCA(n_components=n_comps)
        pca = PCA(n_components=7)
        pca_A_list = []
        pca_X_list = []

        for b in range(len(data_list)):
            # 归一化A
            v, Q = la.eig(self.data[b].degree_matrix)  # 求度矩阵的特征值和特征向量
            V = np.diag(v ** (-0.5))
            T = Q * V * la.inv(Q)
            D_a = np.matmul(T,self.data[b].adjacency)
            hat_a = np.matmul(D_a,T)

            x[:N_nodes[b]] = torch.FloatTensor(self.data[b].block_feature)
            A[:N_nodes[b], :N_nodes[b]] = torch.FloatTensor(hat_a)

            A_b_pca = pca.fit_transform(A)
            A_b_pca = A_b_pca.T
            A_b_pca = pca.fit_transform(A_b_pca)
            pca_A_list.append(A_b_pca.T)
            # print(pca.explained_variance_ratio_)
            # sum = 0
            # for i in range(len(pca.explained_variance_ratio_)):
            #     sum += pca.explained_variance_ratio_[i]
            # print(sum)
            x_b_pca = x.T
            x_b_pca = pca.fit_transform(x_b_pca)
            x_b_pca = x_b_pca.T
            pca_X_list.append(x_b_pca)

        self.adj_list = pca_A_list
        self.block_feature_list = pca_X_list

    def __getitem__(self, index):
        return [
                torch.from_numpy(self.block_feature_list[index]).float(),  # 节点特征
                torch.from_numpy(self.adj_list[index]).float(),  # 邻接矩阵
                int(self.data[index].label)  # 标签
                ]

    def __len__(self):
        return len(self.data)

# 图数据-无压缩处理
class GraphData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.adj_list = []
        self.degree_invert_list = []
        self.block_feature_list = []


        for i in range(len(data_list)):
            # 归一化A
            v, Q = la.eig(self.data[i].degree_matrix)  # 求度矩阵的特征值和特征向量
            V = np.diag(v ** (-0.5))
            T = Q * V * la.inv(Q)
            D_a = np.matmul(T,self.data[i].adjacency)
            hat_a = np.matmul(D_a,T)

            self.adj_list.append(hat_a)

            # self.adj_list.append(self.data[i].adjacency)
            # self.degree_invert_list.append(T)
            self.block_feature_list.append(self.data[i].block_feature)

    def __getitem__(self, index):
        return [
                torch.from_numpy(self.block_feature_list[index]).float(),  # 节点特征
                torch.from_numpy(self.adj_list[index]).float(),  # 邻接矩阵
                # torch.from_numpy(self.degree_invert_list[index]).float(),  # 度矩阵
                int(self.data[index].label)  # 标签
                ]

    def __len__(self):
        return len(self.data)

def collate_graph_batch(batch):  # 这里的输入参数便是__getitem__的返回值
    """
            Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
            the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
            Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
            :param batch: [node_features * batch_size, A * batch_size, label * batch_size, id]
            :return: [node_features, A, graph_support, N_nodes, label， ids]
    """
    # print(batch)
    B = len(batch)
    N_nodes = [len(batch[b][1]) for b in range(B)]
    C = batch[0][0].shape[1] # 自定义的特征维度
    N_nodes_max = int(np.max(N_nodes)) # 获取最大的节点长度

    graph_support = torch.zeros(B, N_nodes_max) # mask
    x = torch.zeros(B, N_nodes_max, C) # 节点特征矩阵
    A = torch.zeros(B, N_nodes_max, N_nodes_max) # 邻接矩阵
    # A = torch.full((B, N_nodes_max, N_nodes_max),0.01) # 邻接矩阵
    # D = torch.zeros(B, N_nodes_max, N_nodes_max) # 度矩阵

    for b in range(B):
        x[b, :N_nodes[b]] = batch[b][0]
        A[b, :N_nodes[b], :N_nodes[b]] = batch[b][1]
        # D[b, :N_nodes[b], :N_nodes[b]] = batch[b][2]
        graph_support[b][:N_nodes[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1

    N_nodes = torch.from_numpy(np.array(N_nodes)).long()
    labels = torch.from_numpy(np.array([batch[b][2] for b in range(B)])).long()

    return [x, A,graph_support, N_nodes, labels]
    # return [x, A, graph_support, N_nodes, labels]


# 操作码序列数据
class OpCodeData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return [torch.FloatTensor(np.array(self.data[index].allinstructions_feature).reshape(-1,300)), # 操作码序列
                int(self.data[index].label), # 标签
                ]

    def __len__(self):
        return len(self.data)

def collate_opcode_batch(batch):  # 这里的输入参数便是__getitem__的返回值
    # print(batch)
    B = len(batch)
    seq_list =[len(batch[b][0]) for b in range(B)]
    seq_list_tensor =torch.IntTensor([sl for sl in seq_list])
    C = batch[0][0].shape[1] # 自定义的特征维度
    seq_max = int(np.max(seq_list)) # 获取最大的序列长度

    x = torch.zeros(B, seq_max, C) # opcode序列矩阵
    for b in range(B):
        x[b, :seq_list[b]] = batch[b][0]

    labels = torch.from_numpy(np.array([batch[b][1] for b in range(B)])).long()

    return [x, seq_list_tensor ,labels]


# 专家模式数据
class PatternData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list


    def __getitem__(self, index):
        return [torch.FloatTensor(np.array(self.data[index].pattern[0])),
                torch.FloatTensor(np.array(self.data[index].pattern[1])),
                torch.FloatTensor(np.array(self.data[index].pattern[2])),
                int(self.data[index].label)  # 标签
                ]

    def __len__(self):
        return len(self.data)


# 所有数据融合处理
class MLPData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list


    def __getitem__(self, index):
        return [torch.FloatTensor(self.data[0][index].cpu()), # dgl_gcn_feature
                torch.FloatTensor(self.data[1][index].cpu()), # gru_feature
                torch.FloatTensor(self.data[2][index].cpu()), # pattern_feature
                int(self.data[3][index])  # label
                ]

    def __len__(self):
        return len(self.data[0])

class MLP_AllData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.adj_list = []
        self.degree_invert_list = []
        self.block_feature_list = []

        B = len(data_list)
        N_nodes = [self.data[b].basicBlock_len for b in range(B)]
        n_comps = math.floor(sum(N_nodes) / len(N_nodes))

        C = data_list[0].block_feature[0].shape[0]  # 自定义的特征维度
        N_nodes_max = int(np.max(N_nodes))  # 获取最大的节点长度
        x = torch.zeros(N_nodes_max, C)  # 节点特征矩阵
        A = torch.zeros(N_nodes_max, N_nodes_max)  # 邻接矩阵
        print(n_comps)
        # pca = PCA(n_components=n_comps)
        pca = PCA(n_components=8)
        pca_A_list = []
        pca_X_list = []

        for b in range(len(data_list)):
            # 归一化A
            v, Q = la.eig(self.data[b].degree_matrix)  # 求度矩阵的特征值和特征向量
            V = np.diag(v ** (-0.5))
            T = Q * V * la.inv(Q)
            D_a = np.matmul(T, self.data[b].adjacency)
            hat_a = np.matmul(D_a, T)

            x[:N_nodes[b]] = torch.FloatTensor(self.data[b].block_feature)
            A[:N_nodes[b], :N_nodes[b]] = torch.FloatTensor(hat_a)

            A_b_pca = pca.fit_transform(A)
            A_b_pca = A_b_pca.T
            A_b_pca = pca.fit_transform(A_b_pca)
            pca_A_list.append(A_b_pca.T)
            x_b_pca = x.T
            x_b_pca = pca.fit_transform(x_b_pca)
            x_b_pca = x_b_pca.T
            pca_X_list.append(x_b_pca)

        self.adj_list = pca_A_list
        self.block_feature_list = pca_X_list


    def __getitem__(self, index):
        return [
                torch.from_numpy(self.block_feature_list[index]).float(),  # 节点特征
                torch.from_numpy(self.adj_list[index]).float(),  # 邻接矩阵
                torch.FloatTensor(np.array(self.data[index].pattern[0])), # pattern1
                torch.FloatTensor(np.array(self.data[index].pattern[1])), # pattern2
                torch.FloatTensor(np.array(self.data[index].pattern[2])), # pattern3
                torch.FloatTensor(np.array(self.data[index].allinstructions_feature).reshape(-1, 300)),  # 操作码序列
                int(self.data[index].label)  # label
                ]

    def __len__(self):
        return len(self.data[0])

def collate_all_batch(batch):  # 这里的输入参数便是__getitem__的返回值
    # print(batch)
    B = len(batch)
    seq_list =[len(batch[b][5]) for b in range(B)]
    seq_list_tensor =torch.IntTensor([sl for sl in seq_list])
    C = batch[0][5].shape[1] # 自定义的特征维度
    seq_max = int(np.max(seq_list)) # 获取最大的序列长度

    x_seq = torch.zeros(B, seq_max, C) # opcode序列矩阵
    for b in range(B):
        x_seq[b, :seq_list[b]] = batch[b][5]

    labels = torch.from_numpy(np.array([batch[b][1] for b in range(B)])).long()

    return [batch, x_seq, seq_list_tensor ,labels]

