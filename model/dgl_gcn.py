import dgl
import dgl.function as fn
from dgl.utils import check_eq_shape
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution
from parser1 import parameter_parser
from torch.nn.parameter import Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math
import time
import numpy as np
from sklearn import metrics
from dgl import DGLGraph

args = parameter_parser()
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
LR = 0.01

from dgl.utils import expand_as_pair


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN1(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN1, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.gcn_msg = fn.copy_u(u='h', out='m')
        self.gcn_reduce = fn.sum(msg='m', out='h')

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.gcn_msg, self.gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, dropout):
        super(Net, self).__init__()
        self.gcn1 = GCN1(input_dim, 128, F.relu)
        self.gcn2 = GCN1(128, hidden_dim, F.relu)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x_out = self.gcn2(g, x)
        x = self.fc1(x_out)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x, x_out


# 定义模型
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes,dropout):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, hidden_size)
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_size, 2)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32,2)

        self.relu = nn.ReLU()

    def forward(self, g, inputs):
        x = inputs
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        x = self.relu(x)
        g.ndata['h'] = x
        # 以平均值来代表图
        hg = dgl.mean_nodes(g, 'h')
        x = self.fc1(hg)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.fc3(x)
        return x, hg



class dgl_GCN_normal():
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(dgl_GCN_normal, self).__init__()
        self.model = GCN(input_dim, hidden_dim, output_dim, args.dropout).to(device)
        self.state_dim = input_dim
        self.result_dim = output_dim

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self,train_loader,epoch):
        self.model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        pre_loss = 0.9
        for batch_idx,(batched_graph, labels) in enumerate(train_loader):
            # for i in range(len(batched_graph)):
            #     graph = batched_graph[i].to(device)
            #     feats = batched_graph[i].ndata['feat'].to(device)
            graph = batched_graph.to(device)
            feats = batched_graph.ndata['feat'].to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()  # 梯度置零
            output, gcn_output = self.model(graph, feats)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 调整lr
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if loss < pre_loss:
                torch.save(self.model, '/model/pth/dgl_GCN_origin_D2.pth')
                pre_loss = loss

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        return  train_loss / n_samples
        # torch.save(self.model, 'dgl_GCN_origin.pth')

    def test(self,test_loader,epoch):
        self.model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
        accuracy, recall, precision, F1 = 0, 0, 0, 0
        fn_list = []  # Store the contract id corresponding to the fn
        fp_list = []  # Store the contract id corresponding to the fp

        for batch_idx,(batched_graph, labels)in enumerate(test_loader):
            graph = batched_graph.to(device)
            feats = batched_graph.ndata['feat'].to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()  # 梯度置零
            output, output_hidden = self.model(graph, feats)
            loss = self.loss_fn(output, labels)
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            for k in range(len(pred)):  # view_as是确保比较的两个向量维度一致，PCA_loader下data[2]否则[4]
                if (np.array(pred.view_as(labels)[k]).tolist() == 1) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 1):
                    # print(pred.view_as(data[4])[k]) # tensor(1)
                    # print(np.array(pred.view_as(data[4])[k]).tolist())
                    # TP predict == 1 & label == 1
                    tp += 1
                    continue
                elif (np.array(pred.view_as(labels)[k]).tolist() == 0) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 0):
                    # TN predict == 0 & label == 0
                    tn += 1
                    continue
                elif (np.array(pred.view_as(labels)[k]).tolist() == 0) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 1):
                    # FN predict == 0 & label == 1
                    fn += 1
                    # fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(labels)[k]).tolist() == 1) & (
                        np.array(labels.detach().cpu()[k]).tolist() == 0):
                    # FP predict == 1 & label == 0
                    fp += 1
                    # fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue

            accuracy += metrics.accuracy_score(labels.cpu(), pred.view_as(labels))
            recall += metrics.recall_score(labels.cpu(), pred.view_as(labels))
            precision += metrics.precision_score(labels.cpu(), pred.view_as(labels))
            F1 += metrics.f1_score(labels.cpu(), pred.view_as(labels))

        print(tp, fp, tn, fn)
        accuracy = 100. * accuracy / count
        recall = 100. * recall / count
        precision = 100. * precision / count
        F1 = 100. * F1 / count
        FPR = fp / (fp + tn)
        # FPR = 0

        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.2f}%), Recall: ({:.2f}%), Precision: ({:.2f}%), '
            'F1-Score: ({:.2f}%), FPR: ({:.2f}%)  sec/iter: {:.4f}\n'.format(
                epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, FPR,
                (time.time() - start) / len(test_loader))
        )

        # print("fn_list(predict == 0 & label == 1):", fn_list)
        # print("fp_list(predict == 1 & label == 0):", fp_list)
        # print()

        return accuracy, recall, precision, F1, FPR

    def get_hidden(self,dataset):
        self.model.eval()
        for batch_idx, (batched_graph, labels)  in enumerate(dataset):  # 读取的data:节点特征、边矩阵、图的标签、图的id
            graph = batched_graph.to(device)
            feats = batched_graph.ndata['feat'].to(device)
            labels = labels.to(device)
            output, output_hidden = self.model(graph, feats)
            return [output_hidden, labels]

