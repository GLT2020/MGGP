import torch
import torch.nn as nn
from parser1 import parameter_parser
from torch.nn.parameter import Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math
import time
import numpy as np
from sklearn import metrics
import torch.nn.functional as F

args = parameter_parser()
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
print(device)
# hyper-parameters
BATCH_SIZE = 32
LR = 0.01


# GraphConv layers and models
class GraphConv(nn.Module):
    """
    Graph Convolution Layer & Additional tricks (power of adjacency matrix and weighted self connections)
    n_relations: number of relation types (adjacency matrices)
    """

    def __init__(self, in_features, out_features, n_relations=1,
                 activation=None, adj_sq=False, scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features * n_relations, out_features=out_features)
        self.n_relations = n_relations
        self.activation = activation
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity

    def laplacian_batch(self, A): # 计算文章的公式（A^2 + I）
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # 矩阵乘法 use A^2 to increase graph connectivity
        I = torch.eye(N).unsqueeze(0).to(args.device)
        a = I.shape
        if self.scale_identity:
            I = 2 * I  # increase weight of self connections
        # A_hat = A
        # add I represents self connections of nodes
        A_hat = I + A
        # D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)  # Adjacent matrix normalization
        # L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        L = A_hat  # remove D_hat
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []
        for rel in range(self.n_relations):
            x_hat.append(torch.bmm(self.laplacian_batch(A[:, :, :, rel]), x))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)
        x = x * mask
        # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc
        # which affects node embeddings in the following layers
        if self.activation is not None:
            x = self.activation(x)
        return x, A, mask


class GCN_MODIFY(nn.Module):
    """
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    """
    def __init__(self, in_features, out_features, filters=args.filters,
                 n_hidden=args.n_hidden, dropout=args.dropout, adj_sq=False, scale_identity=False):
        super(GCN_MODIFY, self).__init__()
        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f, activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq, scale_identity=scale_identity) for layer, f in enumerate(filters)]))
        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x


class GCN_modify():
    def __init__(self, input_dim, output_dim,filters,n_hidden,dropout,adj_sq,scale_identity):
        super(GCN_modify, self).__init__()
        self.model = GCN_MODIFY(input_dim,output_dim,filters=filters,n_hidden=n_hidden,dropout=dropout,adj_sq=adj_sq,scale_identity=scale_identity).to(device)
        self.state_dim = input_dim
        self.result_dim = output_dim

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
        self.loss_fn = F.cross_entropy

    def train(self,train_loader,epoch):
        self.scheduler.step()  # 调整lr
        self.model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):  # 读取的data:节点特征,邻接矩阵，度矩阵，图的标签、
            for i in range(len(data)):
                data[i] = data[i].to(device)
            self.optimizer.zero_grad()  # 梯度置零
            output = self.model(data[0], data[1],data[2])  # when model is gcn_origin or gat, use this
            # output = self.model(data)  # when model is gcn_modify, use this
            loss = self.loss_fn(output, data[5])
            loss.backward()
            self.optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        # torch.save(self.model, 'MGCE.pth')

    def test(self,test_loader,epoch):
        self.model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
        accuracy, recall, precision, F1 = 0, 0, 0, 0
        fn_list = []  # Store the contract id corresponding to the fn
        fp_list = []  # Store the contract id corresponding to the fp

        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            # output = model(training_data[0], training_data[1])  # when model is gcn_origin or gat, use this
            output = self.model(data)  # when model is gcn_modify, use this
            loss = self.loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            for k in range(len(pred)):  # view_as是确保比较的两个向量维度一致
                if (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 1):
                    # print(pred.view_as(data[4])[k]) # tensor(1)
                    # print(np.array(pred.view_as(data[4])[k]).tolist())
                    # TP predict == 1 & label == 1
                    tp += 1
                    continue
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 0):
                    # TN predict == 0 & label == 0
                    tn += 1
                    continue
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 1):
                    # FN predict == 0 & label == 1
                    fn += 1
                    fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 0):
                    # FP predict == 1 & label == 0
                    fp += 1
                    fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue

            accuracy += metrics.accuracy_score(data[4], pred.view_as(data[4]))
            recall += metrics.recall_score(data[4], pred.view_as(data[4]))
            precision += metrics.precision_score(data[4], pred.view_as(data[4]))
            F1 += metrics.f1_score(data[4], pred.view_as(data[4]))

        print(tp, fp, tn, fn)
        accuracy = 100. * accuracy / count
        recall = 100. * recall / count
        precision = 100. * precision / count
        F1 = 100. * F1 / count
        FPR = fp / (fp + tn)

        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.2f}%), Recall: ({:.2f}%), Precision: ({:.2f}%), '
            'F1-Score: ({:.2f}%), FPR: ({:.2f}%)  sec/iter: {:.4f}\n'.format(
                epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, FPR,
                (time.time() - start) / len(test_loader))
        )

        print("fn_list(predict == 0 & label == 1):", fn_list)
        print("fp_list(predict == 1 & label == 0):", fp_list)
        print()

        return accuracy, recall, precision, F1, FPR
