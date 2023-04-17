import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution
from parser1 import parameter_parser
from torch.nn.parameter import Parameter
import torch.optim.lr_scheduler as lr_scheduler
import scipy as sp
import math
import time
import numpy as np
from sklearn import metrics

args = parameter_parser()
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
print(device)
# hyper-parameters
BATCH_SIZE = 32
LR = 0.01


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        output = []
        batch = input.shape[0]
        for i in range(batch):
            if len(input.shape) == 3:
                support = torch.mm(input[i], self.weight)
                out = torch.spmm(adj[i], support)
                output.append(out)
            else:
                support = torch.mm(input, self.weight)
                out = torch.spmm(adj, support)
                output.append(out)
        # output = torch.mean(output, dim=0, keepdim=True)
        output = torch.tensor([item.cpu().detach().numpy() for item in output]).cuda()
        if self.bias is not None:
            output = output + self.bias
            return output
        else:
            return output



    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCN_ORIGIN(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out, dropout):
        super(GCN_ORIGIN, self).__init__()
        self.gc1 = GraphConvolution(n_feature, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_hidden)
        self.dropout = dropout
        self.fc1 = nn.Linear(n_hidden,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax()


    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x_out = self.fc1(x)
        x = self.relu(x_out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x,x_out


class GCN_normal():
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_normal, self).__init__()
        self.model = GCN_ORIGIN(input_dim, hidden_dim, output_dim, args.dropout).to(device)
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
        for batch_idx, data in enumerate(train_loader):  # 读取的data:节点特征、边矩阵、图的标签、图的id
            for i in range(len(data)):
                data[i] = data[i].to(device)
            self.optimizer.zero_grad()  # 梯度置零
            output,gcn_output = self.model(data[0], data[1])
            # loss = self.loss_fn(output, data[4])
            loss = self.loss_fn(output, data[2])      # PCA_loader使用这个
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 调整lr
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if loss < pre_loss:
                torch.save(self.model, 'GCN_origin.pth')
                pre_loss = loss

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        # torch.save(self.model, 'GCN_origin.pth')
        return train_loss / n_samples

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
            output,gcn_output = self.model(data[0], data[1])  # when model is gcn_origin or gat, use this
            # output = self.model(data)  # when model is gcn_modify, use this
            # loss = self.loss_fn(output, data[4])
            loss = self.loss_fn(output, data[2]) # pca_loader使用这个
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            for k in range(len(pred)):  # view_as是确保比较的两个向量维度一致，PCA_loader下data[2]否则[4]
                if (np.array(pred.view_as(data[2])[k]).tolist() == 1) & (
                        np.array(data[2].detach().cpu()[k]).tolist() == 1):
                    # print(pred.view_as(data[4])[k]) # tensor(1)
                    # print(np.array(pred.view_as(data[4])[k]).tolist())
                    # TP predict == 1 & label == 1
                    tp += 1
                    continue
                elif (np.array(pred.view_as(data[2])[k]).tolist() == 0) & (
                        np.array(data[2].detach().cpu()[k]).tolist() == 0):
                    # TN predict == 0 & label == 0
                    tn += 1
                    continue
                elif (np.array(pred.view_as(data[2])[k]).tolist() == 0) & (
                        np.array(data[2].detach().cpu()[k]).tolist() == 1):
                    # FN predict == 0 & label == 1
                    fn += 1
                    # fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(data[2])[k]).tolist() == 1) & (
                        np.array(data[2].detach().cpu()[k]).tolist() == 0):
                    # FP predict == 1 & label == 0
                    fp += 1
                    # fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue

            accuracy += metrics.accuracy_score(data[2].cpu(), pred.view_as(data[2]))
            recall += metrics.recall_score(data[2].cpu(), pred.view_as(data[2]))
            precision += metrics.precision_score(data[2].cpu(), pred.view_as(data[2]))
            F1 += metrics.f1_score(data[2].cpu(), pred.view_as(data[2]))

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
        for batch_idx, data in enumerate(dataset):  # 读取的data:节点特征、边矩阵、图的标签、图的id
            for i in range(len(data)):
                data[i] = data[i].to(device)
        output, hidden = self.model(data[0], data[1])
        label_list = data[2]
        return [hidden, label_list]
