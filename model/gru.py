import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from parser1 import parameter_parser
import torch.optim.lr_scheduler as lr_scheduler
import time
from torch.nn.utils.rnn import pack_padded_sequence # https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html
import math
from sklearn import metrics

args = parameter_parser()
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
print(device)

class RNNClassifier(nn.Module):
    """
    这里的bidirectional就是GRU是不是双向的，双向的意思就是既考虑过去的影响，也考虑未来的影响（如一个句子）
    具体而言：正向hf_n=w[hf_{n-1}, x_n]^T,反向hb_0,最后的h_n=[hb_0, hf_n],方括号里的逗号表示concat。
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1  # 双向2、单向1

        # self.embedding = nn.Embedding(input_size, hidden_size)
        '''
        input_dim:输入特征维度
        hidden_dim: 输出特征维度，没有特殊变化相当于out
        num_layers:网络层数
        batch_first:默认是False即[序列长度seq,批大小batch,特征维度feature]；若true，[batch，seq，feature]
        '''
        self.gru = nn.GRU(input_size, hidden_size, n_layers,  # 输入维度、输出维度、层数、bidirectional用来说明是单向还是双向
                          bidirectional=bidirectional,batch_first=False)
        self.fc1 = nn.Linear(hidden_size * self.n_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def __init__hidden(self, batch_size):  # 工具函数，作用是创建初始的隐藏层h0
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self, input, seq_lengths):
        # input shape:B * S -> S * B
        input = input.transpose(0, 1)
        batch_size = input.size(1)

        hidden = self.__init__hidden(batch_size)  # 隐藏层h0
        # embedding = self.embedding(input)

        # pack them up
        gru_input = pack_padded_sequence(input, seq_lengths.cpu(),enforce_sorted=False)  # 填充了可能有很多的0，所以为了提速，将每个序列以及序列的长度给出

        output, hidden = self.gru(gru_input, hidden)  # 只需要hidden
        if self.n_directions == 2:  # 双向的，则需要拼接起来
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]  # 单向的，则不用处理
        # 最后来个全连接层,确保层想要的维度（类别数）
        fc_output = self.fc1(hidden_cat)
        fc = self.relu(fc_output)
        fc = self.fc2(fc)
        return fc, fc_output

class GRU_normal():
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU_normal, self).__init__()
        self.model = RNNClassifier(input_dim, hidden_dim, output_dim).to(device)
        self.state_dim = input_dim
        self.result_dim = output_dim

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
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
            output, hidden = self.model(data[0], data[1])
            loss = self.loss_fn(output, data[2])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 调整lr
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if loss < pre_loss:
                torch.save(self.model, '/model/pth/gru_D2.pth')
                pre_loss = loss

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        return train_loss / n_samples
        # torch.save(self.model, 'gru.pth')

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
            output,hidden = self.model(data[0], data[1])
            loss = self.loss_fn(output, data[2])
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            for k in range(len(pred)):  # view_as是确保比较的两个向量维度一致
                if (np.array(pred.view_as(data[2])[k]).tolist() == 1) & (
                        np.array(data[2].detach().cpu()[k]).tolist() == 1):
                    # print(pred.view_as(data[2])[k]) # tensor(1)
                    # print(np.array(pred.view_as(data[2])[k]).tolist())
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

        return [hidden, data[2]]

