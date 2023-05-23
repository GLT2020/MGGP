from torch import nn
from parser1 import parameter_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from parser1 import parameter_parser
import torch.optim.lr_scheduler as lr_scheduler
import time
import numpy as np
from sklearn import metrics
"""
The simple fc layer
"""
args = parameter_parser()
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)
# print(device)

PER = args.D

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        # # 自适应权重设置
        # self.w = nn.Parameter(torch.Tensor(3))
        # nn.init.uniform_(self.w.data)
        # # self-attention
        # self.query = nn.Linear(in_dim, in_dim)
        # self.key = nn.Linear(in_dim, in_dim)
        # self.value = nn.Linear(in_dim, in_dim)
        # self.softmax = nn.Softmax(dim=-1)
        # self.fc = nn.Linear(in_dim,32)
        # 全连接输出
        self.layer1 = nn.Linear(in_dim*3, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x1,x2,x3):

        # 加入自适应权重输出
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # x = x1 * w1 + x2 * w2 + x3 * w3

        # 加入self-atttention
        # x = torch.stack([x1, x2, x3], dim=0)
        # q = self.query(x)
        # k = self.key(x)
        # v = self.value(x)
        # scores = torch.matmul(q, k.transpose(-2, -1))  # 输出形状: (10, 10, 10)
        # scores = scores / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))  # 输出形状: (10, 10, 10)
        # attention_weights = self.softmax(scores)  # 输出形状: (10, 10, 10)
        # attention_output = torch.matmul(attention_weights, v)  # 输出形状: (10, 10, 64)
        # attention_output = attention_output.mean(dim=0)
        # x = self.fc(attention_output)
        # x = self.layer3(x)

        # 全连接输出
        x = torch.cat((x1, x2, x3), 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


class Out_normal():
    def __init__(self, input_dim, output_dim):
        super(Out_normal, self).__init__()
        self.model = MLP(input_dim, output_dim).to(device)
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
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            self.optimizer.zero_grad()  # 梯度置零
            output= self.model(data[0],data[1],data[2])
            loss = self.loss_fn(output, data[3])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 调整lr
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            # 存储模型
            if loss < pre_loss:
                # torch.save(self.model, 'model/pth/output_D2.pth')
                # torch.save(self.model, 'model/pth/output_D2_address.pth')
                torch.save(self.model, 'model/pth/output_D2_'+ PER +'.pth')
                pre_loss = loss

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        return train_loss / n_samples

    def test(self,test_loader,epoch):
        self.model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
        accuracy, recall, precision, F1 = 0.0, 0.0, 0.0, 0.0
        fn_list = []  # Store the contract id corresponding to the fn
        fp_list = []  # Store the contract id corresponding to the fp

        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            output = self.model(data[0],data[1],data[2])
            loss = self.loss_fn(output, data[3])
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            for k in range(len(pred)):  # view_as是确保比较的两个向量维度一致
                if (np.array(pred.view_as(data[3])[k]).tolist() == 1) & (
                        np.array(data[3].detach().cpu()[k]).tolist() == 1):
                    # print(pred.view_as(data[2])[k]) # tensor(1)
                    # print(np.array(pred.view_as(data[2])[k]).tolist())
                    # TP predict == 1 & label == 1
                    tp += 1
                    continue
                elif (np.array(pred.view_as(data[3])[k]).tolist() == 0) & (
                        np.array(data[3].detach().cpu()[k]).tolist() == 0):
                    # TN predict == 0 & label == 0
                    tn += 1
                    continue
                elif (np.array(pred.view_as(data[3])[k]).tolist() == 0) & (
                        np.array(data[3].detach().cpu()[k]).tolist() == 1):
                    # FN predict == 0 & label == 1
                    fn += 1
                    # fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(data[3])[k]).tolist() == 1) & (
                        np.array(data[3].detach().cpu()[k]).tolist() == 0):
                    # FP predict == 1 & label == 0
                    fp += 1
                    # fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue

            # accuracy += metrics.accuracy_score(data[3].cpu(), pred.view_as(data[3]))
            # recall += metrics.recall_score(data[3].cpu(), pred.view_as(data[3]))
            # precision += metrics.precision_score(data[3].cpu(), pred.view_as(data[3]))
            # F1 += metrics.f1_score(data[3].cpu(), pred.view_as(data[3]))

        print(tp, fp, tn, fn)
        # accuracy = 100. * accuracy / count
        # recall = 100. * recall / count
        # precision = 100. * precision / count
        # F1 = 100. * F1 / count
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        F1 = 2*precision*recall/(precision+recall)
        FPR = fp / (fp + tn)
        # FPR = 0
        print("all:", epoch + 1)
        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.4f}%), Recall: ({:.4f}%), Precision: ({:.4f}%), '
            'F1-Score: ({:.4f}%), FPR: ({:.4f}%)  sec/iter: {:.4f}\n'.format(
                epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, FPR,
                (time.time() - start) / len(test_loader))
        )


        # print("fn_list(predict == 0 & label == 1):", fn_list)
        # print("fp_list(predict == 1 & label == 0):", fp_list)
        # print()

        return [tp, fp, tn, fn, accuracy, recall, precision, F1, FPR]
