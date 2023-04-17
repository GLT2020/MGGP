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
print(device)

class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(MLP, self).__init__()
        self.ex_layer1 = nn.Linear(in_dim, 32)
        self.ex_layer2 = nn.Linear(in_dim, 32)
        self.ex_layer3 = nn.Linear(in_dim, 32)
        self.layer1 = nn.Linear(96, n_hidden)
        self.layer2 = nn.Linear(n_hidden, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x1,x2,x3):
        x1 = self.ex_layer1(x1)
        x2 = self.ex_layer1(x2)
        x3 = self.ex_layer1(x3)
        x =  torch.cat((x1,x2,x3),1)
        x_out = self.layer1(x)
        x = self.relu(x_out)
        x = self.layer2(x)
        return x,x_out


class Pattern_normal():
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Pattern_normal, self).__init__()
        self.model = MLP(input_dim, hidden_dim, output_dim).to(device)
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
            output, hidden = self.model(data[0],data[1],data[2])
            loss = self.loss_fn(output, data[3])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 调整lr
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if loss < pre_loss:
                torch.save(self.model, '/model/pth/pattern_D2.pth')
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
        accuracy, recall, precision, F1 = 0, 0, 0, 0
        fn_list = []  # Store the contract id corresponding to the fn
        fp_list = []  # Store the contract id corresponding to the fp

        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            output, hidden = self.model(data[0],data[1],data[2])
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

            accuracy += metrics.accuracy_score(data[3].cpu(), pred.view_as(data[3]))
            recall += metrics.recall_score(data[3].cpu(), pred.view_as(data[3]))
            precision += metrics.precision_score(data[3].cpu(), pred.view_as(data[3]))
            F1 += metrics.f1_score(data[3].cpu(), pred.view_as(data[3]))

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
        output, hidden = self.model(data[0], data[1], data[2])
        return [hidden, data[3]]