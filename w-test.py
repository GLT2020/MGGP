"""
Author: yida
Time is: 2021/11/17 15:45
this Code:
1.实现<基于自适应特征融合与转换的小样本图像分类>中的自适应特征处理模块AFP
2.演示: nn.Parameter的使用
"""
import torch
import torch.nn as nn


class AFP(nn.Module):
    def __init__(self):
        super(AFP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),  # 1.最大池化分支,原文设置的尺寸大小为3, 未说明stride以及padding, 为与原图大小保持一致, 使用(3, 1, 1)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1),  # 2.平均池化分支, 原文设置的池化尺寸为2, 未说明stride以及padding, 为与原图大小保持一致, 使用(3, 1, 1)
        )

        self.branch3_1 = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.Conv2d(1, 1, 3, padding=1),  # 3_1分支, 先用1×1卷积压缩通道维数, 然后使用两个3×3卷积进行特征提取, 由于通道数为3//2, 此时输出维度设为1
            nn.Conv2d(1, 1, 3, padding=1),
        )

        self.branch3_2 = nn.Sequential(
            nn.Conv2d(3, 2, 1),  # 3_2分支, 由于1×1卷积压缩通道维数减半, 但是这儿维度为3, 上面用的1, 所以这儿输出维度设为2
            nn.Conv2d(2, 2, 3, padding=1)
        )
        # 注意力机制
        self.branch_SE = SEblock(channel=3)

        # 初始化可学习权重系数
        # nn.Parameter 初始化的权重, 如果作用到网络中的话, 那么它会被添加到优化器更新的参数中, 优化器更新的时候会纠正Parameter的值, 使得向损失函数最小化的方向优化

        self.w = nn.Parameter(torch.ones(4))  # 4个分支, 每个分支设置一个自适应学习权重, 初始化为1, nn.Parameter需放入Tensor类型的数据
        # self.w = nn.Parameter(torch.Tensor([0.5, 0.25, 0.15, 0.1]), requires_grad=False)  # 设置固定的权重系数, 不用归一化, 直接乘过去

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        b3_1 = self.branch3_1(x)
        b3_2 = self.branch3_2(x)
        b3_Combine = torch.cat((b3_1, b3_2), dim=1)
        b3 = self.branch_SE(b3_Combine)

        b4 = x

        print("b1:", b1.shape)
        print("b2:", b2.shape)
        print("b3:", b3.shape)
        print("b4:", b4.shape)

        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))

        # 多特征融合
        x_out = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4
        print("特征融合结果:", x_out.shape)
        return x_out


class SEblock(nn.Module):  # 注意力机制模块
    def __init__(self, channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        # 全连接层得到权重
        weight = self.fc(branch)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        scale = weight * x
        return scale


if __name__ == '__main__':
    model = AFP()
    print(model)
    inputs = torch.randn(10, 3, 84, 84)
    print("输入维度为: ", inputs.shape)
    outputs = model(inputs)
    print("输出维度为: ", outputs.shape)

    # 查看nn.Parameter中值的变化, 训练网络时, 更新优化器之后, 可以循环输出, 查看权重变化
    for name, p in model.named_parameters():
        if name == 'w':
            print("特征权重: ", name)
            w0 = (torch.exp(p[0]) / torch.sum(torch.exp(p))).item()
            w1 = (torch.exp(p[1]) / torch.sum(torch.exp(p))).item()
            w2 = (torch.exp(p[2]) / torch.sum(torch.exp(p))).item()
            w3 = (torch.exp(p[3]) / torch.sum(torch.exp(p))).item()
            print(w0, w1, w2, w3)
