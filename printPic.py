import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# 读取字节码列表
def read_bytecode(PER):
    info_data = []
    # 混合的数据集按比例切割
    path = "./result/"+ PER +".json"
    print(path)
    with open(path, "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    return info_data

def get_ave_maxmin(data:dict):
    acc = []
    for i in range(len(data)):
        acc.append(data[i]['accuracy'])
    mean = np.mean(acc)  # 平均值
    max = np.max(acc) - mean  # 最大值
    min = mean - np.min(acc)  # 最小值
    print("mean:",mean,";max:",np.max(acc),";min:",np.min(acc), ";Maxerr:" ,max, ";Minerr:", min)
    print("------------------")
    return np.array([mean,max,min,np.max(acc),np.min(acc)])

# 画不同比例下的准确率
def print_percentage():
    path = ["D19", "D28", "D37", "D46", "D55", "D64", "D73", "D82", "D91"]
    all_data = []
    y = []
    yerr = np.zeros((2, 9))
    all = []
    for n in range(len(path)):
        path[n] = path[n] +"/all"
    for n in range(len(path)):
        info = read_bytecode(path[n])
        temp = get_ave_maxmin(info)
        all.append(temp[[0, 3, 4]])
        y.append(temp[0])
        yerr[0][n] = temp[2]
        yerr[1][n] = temp[1]

    x = np.arange(9)
    # yerr =np.array(yerr).transpose()

    plt.figure(figsize=(10, 8))  # 设置图片大小800*600
    plt.bar(x, y, color='#40d0a8', width=0.5, label='平均值')  # 柱状图
    plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='black', capsize=10, label='最大值与最小值')  # 误差线

    # 设置图形属性
    plt.xticks(x, ["D19", "D28", "D37", "D46", "D55", "D64", "D73", "D82", "D91"], fontsize=20)  # 设置x轴刻度标签
    plt.ylim(0.7, 1)
    plt.yticks(fontsize=20)
    plt.xlabel('Train/Test', fontsize=26)  # 设置x轴标签
    plt.ylabel('Accuracy', fontsize=28)  # 设置y轴标签
    # plt.title('')  # 设置标题
    plt.grid(True)  # 是否现在网格线
    plt.rcParams.update({'font.size': 20, 'font.sans-serif': 'SimHei'})  # 设置图例文中大小
    plt.legend(loc="upper center")  # 显示图例和设置图例位置

    # 显示图形
    plt.show()

# 画all和double的对比
def print_all_double():
    path = ["D37", "D46", "D55", "D64"]
    file = ["/all", "/double_gcnp", "/double_grup", "/double_gg"]
    y = np.zeros(16)
    yerr = np.zeros((2, 16))
    yerr_all = np.zeros((2, 4))
    yerr_gcnp = np.zeros((2, 4))
    yerr_grup = np.zeros((2, 4))
    yerr_gg = np.zeros((2, 4))
    all = np.zeros(4)
    gcnp = np.zeros(4)
    grup = np.zeros(4)
    gg = np.zeros(4)
    i = 0
    for f in range(len(file)):
        for n in range(len(path)):
            info = read_bytecode(path[n] + file[f])
            temp = get_ave_maxmin(info)
            # all.append(temp[[0, 3, 4]])
            y[i] = temp[0]
            yerr[0][i] = temp[2]
            yerr[1][i] = temp[1]
            i += 1
    yerr_all[:, 0:4] = yerr[:, 0:4]
    yerr_gcnp[:, 0:4] = yerr[:, 4:8]
    yerr_grup[:, 0:4] = yerr[:, 8:12]
    yerr_gg[:, 0:4] = yerr[:, 12:16]
    all[:] = y[0:4]
    gcnp[:] = y[4:8]
    grup[:] = y[8:12]
    gg[:] = y[12:16]

    x = np.arange(len(path))  # x轴刻度标签位置
    width = 0.2  # 柱子的宽度
    err_attri = {"elinewidth": 2, "ecolor": 'black', "capsize": 10}
    plt.figure(figsize=(10, 8))  # 设置图片大小800*600
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    plt.bar(x - 1.5 * width, all, width, yerr=yerr_all, error_kw=err_attri, label='ALL')
    plt.bar(x - 0.5 * width, gcnp, width, yerr=yerr_gcnp, error_kw=err_attri, label='GCN-Pattern')
    plt.bar(x + 0.5 * width, grup, width, yerr=yerr_grup, error_kw=err_attri, label='GRU-Pattern')
    plt.bar(x + 1.5 * width, gg, width, yerr=yerr_gg, error_kw=err_attri, label='GCN-GRU')

    plt.ylabel('Accuracy')
    # plt.title('4 datasets')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=path, fontsize=20)
    plt.ylim(0.7, 1)
    plt.yticks(fontsize=20)
    plt.xlabel('Train/Test', fontsize=26)  # 设置x轴标签
    plt.ylabel('Accuracy', fontsize=28)  # 设置y轴标签
    # plt.title('')  # 设置标题
    plt.grid(True)  # 是否现在网格线
    plt.rcParams.update({'font.size': 20, 'font.sans-serif': 'SimHei'})  # 设置图例文中大小
    plt.legend(loc="upper center")

    plt.show()

# 画all和double的对比
def print_singel():
    path = ["D37", "D46", "D55"]
    file = ["/all", "/double_gcnp", "/double_grup", "/double_gg","/gcn","/gru"]
    y = np.zeros(18)
    yerr = np.zeros((2, 18))
    yerr_all = np.zeros((2, 3))
    yerr_gcnp = np.zeros((2, 3))
    yerr_grup = np.zeros((2, 3))
    yerr_gg = np.zeros((2, 3))
    yerr_gcn = np.zeros((2, 3))
    yerr_gru = np.zeros((2, 3))
    all = np.zeros(3)
    gcnp = np.zeros(3)
    grup = np.zeros(3)
    gg = np.zeros(3)
    gcn = np.zeros(3)
    gru = np.zeros(3)
    i = 0
    for f in range(len(file)):
        for n in range(len(path)):
            info = read_bytecode(path[n] + file[f])
            temp = get_ave_maxmin(info)
            # all.append(temp[[0, 3, 4]])
            y[i] = temp[0]
            yerr[0][i] = temp[2]
            yerr[1][i] = temp[1]
            i += 1
    yerr_all[:, 0:3] = yerr[:, 0:3]
    yerr_gcnp[:, 0:3] = yerr[:, 3:6]
    yerr_grup[:, 0:3] = yerr[:, 6:9]
    yerr_gg[:, 0:3] = yerr[:, 9:12]
    yerr_gcn[:,0:3] = yerr[:,12:15]
    yerr_gru[:,0:3] = yerr[:,15:18]
    all[:] = y[0:3]
    gcnp[:] = y[3:6]
    grup[:] = y[6:9]
    gg[:] = y[9:12]
    gcn[:] = y[12:15]
    gru[:] = y[15:18]

    x = np.arange(len(path))  # x轴刻度标签位置
    width = 0.13  # 柱子的宽度
    err_attri = {"elinewidth": 1.2, "ecolor": 'black', "capsize": 10}
    plt.figure(figsize=(10, 8))  # 设置图片大小800*600
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    plt.bar(x - 2.5 * width, all, width, yerr=yerr_all, error_kw=err_attri, label='ALL')
    plt.bar(x - 1.5 * width, gcnp, width, yerr=yerr_gcnp, error_kw=err_attri, label='GCN-Pattern')
    plt.bar(x - 0.5 * width, grup, width, yerr=yerr_grup, error_kw=err_attri, label='GRU-Pattern')
    plt.bar(x + 0.5 * width, gg, width, yerr=yerr_gg, error_kw=err_attri, label='GCN-GRU')
    plt.bar(x + 1.5 * width, gcn, width, yerr=yerr_gcn, error_kw=err_attri, label='GCN')
    plt.bar(x + 2.5 * width, gru, width, yerr=yerr_gru, error_kw=err_attri, label='GRU')

    plt.ylabel('Accuracy')
    # plt.title('4 datasets')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=path, fontsize=20)
    plt.ylim(0.7, 1)
    plt.yticks(fontsize=20)
    plt.xlabel('Train/Test', fontsize=26)  # 设置x轴标签
    plt.ylabel('Accuracy', fontsize=28)  # 设置y轴标签
    # plt.title('')  # 设置标题
    plt.grid(True)  # 是否现在网格线
    plt.rcParams.update({'font.size': 20, 'font.sans-serif': 'SimHei'})  # 设置图例文中大小
    # plt.legend(loc="upper center")
    plt.legend()

    plt.show()

if __name__ == '__main__':
    # print_percentage()
    print_all_double()
    # print_singel()