import pandas as pd
import xlrd

file_path = 'data/reentrancy/其他工具检测结果.xlsx'
data = xlrd.open_workbook(file_path)
# table = data.sheet_by_name('Sheet0')
table = data.sheets()[0]

# 获取总行数
nrows = table.nrows
# 获取总列数
ncols = table.ncols

# # 获取一行的全部数值，例如第5行
# row_value = table.row_values(5)
# 获取一列的全部数值，例如第6列
label = table.col_values(5)
tools = table.col_values(8)
del(label[0])
del(tools[0])
print(label)
print(tools)

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(label)):
    if label[i] == 1 and tools[i] == 1:
        tp+=1
        continue
    elif label[i] == 0 and tools[i] == 0:
        tn += 1
        continue
    elif label[i] == 1 and tools[i] == 0:
        fn += 1
        continue
    elif label[i] ==0 and tools[i] == 1:
        fp += 1
        continue

accuracy = (tp + tn) / ( tp + tn + fn + fp )
print("tp:",tp)
print("fp:",fp)
print("tn:",tn)
print("fn:",fn)
print("accuracy:",accuracy)
