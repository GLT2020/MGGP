import os
import numpy as np
import csv
import shutil
def mkdir_multi(path,list):
    # 判断路径是否存在
    isExists=os.path.exists(path)

    if not isExists:
        path_build = path + '/build/contracts'
        path_contracts = path + '/contracts'
        path_migrations = path + '/migrations'
        path_data = "../data/D1/"+list[0] +".sol"
        cmd = "hello.py -name "+list[1]
        # 如果不存在，则创建目录（多层）
        os.makedirs(path_build)
        os.makedirs(path_contracts)
        os.makedirs(path_migrations)

        shutil.copy('../example/crowdsale/contracts/Migrations.sol', path_contracts)
        shutil.copy('../example/crowdsale/migrations/1_initial_migration.js', path_migrations)
        shutil.copy(path_data, path_contracts)
        read = os.popen(cmd,mode="r")
        print(read)
        print('目录创建成功！')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('目录已存在！')
        return False

if __name__ == '__main__':
    # 读取数据文件
    file_name = "E:\Demo_PyCharm\MGCE\data\D1_labels.csv"
    f = open(file_name, 'r')
    csvreader = csv.reader(f)
    final_list = list(csvreader)
    del final_list[0]
    # print(final_list)

    for i in range(len(final_list)):
        path_name = "../example/" + final_list[i][1]
        flag = mkdir_multi(path_name,final_list[i])
        if (i == 2):
            break


