from solc import compile_files
import os
import pickle
import json
import csv
import numpy as np
from evm_cfg_builder.cfg import CFG

def store_bytecode(data):
    # info_json = json.dumps(data)
    # TODO： 需要更改对应的写入文件名
    # with open("../data/reentrancy/percentage/D19/train.json", "a+") as f:
    # with open("../data/reentrancy/percentage/D19/test.json", "a+") as f:
    #     # pickle.dump(data, my_file)
    #     f.write(info_json+"\n")

    # 将cfg一起保存
    with open("../data/reentrancy/percentage/D19/test.pkl", "wb") as f:
        pickle.dump(data, f)

    with open("../data/reentrancy/percentage/D19/test.pkl", "rb") as f:
        s = pickle.load(f)
    print(s)


def read_label():
    # TODO:修改label文件读取路径
    # path = "../data/reentrancy/reentrancy_D2_test_label" # 混合D1,D2
    # path = "../data/reentrancy/reentrancy_D2_train_label"
    # 不同比例的D2数据集
    # path = "../data/reentrancy/percentage/D19/train_label"
    path = "../data/reentrancy/percentage/D19/test_label"
    data = []
    data_dict = {}
    with open(path+".csv") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data.append(row)  # 选择某一行加入到data数组中
    for i in range(len(data)):
        data_dict[data[i][0]] = data[i][1]
    return  data,data_dict

# 读取pattern列表
def read_pattern():
    info_data = []
    data_dict = {}
    # TODO： 需要更改对应的pattern路径
    with open("../pattern_feature/reentrancy_D2.json", "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    for i in range(len(info_data)):
        data_dict[info_data[i]["name"]] = info_data[i]["pattern"]
    return info_data,data_dict

if __name__ == '__main__':
    # TODO： 需要更改读取的源码文件名
    # 混合D1,D2的数据
    path = "../data/reentrancy/D2_sourcecode"

    cfg_feature_list = []
    list_dict = []
    # 读取label
    label_list,label_dict = read_label()
    # 读取pattern
    pattern_list,pattern_dict = read_pattern()
    for root,dirs,files in os.walk(path):
        print("当前目录路径：",root)
        print("当前目录下所有子目录：",dirs)
        print("当前路径下所有非目录子文件",files)
        num = 0
        for i in range(len(files)):
            print(root+files[i])
            # 判断文件名是否在label表中，如果不存在则跳过
            if files[i] in label_dict.keys():
                file = compile_files([root+'/'+files[i]])
                max_bin_len = 0
                max_file_value = object
                max_file_key = ""
                for key,value in file.items():
                    print(file)
                    if(len(value['bin-runtime'])) > max_bin_len:
                        max_file_value = value
                        # max_file_key = key
                        max_bin_len = len(value['bin-runtime'])
                print(key)
                # max_file_key = max_file_key.split('/')[-1].split(':')[0]
                max_file_key = key.split('/')[-1].split(':')[0]
                runtime_bytecode = max_file_value['bin-runtime']
                cfg = CFG(runtime_bytecode)
                num += 1
                store_dict = {"name": max_file_key, "label": label_dict[max_file_key],
                              "runtime_bytecode": runtime_bytecode, "pattern":pattern_dict[max_file_key],
                              "cfg_basic_blocks":cfg.basic_blocks, "cfg_instructions":cfg.instructions}
                cfg_feature_list.append(store_dict)
        store_bytecode(store_dict)
        print(num)

            # create the w2v corpus 生成w2v的词汇表
            # with open('compile/word.txt', 'a+') as writers:
            #     for i in range(len(cfg_feature.allInstructions)):
            #         writers.write(str(cfg.instructions[i]) + " ")
            # writers.write("\n")

            # # 匹配对应的label
            # for j in range(len(label_list)):
            #     if(label_list[j][0] == max_file_key):
            #         # TODO 将runtime_bytecode和对应的file_key 进行存储，将编译合约切割出去
            #         # label = [0,0]
            #         # label[int(lable_list[j][1])] = 1
            #         store_dict = {"name": max_file_key, "label": label_list[j][1], "runtime_bytecode": runtime_bytecode}
            #         store_bytecode(store_dict)
            #         break



# 分析所有block
# idx = 0
# for basic_block in sorted(cfg.basic_blocks, key=lambda x: x.start.pc):
#     print(idx)
#     idx += 1
#     print(
#         # f"{basic_block} -> {sorted(basic_block.all_outgoing_basic_blocks, key=lambda x:x.start.pc)}"
#         f"{basic_block} -> {sorted(basic_block.all_outgoing_basic_blocks, key=lambda x: x.start.pc)}"
#     )

# 分析每个function
# for function in sorted(cfg.functions, key=lambda x: x.start_addr):
#     print(f"Function {function.name}")
#     # Each function may have a list of attributes
#     # An attribute can be:
#     # - payable
#     # - view
#     # - pure
#     if sorted(function.attributes):
#         print("\tAttributes:")
#         for attr in function.attributes:
#             print(f"\t\t-{attr}")
#
#     print("\n\tBasic Blocks:")
#     for basic_block in sorted(function.basic_blocks, key=lambda x: x.start.pc):
#         # Each basic block has a start and end instruction
#         # instructions are pyevmasm.Instruction objects
#         print(f"\t- @{hex(basic_block.start.pc)}-{hex(basic_block.end.pc)}")
#
#         print("\t\tInstructions:")
#         for ins in basic_block.instructions:
#             print(f"\t\t- {ins.name}")
#
#         # Each Basic block has a list of incoming and outgoing basic blocks
#         # A basic block can be shared by different functions
#         # And the list of incoming/outgoing basic blocks depends of the function
#         # incoming_basic_blocks(function_key) returns the list for the given function
#         print("\t\tIncoming basic_block:")
#         for incoming_bb in sorted(
#                 basic_block.incoming_basic_blocks(function.key), key=lambda x: x.start.pc
#         ):
#             print(f"\t\t- {incoming_bb}")
#
#         print("\t\tOutgoing basic_block:")
#         for outgoing_bb in sorted(
#                 basic_block.outgoing_basic_blocks(function.key), key=lambda x: x.start.pc
#         ):
#             print(f"\t\t- {outgoing_bb}")



