import numpy as np
from gensim.models import KeyedVectors,word2vec,Word2Vec
from sklearn.decomposition import PCA

'''
存储合约控制流图、邻接矩阵特征
Input:
key: contract_name 
value : the value of compiled contract
cfg: Control Flow Graph of contract 
'''
class CFGFeature:
    def __init__(self,key,cfg,pattern,label=0):
        self.name = key
        # self.value = value
        self.cfg = cfg
        self.label = int(label)
        self.pattern = np.array(pattern)
        self.basicBlock = []
        self.basicBlock_pc = []
        self.allInstructions = cfg.instructions # 所有的opcode序列
        self.allinstructions_feature = []
        self.basicBlock_len = len(cfg.basic_blocks)
        self.adjacency = np.eye(self.basicBlock_len) # 带自连接的邻接矩阵
        # self.adjacency = np.full((self.basicBlock_len,self.basicBlock_len),0.01) + np.eye(self.basicBlock_len) # 带自连接的邻接矩阵
        self.degree_matrix = np.eye(self.basicBlock_len) # 度矩阵
        self.block_feature = []
        self.edge_src = []
        self.edge_dst = []
        self.opcode_vec()
        self.initBasicBlock()
        self.init_Degree_adjacency()
        self.create_block_feature()

    # 排序初始化block
    def initBasicBlock(self):
        for basic_block in sorted(self.cfg.basic_blocks, key=lambda x: x.start.pc):
            # print(f"{basic_block} -> {sorted(basic_block.all_outgoing_basic_blocks, key=lambda x: x.start.pc)}")
            # print(f"{basic_block} <- {sorted(basic_block.all_incoming_basic_blocks, key=lambda x: x.start.pc)}")
            self.basicBlock.append(basic_block)
            self.basicBlock_pc.append(np.arange(basic_block.start.pc, basic_block.end.pc+1))

    # 初始化度矩阵和邻接矩阵
    def init_Degree_adjacency(self):
        for i in range(self.basicBlock_len):
            degree = len(self.basicBlock[i].all_outgoing_basic_blocks) + len(self.basicBlock[i].all_incoming_basic_blocks)
            self.degree_matrix[i][i] += degree
            # 处理邻接矩阵
            for j in range(len(self.basicBlock[i].all_outgoing_basic_blocks)):
                block_idx = self.get_BlockIndex_pc(self.basicBlock[i].all_outgoing_basic_blocks[j].start.pc)
                self.adjacency[i][block_idx] += 1
                self.edge_src.append(i)
                self.edge_dst.append(block_idx)

            for j in range(len(self.basicBlock[i].all_incoming_basic_blocks)):
                block_idx = self.get_BlockIndex_pc(self.basicBlock[i].all_incoming_basic_blocks[j].start.pc)
                self.adjacency[i][block_idx] += 1
                self.edge_src.append(block_idx)
                self.edge_dst.append(i)

    # 生成所有block的向量特征
    def create_block_feature(self):
        pc_idx = 0
        pca1 = PCA(n_components=1)
        # 遍历block
        for i in range(self.basicBlock_len):
            temp_feature = []

            # 获取每个block中的opcode向量序列
            block_op_len = len(self.basicBlock[i].instructions)
            temp_op_feature = self.allinstructions_feature[pc_idx: pc_idx+block_op_len]
            pc_idx += block_op_len
            temp_feature.append(temp_op_feature)
            # （x,300）-> （1，300）
            temp_feature = np.array(temp_feature).reshape((-1,300))
            # 计算每行的范数
            # row_norms = np.linalg.norm(temp_feature,ord=2, axis=1)
            # row_norms = row_norms.reshape(-1, 1) # 范数转化（x) -> (x,1)
            # # 对每个向量的列进行平均
            # normalized = temp_feature / row_norms

            # x_min = np.min(temp_feature)
            # x_max = np.max(temp_feature)
            # temp_feature = (temp_feature-x_min)/(x_max-x_min)
            # col_means = np.mean(temp_feature, axis=0)
            # self.block_feature.append(col_means.reshape(1, -1))

            pca_feature = temp_feature.T
            pca_feature = pca1.fit_transform(pca_feature)
            pca_feature = pca_feature.T
            self.block_feature.append(pca_feature.reshape(1, -1))
        self.block_feature = np.array(self.block_feature).reshape((-1,300))

    # 根据pc值获取对应的block的index
    def get_BlockIndex_pc(self, pc):
        for idx in range(self.basicBlock_len):
            if pc in self.basicBlock_pc[idx]:
                return idx
        assert ("pc does not in block!")

    # 将所有操作码进行向量化
    def opcode_vec(self):
        # model = Word2Vec.load('word2vec.model')
        model = Word2Vec.load('word2vec_2.model')
        for i in range(len(self.allInstructions)):
            temp_op_feature = []
            temp = np.zeros((1,600))
            opcode_seq = str(self.allInstructions[i]).split(" ")
            # 遍历opcode序列的每个操作码
            for k in range(len(opcode_seq)):
                if temp_op_feature == []:
                    temp_op_feature = model.wv[opcode_seq[k]]
                else:
                    vec = model.wv[opcode_seq[k]]
                    temp_op_feature = np.concatenate([temp_op_feature, vec], axis=0)
            # 使用线性插值 (x*300)-> (300)
            temp_op_feature = np.interp(np.linspace(0, len(temp_op_feature) - 1, 300), np.arange(len(temp_op_feature)),
                                        temp_op_feature)
            # 使用线性插值的返回
            self.allinstructions_feature.append(temp_op_feature.reshape(1,-1))