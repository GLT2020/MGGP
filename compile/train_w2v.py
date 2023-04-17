from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import multiprocessing
from parser1 import parameter_parser
import json
from evm_cfg_builder.cfg import CFG
from CFGFeature import CFGFeature

args = parameter_parser()

def stopwordslist():
    stopwords = [ line.strip() for line in open('update_word.txt','r').readlines()]
    return stopwords

# 读取字节码列表
def read_bytecode():
    info_data = []
    with open("../compile/bytecode_dict_reentrancy.json", "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    return info_data

def create_w2v_word(cfg_feature_list:list):
    # create the w2v corpus 生成w2v的词汇表
    # with open('word.txt','a+') as writers:
    with open('update_word.txt','a+') as writers:
        for i in range(len(cfg_feature_list)):
            for j in range(len(cfg_feature_list[i])):
                writers.write(str(cfg_feature_list[i][j]) + " ")
    writers.write("\n")

# 模型训练
def create_w2v(sentences):
    model = Word2Vec(sentences, vector_size=args.node_input_dim, min_count=1, window=5, sg=0,
                     workers=multiprocessing.cpu_count())
    model.save('../word2vec.model')
    model.wv.save_word2vec_format('../word2vec.vector')


# 增量训练
def update_w2v():
    sentences = list(word2vec.LineSentence('update_word.txt'))
    model = Word2Vec.load('../word2vec.model')
    print(model)
    # new_sentence = [
    #     '我喜欢吃苹果',
    #     '大话西游手游很好玩',
    #     '人工智能包含机器视觉和自然语言处理'
    # ]
    stopwords = stopwordslist()
    sentences_cut = []
    # 结巴分词
    # for ele in new_sentence:
    #     cuts = jieba.cut(ele, cut_all=False)
    #     new_cuts = []
    #     for cut in cuts:
    #         if cut not in stopwords:
    #             new_cuts.append(cut)
    #     sentences_cut.append(new_cuts)

    # 增量训练word2vec
    model.build_vocab(sentences, update=True)  # 注意update = True 这个参数很重要
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    model.save('../word2vec_2.model')
    model.wv.save_word2vec_format('../word2vec_2.vector')
    print(model)

if __name__ == '__main__':

    # 根据合约创建语料库
    # cfg_feature_list = []
    # cfg_opcodes_list = []
    # bytecode_dict_list = read_bytecode()
    # for i in range(len(bytecode_dict_list)):
    #     bytecode = bytecode_dict_list[i]
    #     cfg = CFG(bytecode['runtime_bytecode'])
    #     cfg_opcodes_list.append(cfg.instructions)

    # create_w2v_word(cfg_opcodes_list)

    # 读取语料库
    # sentences = list(word2vec.LineSentence('word.txt'))
    # print(sentences)
    # 训练新的w2v
    # create_w2v(sentences)

    # 增量训练
    update_w2v()


    # 使用
    # model = Word2Vec.load('word2vec.model')
    # vec = model.wv['PUSH1 0X80']
    # print(vec)



