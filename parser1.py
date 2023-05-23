import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection based on graph neural network')
    parser.add_argument('-D', '--dataset', type=str, default='REENTRANCY_CORENODES_1671',
                        choices=['REENTRANCY_CORENODES_1671', 'REENTRANCY_FULLNODES_1671',
                                 'LOOP_CORENODES_1317', 'LOOP_FULLNODES_1317'])
    parser.add_argument('-M', '--model', type=str, default='gcn',
                        choices=['gcn', 'dgl', 'gru', 'pattern','all','all_double'])
    parser.add_argument('--mode', dest='mode', type=str, default='test')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--node_input_dim',dest='node_input_dim',type=int,default=300,help='节点特征维度')
    parser.add_argument('--times',dest='epochs_times',type=int,default=1,help='训练一个模型的次数')
    parser.add_argument('--double',dest='double',type=str,help='double时选择哪两个相融合')
    parser.add_argument('--D',dest='D',type=str,help='选择训练集和测试集的比例')


    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--lr_decay_steps', type=str, default='10,30', help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
    parser.add_argument('--n_hidden', type=int, default=100,
                        help='number of hidden units in a fully connected layer after the last conv layer')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--shuffle_nodes', action='store_true', default=True, help='shuffle nodes for debugging')
    parser.add_argument('-F', '--folds', default=5, choices=[3, 5, 10], help='n-fold cross validation')
    parser.add_argument('--multi_head', type=int, default=4, help='number of head attentions(Multi-Head)')

    return parser.parse_args()