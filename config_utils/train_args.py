
import argparse

def obtain_train_args():
    # Training settings
    parser = argparse.ArgumentParser(description='MFPS training...')
    parser.add_argument('--model_name', type=str, default='MFPS')
    parser.add_argument('--lq_full_path', type=str, default='')
    parser.add_argument('--hq_path', type=str, default='')
    parser.add_argument('--parse_path', type=str, default='')
    parser.add_argument('--heat_path', type=str, default='')
    parser.add_argument('--dict_path', type=str, default='')


    parser.add_argument('--show_result_feq', type=int, default=50)
    parser.add_argument('--start_save_epoch', type=str, default=1)
    parser.add_argument('--resume', type=str, default='', 
                        help="resume from saved model")
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, 
                        help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=104, 
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning Rate. Default=0.0001')
    parser.add_argument('--cuda', type=int, default=2, 
                        help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=12, 
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, 
                        help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, 
                        help='random shift of left image. Default=0')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', 
                        help="location to save models")
    parser.add_argument('--milestones', default=[3,7,10,20,30,40,50,60,70,80,90,100], metavar='N', nargs='*', 
                        help='epochs at which learning rate is divided by 2')    
    parser.add_argument('--stage', type=str, default='train', choices=['search', 'train'])


    ######### MFPS params ##################
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--block_multiplier', type=int, default=3)
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--sum_iterations', type=int, default=2)

    parser.add_argument('--sr_cell_arch', default='./checkpoint/SRmodule_l6_baseline/search/net_arch/checkpoint_99_arch/cell_structure.npy', type=str)
    parser.add_argument('--sr_net_arch', default='./checkpoint/SRmodule_l6_baseline/search/net_arch/checkpoint_99_arch/net_path.npy', type=str)
    parser.add_argument('--parse_cell_arch', default='./checkpoint/SRmodule_l6_withparse/search/net_arch/checkpoint_99_arch/cell_structure.npy', type=str)
    parser.add_argument('--parse_net_arch', default='./checkpoint/SRmodule_l6_withparse/search/net_arch/checkpoint_99_arch/net_path.npy', type=str)
    parser.add_argument('--heat_cell_arch', default='./checkpoint/SRmodule_l6_withheatmap/search/net_arch/checkpoint_90_arch/cell_structure.npy', type=str)
    parser.add_argument('--heat_net_arch', default='./checkpoint/SRmodule_l6_withheatmap/search/net_arch/checkpoint_90_arch/net_path.npy', type=str)
    parser.add_argument('--dict_cell_arch', default='./checkpoint/SRmodule_l6_withfacedict/search/net_arch/checkpoint_60_arch/cell_structure.npy', type=str)
    parser.add_argument('--dict_net_arch', default='./checkpoint/SRmodule_l6_withfacedict/search/net_arch/checkpoint_60_arch/net_path.npy', type=str)
    parser.add_argument('--fusion_cell_arch', default='./checkpoint/SRmodule_l6_fusion/search/net_arch/checkpoint_45_arch/cell_structure.npy', type=str)
    parser.add_argument('--fusion_net_arch', default='./checkpoint/SRmodule_l6_fusion/search/net_arch/checkpoint_45_arch/net_path.npy', type=str)

    args = parser.parse_args()
    return args