
import argparse
def obtain_test_args():
    parser = argparse.ArgumentParser(description='MFPS training...')
    parser.add_argument('--model_name', type=str, default='MFPS')
    parser.add_argument('--lq_path', type=str, default='./data/img')
    parser.add_argument('--gt_path', type=str, default='./data/gt')
    parser.add_argument('--test_parse_path', type=str, default='./data/priors/parse')
    parser.add_argument('--test_heat_path', type=str, default='./data/priors/heat')
    parser.add_argument('--test_dict_path', type=str, default='./data/priors/dict')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_test.pth')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--cuda', type=int, default=0, help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
    parser.add_argument('--save_path', type=str, default='./results/', help="location to save images")
    
    args = parser.parse_args()
    return args