import argparse

def obtain_search_args():
    parser = argparse.ArgumentParser(description="MFPSNet Searching")

    parser.add_argument('--lq_full_path', type=str, default='')
    parser.add_argument('--hq_path', type=str, default='')
    parser.add_argument('--parse_path', type=str, default='')
    parser.add_argument('--heat_path', type=str, default='')
    parser.add_argument('--dict_path', type=str, default='')
    parser.add_argument('--threads', type=int, default=0, 
                        help='number of threads for data loader to use')
    parser.add_argument('--milestones', default=[10*i for i in range(1,10)], metavar='N', nargs='*', 
                        help='epochs at which learning rate is divided by 2') 
   
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')   
    parser.add_argument('--max_disp', type=int, default=192, help="max disp")
    parser.add_argument('--crop_height', type=int, default=384, 
                        help="crop height")
    parser.add_argument('--crop_width', type=int, default=576,
                        help="crop width")
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--alpha_epoch', type=int, default=10,
                        metavar='N', help='epoch to start training alphas')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--testBatchSize', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate for alpha and beta in architect searching process')

  
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--cuda', type=int, default=1, 
                        help='use cuda? Default=True')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    args = parser.parse_args()
    return args
