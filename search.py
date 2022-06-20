import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.FFHQ_search import FFHQ_search_MFPS
import torch.nn.functional as F
from config_utils.search_args import obtain_search_args
from models.build_model import MFPSNet
import torch.optim as optim
opt = obtain_search_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


class Trainer(object):
    def __init__(self, args):
        
        self.args = args

        trainA = FFHQ_search_MFPS(opt, 'arch')
        trainB = FFHQ_search_MFPS(opt, 'cell')
        self.train_loaderA = DataLoader(trainA, opt.batch_size, shuffle=True, num_workers=opt.threads, pin_memory=True)
        self.train_loaderB = DataLoader(trainB, opt.batch_size, shuffle=True, num_workers=opt.threads, pin_memory=True)

        # Define network
        self.model = MFPSNet()

        self.optimizer_weight = torch.optim.SGD(
                self.model.weight_parameters(), 
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )        


        self.optimizer_arch = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        self.scheduler_weight = optim.lr_scheduler.MultiStepLR(self.optimizer_weight, milestones=opt.milestones, gamma=0.5)
        self.scheduler_arch = optim.lr_scheduler.MultiStepLR(self.optimizer_arch, milestones=opt.milestones, gamma=0.5)
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()

        print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

    def training(self, epoch):
        train_loss = 0.0
        valid_iteration = 0
        self.model.train()
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)

        for i, batch in enumerate(tbar):
            input, gt, parsing, heatmaps, dict, name = batch
            if self.args.cuda:
                input = input.cuda()
                gt = gt.cuda()
                parsing = parsing.cuda()
                heatmaps = heatmaps.cuda()
                dict = dict.cuda()

            
                self.optimizer_weight.zero_grad()
                output = self.model(input, parsing, heatmaps, dict) 
                loss = F.smooth_l1_loss(output, gt, reduction='mean')
                loss.backward()            
                self.optimizer_weight.step()     


                if epoch >= self.args.alpha_epoch:
                    print("Start searching architecture!...........")
                    search = next(iter(self.train_loaderB))
                    input_search, gt_search, name_search = search
                    if self.args.cuda:
                        input_search = input_search.cuda()
                        gt_search = gt_search.cuda()

                    self.optimizer_arch.zero_grad()
                    output_search = self.model(input, gt)
                    arch_loss = F.smooth_l1_loss(output_search, gt_search, reduction='mean')
                    arch_loss.backward()
                    self.optimizer_arch.step()
                    tbar.set_description('Train architecture loss: %.3f' % (arch_loss.item()))   

                train_loss += loss.item()
                valid_iteration += 1
                tbar.set_description('Train loss: %.3f' % (loss.item()))
                
        self.scheduler_weight.step()
        if epoch >= self.args.alpha_epoch:
            self.scheduler_arch.step()
        
     
        print("=== Train ===> Epoch :{} Error: {:.4f}".format(epoch, train_loss/valid_iteration))

        #save checkpoint every epoch
        is_best = False
        if torch.cuda.device_count() > 1:
           state_dict = self.model.module.state_dict()
        else:
           state_dict = self.model.state_dict()
        self.saver.save_checkpoint({
               'epoch': epoch + 1,
               'state_dict': state_dict,
               'optimizer_net': self.optimizer_net.state_dict(),
               'optimizer_arch': self.optimizer_arch.state_dict(),
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch))

if __name__ == "__main__":
    trainer = Trainer(opt)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)