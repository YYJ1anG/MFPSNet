from tqdm import tqdm
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from time import time
from retrain.MFPSNet import MFPSNet
from utils.loss import PerceptualLoss
from dataloader.FFHQ_train_MFPS import FFHQ_MFPS_train
from config_utils.train_args import obtain_train_args


from tensorboardX import SummaryWriter

opt = obtain_train_args()
print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
kwargs = {'num_workers': opt.threads, 'pin_memory': True, 'drop_last':True}
training_data = FFHQ_MFPS_train(opt)
training_data_loader = DataLoader(training_data, opt.batch_size, shuffle=True, num_workers=opt.threads, pin_memory=True)


print('===> Building model')
model = MFPSNet(opt)
# compute parameters
print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in model.parameters()])))


cri_perceptual = PerceptualLoss(perceptual_weight=0.005).cuda()



if cuda:
    model = model.cuda()

torch.backends.cudnn.benchmark = True

if opt.solver == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.001, betas=(0.9,0.99))

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)
start_epoch = 1
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def train(epoch):

    tbar = tqdm(training_data_loader)
    model.train()
    valid_iteration = len(training_data_loader) * (epoch-1)
    for iteration, batch in enumerate(tbar):
        hq_img, lq_img, parsemap, heatmap, facedict,  name = batch

        if cuda:
            hq_img = hq_img.cuda()
            lq_img = lq_img.cuda()
            parsemap = parsemap.cuda()
            heatmap = heatmap.cuda()
            facedict = facedict.cuda()
            
        train_start_time = time()

        optimizer.zero_grad()

        output_imgs = model(lq_img, parsemap, heatmap, facedict, opt.sum_iterations)

        loss1 = 0
        l1_loss1 = F.l1_loss(output_imgs[0], hq_img, reduction='mean')
        loss1 += l1_loss1
        l_g_percep, l_g_style = cri_perceptual(output_imgs[0], hq_img)
        lp1 = 0
        if l_g_percep is not None:
            lp1 += l_g_percep
        if l_g_style is not None:
            lp1 += l_g_style
        loss1 += lp1
        
        loss2 = 0
        l1_loss2 = F.l1_loss(output_imgs[1], hq_img, reduction='mean')
        loss2 += l1_loss2
        l_g_percep, l_g_style = cri_perceptual(output_imgs[1], hq_img)
        lp2 = 0
        if l_g_percep is not None:
            lp2 += l_g_percep
        if l_g_style is not None:
            lp2 += l_g_style
        loss2 += lp2 

        loss = loss1 + loss2  
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss, global_step=valid_iteration)
        valid_iteration += 1
        if valid_iteration % opt.show_result_feq == 0:
            for i in range(opt.sum_iterations):
                lq_img1 = lq_img[0].cpu().detach().numpy() * 255
                output_img1 = output_imgs[i][0].cpu().detach().numpy() * 255
                hq_img1 = hq_img[0].cpu().detach().numpy() * 255

                output = np.concatenate((lq_img1, output_img1, hq_img1), axis=2).transpose(1, 2, 0)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                if not os.path.exists('./checkpoint/{}/retrain/tmp'.format(opt.model_name)):
                    os.makedirs('./checkpoint/{}/retrain/tmp'.format(opt.model_name))
                cv2.imwrite('./checkpoint/{}/retrain/tmp/SRmodule_{}_{}_iter{}_{}'.format(opt.model_name, epoch, valid_iteration,i, name[0]), output)
        train_end_time = time()
        train_time = train_end_time - train_start_time

        tbar.set_description("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), loss.item(), train_time))                      


if __name__ == '__main__':

    if not os.path.exists('./checkpoint/{}/retrain'.format(opt.model_name)):
        os.makedirs('./checkpoint/{}/retrain'.format(opt.model_name))
        os.makedirs('./checkpoint/{}/retrain/tmp'.format(opt.model_name))
    writer = SummaryWriter(log_dir='./checkpoint/{}/retrain'.format(opt.model_name))

    for i in range(1, start_epoch):
        scheduler.step()
    for epoch in range(start_epoch, opt.nEpochs + 1):

        train(epoch)

        if epoch >= opt.start_save_epoch:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename='./checkpoint/{}/retrain/checkpoint_{}.pth.tar'.format(opt.model_name, epoch)
            torch.save(state, filename)

        scheduler.step()




