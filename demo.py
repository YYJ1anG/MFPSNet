from tqdm import tqdm
import os
import torch
import torch.nn.parallel
import numpy as np
import cv2
from torch.utils.data import DataLoader
from retrain.MFPSNet import  MFPSNet
from dataloader.test_dataset import TestDataset
from config_utils.test_args import obtain_test_args

opt = obtain_test_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
kwargs = {'num_workers': opt.threads, 'pin_memory': True, 'drop_last':True}

testing_data = TestDataset(opt)
testing_data_loader = DataLoader(testing_data, 1, shuffle=False, num_workers=0, pin_memory=True)

print('===> Building model')
model = MFPSNet(opt)

if cuda:
    model = model.cuda()

torch.backends.cudnn.benchmark = True

print("=> loading checkpoint '{}'".format(opt.checkpoint))
checkpoint = torch.load(opt.checkpoint)
model.load_state_dict(checkpoint, strict=True)


result_name = 'MFPS_demo'
with torch.no_grad():
    tbar = tqdm(testing_data_loader)
    model.eval()
    for i, batch in enumerate(tbar):
        hq_img, lq_img, parsemap, heatmap, facedict, name = batch
        if cuda:
            hq_img = hq_img.cuda()
            lq_img = lq_img.cuda()
            parsemap = parsemap.cuda()
            heatmap = heatmap.cuda()
            facedict = facedict.cuda()
        with torch.no_grad():
            output_imgs = model(lq_img, parsemap, heatmap, facedict, 2)

        for i in range(2):
            lq_img1 = lq_img[0].cpu().detach().numpy() * 255
            output_img0 = output_imgs[i][0].cpu().detach().numpy() * 255
            hq_img1 = hq_img[0].cpu().detach().numpy() * 255
            output = np.concatenate((lq_img1, output_img0,  hq_img1), axis=2).transpose(1, 2, 0)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            if not os.path.exists('./results/{}/iter{}'.format(result_name, i)):
                os.makedirs('./results/{}/iter{}'.format(result_name, i))
            cv2.imwrite('./results/{}/iter{}/{}'.format(result_name, i, name[0]), output)
    print("===> test done")