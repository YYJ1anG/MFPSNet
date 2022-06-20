
import os, torch
import os
import numpy as np
import cv2

class FFHQ_MFPS_train(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_names = os.listdir(opt.lq_full_path)
        self.lq_path = opt.lq_full_path
        self.gt_path = opt.hq_path
        self.parse_path = opt.parse_path
        self.heat_path = opt.heat_path
        self.dict_path = opt.dict_path
        self.baseline_path = opt.baseline_path


    def read_face_pair(self, img_name):
        #name = os.path.splitext(img_name)[-2]
        name = img_name[:-4]
        hq_path = os.path.join(self.gt_path, img_name)
        lq_path = os.path.join(self.lq_path, img_name)
        parse_path = self.parse_path+'/'+name+'.npy'
        heat_path = self.heat_path+'/'+name+'.npy'
        dict_path = self.dict_path+'/'+name+'.npy'
       

        
        hq = cv2.cvtColor(cv2.imread(hq_path), cv2.COLOR_BGR2RGB)
        lq = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)
       
        hq = hq.transpose(2, 0, 1) / 255.
        hq = torch.from_numpy(hq).to(torch.float32)

        lq = lq.transpose(2, 0, 1) / 255.
        lq = torch.from_numpy(lq).to(torch.float32)


        parsemap = np.load(parse_path)
        parsemap = torch.from_numpy(parsemap).to(torch.float32)
        #print(parsemap.shape)
        heatmap = np.load(heat_path)
        heatmap = torch.from_numpy(heatmap).to(torch.float32)

        facedict =np.load(dict_path)
        facedict = torch.from_numpy(facedict).to(torch.float32)

        return hq, lq, parsemap, heatmap, facedict,  img_name


    def __getitem__(self, index):
        img_name = self.img_names[index]
        return self.read_face_pair(img_name)


    def __len__(self):
       return len(self.img_names)
