import cv2
import os, torch
import os
import numpy as np


class FFHQ_search_MFPS(torch.utils.data.Dataset):
    def __init__(self, opt, type):
        self.opt = opt
        self.img_names = os.listdir(opt.lq_full_path)
        if type == 'arch':
            self.img_names =  self.img_names[:len(self.img_names)//2]
        elif type == 'cell':
            self.img_names =  self.img_names[len(self.img_names)//2:]
        self.lq_path = opt.lq_full_path
        self.gt_path = opt.hq_path
        self.parse_path = opt.parse_path
        self.heat_path = opt.heat_path
        self.dict_path = opt.dict_path

    def read_face_pair(self, img_name):
        hq_path = os.path.join(self.gt_path, img_name)
        lq_path = os.path.join(self.lq_path, img_name)
       
        parse_path = self.parse_path+'/'+img_name+'.npy'
        heat_path = self.heat_path+'/'+img_name+'.npy'
        dict_path = self.dict_path+'/'+img_name+'.npy'

        hq = cv2.cvtColor(cv2.imread(hq_path), cv2.COLOR_BGR2RGB)
        lq = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)

        hq = hq.transpose(2, 0, 1) / 255.
        hq = torch.from_numpy(hq).to(torch.float32)

        lq = lq.transpose(2, 0, 1) / 255.
        lq = torch.from_numpy(lq).to(torch.float32)

        parsemap = np.load(parse_path)
        parsemap = torch.from_numpy(parsemap).to(torch.float32)

        heatmap = np.load(heat_path)
        heatmap = torch.from_numpy(heatmap).to(torch.float32)

        facedict =np.load(dict_path)
        facedict = torch.from_numpy(facedict).to(torch.float32)

        return hq, lq, parsemap, heatmap, facedict, img_name

    def __getitem__(self, index):
        img_name = self.img_names[index]
        return self.read_face_pair(img_name)

    def __len__(self):
        return len(self.img_names)



class FFHQ_search(torch.utils.data.Dataset):
    def __init__(self, opt, type):
        self.opt = opt
        self.img_names = os.listdir(opt.lq_full_path)
        if type == 'arch':
            self.img_names =  self.img_names[:len(self.img_names)//2]
        elif type == 'cell':
            self.img_names =  self.img_names[len(self.img_names)//2:]
        self.lq_path = opt.lq_full_path
        self.gt_path = opt.hq_path

    def read_face_pair(self, img_name):
        name = os.path.splitext(img_name)[-2]
        hq_path = os.path.join(self.gt_path, img_name)
        lq_path = os.path.join(self.lq_path, img_name)
       

        hq = cv2.cvtColor(cv2.imread(hq_path), cv2.COLOR_BGR2RGB)
        lq = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)

        hq = hq.transpose(2, 0, 1) / 255.
        hq = torch.from_numpy(hq).to(torch.float32)

        lq = lq.transpose(2, 0, 1) / 255.
        lq = torch.from_numpy(lq).to(torch.float32)

        return hq, lq, img_name


    def __getitem__(self, index):
        img_name = self.img_names[index]
        return self.read_face_pair(img_name)


    def __len__(self):
        return len(self.img_names)



