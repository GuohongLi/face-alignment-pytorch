from __future__ import print_function

import os
import numpy as np
import random
import math
import cv2
from skimage import io

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua

# from utils.utils import *
from utils.imutils import *
from utils.transforms import *


class W300LP(data.Dataset):

    def __init__(self, args, split):
        self.nParts = 68
        self.pointType = args.pointType
        # self.anno = anno
        self.img_folder = args.data
        self.split = split
        self.is_train = True if self.split == 'train' else False
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = args.scale_factor
        self.rot_factor = args.rot_factor
        self.mean, self.std = self._comput_mean()

    def _getDataFaces(self, is_train):
        base_dir = os.path.join(self.img_folder, 'landmarks')
        dirs = os.listdir(base_dir)
        lines = []
        vallines = []
        for d in dirs:
	    #files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.mat')]
            files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.t7')]
            for f in files:
                if f.find('test') == -1:
                    lines.append(f)
                else:
                    vallines.append(f)
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            print('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, c, s, reference_scale = self.generateSampleFace(index)
        self.pts, self.c, self.s, self.reference_scale = pts, c, s, reference_scale
        if self.is_train:
            return inp, out
        else:
            meta = {'index': index, 'center': c, 'scale': s, 'pts': pts, 'reference_scale': reference_scale}
            return inp, out, meta

    def generateSampleFace(self, idx):
        #points
        main_pts = load_lua(
            os.path.join(self.img_folder, 'landmarks', self.anno[idx].split('_')[0],
                         self.anno[idx][:-3] + '.t7'))
        pts = main_pts[0] if self.pointType == '2D' else main_pts[1]

        #image
        sf = self.scale_factor
        rf = self.rot_factor

        imagepath = os.path.join(self.img_folder, self.anno[idx].split('_')[0], self.anno[idx][:-7] + '.jpg')
        img = load_image(imagepath)

        height = img.size(1)
        width = img.size(2)
        hw = max(width, height)
        #consider the landmarks are mainly distributed in the lower half face, so it needs move the center to some lower along height direction
        c = torch.FloatTensor(( float(width*1.0/2), float(height*1.0/2 + height*0.12) ))
        reference_scale = torch.tensor(200.0)
        #we hope face for train to be larger than in raw image, so edge will be short by 0.8 ration
        scale_x = hw*0.8 / reference_scale
        scale_y = hw*0.8 / reference_scale
        s = torch.FloatTensor(( scale_x, scale_y ))

        r = 0
        if self.is_train:
            s[0] = s[0] * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            s[1] = s[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        inp = crop(img, c, s, reference_scale, [256, 256], rot=r)
        # inp = color_normalize(inp, self.mean, self.std)

        tpts = pts.clone()
        out = torch.zeros(self.nParts, 64, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, reference_scale, [64, 64], rot=r))
                out[i] = draw_labelmap(out[i], tpts[i], sigma=1)

        '''
        out_tmp = out.view(1, self.nParts, 64, 64)
        c_tmp = c.view(1, -1)
        s_tmp = s.view(1, -1)
        reference_scale_tmp = reference_scale.view(1, -1)
        pts, pts_img = get_preds_fromhm(out_tmp, c_tmp, s_tmp, reference_scale_tmp)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
        io.imsave('%s_crop_pts.jpg' % idx,im_to_numpy(inp))
        inp_in = io.imread('%s_crop_pts.jpg' % idx)
        for i in range(pts.shape[0]):
            cv2.circle(inp_in, (pts[i][0], pts[i][1]),2,(0,255,0), -1, 8)
        io.imsave('%s_crop_pts.jpg' % idx,inp_in)
        '''

        return inp, out, pts, c, s, reference_scale

    def _comput_mean(self):
        meanstd_file = './data/300W_LP/mean.pth.tar'
        if os.path.isfile(meanstd_file):
            ms = torch.load(meanstd_file)
        else:
            print("\tcomputing mean and std for the first time, it may takes a while, drink a cup of coffe...")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            if self.is_train:
                for i in range(self.total):
                    a = self.anno[i]
                    img_path = os.path.join(self.img_folder, self.anno[i].split('_')[0],
                                            self.anno[i][:-8] + '.jpg')
                    img = load_image(img_path)
                    mean += img.view(img.size(0), -1).mean(1)
                    std += img.view(img.size(0), -1).std(1)

            mean /= self.total
            std /= self.total
            ms = {
                'mean': mean,
                'std': std,
            }
            torch.save(ms, meanstd_file)
        if self.is_train:
            print('\tMean: %.4f, %.4f, %.4f' % (ms['mean'][0], ms['mean'][1], ms['mean'][2]))
            print('\tStd:  %.4f, %.4f, %.4f' % (ms['std'][0], ms['std'][1], ms['std'][2]))
        return ms['mean'], ms['std']

if __name__=="__main__":
    import opts, demo
    args = opts.argparser()
    dataset = W300LP(args, 'test')
    crop_win = None
    for i in range(dataset.__len__()):
        input, target, meta = dataset.__getitem__(i)
        input = input.numpy().transpose(1,2,0) * 255.
        target = target.numpy()
        if crop_win is None:
            crop_win = plt.imshow(input)
        else:
            crop_win.set_data(input)
        plt.pause(0.5)
        plt.draw
