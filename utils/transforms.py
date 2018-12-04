from __future__ import absolute_import

import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch

from .misc import *
from .imutils import *

def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)

def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset == 'mpii':
        matchedParts = ([0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13])
    elif dataset in ['w300lp', 'vw300', 'w300', 'menpo']:
        matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                        [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                        [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                        [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def get_transform(center, scale, reference_scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = float(reference_scale * scale[1])
    w = float(reference_scale * scale[0])
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def transform(point, center, scale, reference_scale, resolution, invert=False, rot=0):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {int,int} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    t = get_transform(center, scale, reference_scale, resolution, rot=rot)

    if invert:
        t = np.linalg.inv(t)
    #new_pt = np.array([_pt[0] - 1, _pt[1] - 1, 1.]).T
    new_pt = np.array([_pt[0], _pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)

    return new_pt[:2].astype(int)

def transform_preds(coords, center, scale, reference_scale, res):
    #print(center)
    #print(scale)
    #print(reference_scale)
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, reference_scale, res, True, 0))
    return coords

def crop(img, center, scale, reference_scale=200, res=[256,256], rot=0):
    #from C*H*W to H*W*C
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale[0] * reference_scale / res[0]
    new_ht,new_wd = ht, wd
    new_size = max(ht, wd)
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1. / sf
            scale = [scale[0] / sf, scale[1] / sf]

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, reference_scale, res, invert=True))
    # Bottom right point
    br = np.array(transform(res, center, scale, reference_scale, res, invert=True))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad
    #print("shape:{}, ul[0]:{}, ul[1]:{}, br[0]:{}, br[1]:{}".format(img.shape, ul[0], ul[1], br[0], br[1]))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    #from H*W*C to C*H*W  
    new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    return new_img


def get_preds_fromhm(hm, center=None, scale=None, reference_scale=200):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    #print("hm 0:{}, 1:{}, 3:{}, 3:{}, center:{}, scale:{}".format(hm.size(0), hm.size(1), hm.size(2), hm.size(3), center, scale))
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)
    
    '''
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                print("p-b:{}".format(preds[i, j]))
                preds[i, j].add_(diff.sign_().mul_(.25))
                print("p-a:{}".format(preds[i, j]))

    preds.add_(-.5)
    '''

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = to_torch(transform(preds[i, j], center[i], scale[i], reference_scale[i], [hm.size(2),hm.size(3)], True))

    return preds, preds_orig


def shuffle_lr(parts, pairs=None):
    """Shuffle the points left-right according to the axis of symmetry
    of the object.

    Arguments:
        parts {torch.tensor} -- a 3D or 4D object containing the
        heatmaps.

    Keyword Arguments:
        pairs {list of integers} -- [order of the flipped points] (default: {None})
    """
    if pairs is None:
        pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                 62, 61, 60, 67, 66, 65]
    if parts.ndimension() == 3:
        parts = parts[pairs, ...]
    else:
        parts = parts[:, pairs, ...]

    return parts


def flip(tensor, is_label=False):
    """Flip an image or a set of heatmaps left-right

    Arguments:
        tensor {numpy.array or torch.tensor} -- [the input image or heatmaps]

    Keyword Arguments:
        is_label {bool} -- [denote wherever the input is an image or a set of heatmaps ] (default: {False})
    """
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_label:
        tensor = shuffle_lr(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor
