import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from _functools import reduce
from sklearn.model_selection._split import train_test_split
from builtins import iter



def countNB(fPath):
    childfiles = os.listdir(fPath)
    files = []
    idx = []
    for ele in childfiles:
        tmp = os.listdir(os.path.join(fPath, ele)) 
        files.extend([os.path.join(fPath, ele, fl) for fl in tmp])
    idx = [np.load(ele).shape[0] for ele in files]    
    cumIdx = np.cumsum(idx)
    return files, idx


class RadarEcho(data.Dataset):

    def __init__(self, allFile, cum_frames_idx, fileIDx, n_frames_input, n_frames_output, num_objects,
                 chanels):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(RadarEcho, self).__init__()
        
        self.length = len(fileIDx)
        self.files = allFile
        self.dataset = fileIDx
        self.cum_frames_idx = cum_frames_idx

        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.chanels = chanels
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        which = self.dataset[idx]
        you = np.nonzero(self.cum_frames_idx > which)[0][0]
        you_in = which - self.cum_frames_idx[you - 1] if you > 0 else which
        fr_to = [you_in *  length ,  length * (you_in + 1)]
        file = self.files[you]
        data = np.load(file)[fr_to[0]:fr_to[1]]
        data = data[...,:self.chanels]
        data = data.transpose(0, 3, 1, 2)
        # if self.transform is not None:
        #     images = self.transform(images)
        
        input = data[:self.n_frames_input]
        output = data[self.n_frames_input:length,::self.chanels,...]

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).float()
        input = torch.from_numpy(input / 255.0).float()  # ##  S,C,H,W
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length

    
if __name__ == '__main__':
    
    n_frames_input = 8
    n_frames_output = 3
    fPath = 'data'
    files, idx = countNB(fPath)
    seq_len = n_frames_input + n_frames_output
    frames = np.array(idx) // seq_len
    cum_frames_idx = np.cumsum(frames)
    allIDX = np.arange(cum_frames_idx[-1])
    trainIdx, testIdx = train_test_split(allIDX, test_size=0.2)
    trainFolder = RadarEcho(files, cum_frames_idx,trainIdx,
                          n_frames_input=n_frames_input,
                          n_frames_output=n_frames_output,
                          num_objects=[3],chanels=3)
    validFolder = RadarEcho(files, cum_frames_idx,testIdx,
                              n_frames_input=n_frames_input,
                              n_frames_output=n_frames_output,
                              num_objects=[3],chanels=3)
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=5,
                                          shuffle=True)
    validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=7,
                                          shuffle=False)
    dddxxx = iter(trainLoader).next()
    dddooo = iter(validLoader).next()
    print(dddxxx[1].shape)
    print(dddxxx[2].shape)
    
