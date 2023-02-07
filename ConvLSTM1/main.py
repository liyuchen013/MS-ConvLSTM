#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
from sklearn.model_selection._split import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from dataset import RadarEcho
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse


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


parser = argparse.ArgumentParser()
parser.add_argument('-clstm',  ### 选择模型convLSTM
                    '--convlstm',
                    default='True',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',  ### 选择模型convGRU
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--data-root', type=str, default='data',  ### 数据跟路径
                    help='root of data directory.')
parser.add_argument('--batch_size',  ### 小批量数据大小
                    default=2,
                    type=int,
                    help='mini-batch size')

parser.add_argument('--channel',  ###  指定输入的层数默认是3
                    default=6,
                    type=int,
                    help='the count of layers')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',  ### 输入序列
                    default=10,
                    type=int,
                    help='number of input frames')
parser.add_argument('-frames_output',  ### 预测序列
                    default=10,
                    type=int,
                    help='number of predict frames')
parser.add_argument('-epochs', default=60, type=int, help='sum of epochs')  ### 训练周期
args = parser.parse_args()

random_seed = 3
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Training arguments:')
for k, v in args._get_kwargs():
    print('\t{}: {}'.format(k, v))

save_dir = './save_model/'  ### 模型保存文件夹名

#####        cv2.imshow('xxxx',cv2.cvtColor(tt,cv2.COLOR_BGR2GRAY))
fPath = args.data_root  #### 数据文件夹
files, idx = countNB(fPath)
seq_len = args.frames_input + args.frames_output  ### 序列总长
frames = np.array(idx) // seq_len  ### 求出每个文件的帧数
cum_frames_idx = np.cumsum(frames)  ### 累计每个文件的帧数
allIDX = np.arange(cum_frames_idx[-1])
trainIdx, testIdx = train_test_split(allIDX, test_size=0.2)  ### 切割训练集测试集
trainFolder = RadarEcho(files, cum_frames_idx, trainIdx,  ### 读取训练集
                        n_frames_input=args.frames_input,
                        n_frames_output=args.frames_output,
                        num_objects=[3], chanels=args.channel)
validFolder = RadarEcho(files, cum_frames_idx, testIdx,  ### 读取测试集
                        n_frames_input=args.frames_input,
                        n_frames_output=args.frames_output,
                        num_objects=[3], chanels=args.channel)

trainLoader = torch.utils.data.DataLoader(trainFolder,  ###  训练集加载器
                                          batch_size=args.batch_size,
                                          shuffle=True)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

if args.convlstm:
    encoder_params = convlstm_encoder_params  ### 定义特征提取模型结构
    decoder_params = convlstm_decoder_params  ### 定义预测模型结构
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

orgINParam = encoder_params[0][0]['conv1_leaky_1']
encoder_params[0][0]['conv1_leaky_1'] = [args.channel] + orgINParam[1:]


def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()  ### 定义特征提取模型
    decoder = Decoder(decoder_params[0], decoder_params[1], args.frames_output).cuda()  ### 定义预测模型
    net = ED(encoder, decoder)
    run_dir = './runs/'
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)  ### 定义辅助结束类早起停止
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(net)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):  ### 间歇性训练的加载上次模型参数
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()  ## 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=args.lr)  # 定义优化器
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,  # 定义学习率调整器
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):  ### 循环训练模型
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        net.train()
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):  ### batch读取训练
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            pred = net(inputs)  # B,S,C,H,W     ### 模型预测
            loss = lossfunction(pred, label)  ###计算损失
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)  ## 防止梯度爆炸的钳住梯度
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():  ### 模型测试在测试集上测试模型
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)  ### 一个周期上训练集所有数据的平均损失
        valid_loss = np.average(valid_losses)  ### 一个周期上测试集所有数据的平均损失
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler 调整学习率
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)  ###模型的测试保存
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:  ###保存所有周期上的训练损失
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:  ###保存所有周期上的测试损失
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
