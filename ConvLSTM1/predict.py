import sys
import cv2
import shutil
from sklearn.model_selection._split import train_test_split
 
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import os
 
import torch
from dataset import RadarEcho 
from encoder import Encoder
from decoder import Decoder
from model import ED
import numpy as np
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
import argparse

### Config
def data_transform(data, sequence_length,step_size):  ### 数据 初步变换，拼装成指定大小的数据格式
    print('data len:',len(data))       #4172
    print('sequence len:',sequence_length)     #50
 
    result = []
    for index in range((len(data) - sequence_length)//step_size +1):
        result.append(data[index*step_size: index*step_size + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label
    print('result len:',len(result))   
    print('result shape:',np.array(result).shape)   
    result = np.array(result )
    return result

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
parser.add_argument('-clstm',               ### 选择模型convLSTM
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',                ### 选择模型convGRU
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--modelFile', type=str, help='modelfile')### 模型文件全路径
parser.add_argument('--data-root', type=str,default='data',### 数据跟路径
                    help='root of data directory.')
parser.add_argument('-testCount',         ### 测试数据大小
                    default=6, type=int,
                    help='test sample size')
parser.add_argument('-frames_input',        ### 输入序列
                    default=8, type=int,
                    help='number of input frames')
parser.add_argument('-frames_output',       ### 预测序列
                    default=3, type=int,
                    help='number of predict frames')
parser.add_argument('--channel',       ###  指定输入的层数默认是3
                    default=3,
                    type=int,
                    help='the count of layers')
                    
args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
fPath = args.data_root 
testNB = args.testCount 
files, idx = countNB(fPath)
n_frames_input , n_frames_output = args.frames_input,args.frames_output
seq_len = n_frames_input+ n_frames_output       ### 序列总长
frames = np.array(idx) // seq_len               ### 求出每个文件的帧数
cum_frames_idx = np.cumsum(frames)              ### 累计每个文件的帧数
allIDX = np.arange(cum_frames_idx[-1])          ### 总帧数
sampleIdx  =  np.random.choice(allIDX,testNB)       ### 抽取指定数量的测试数据
print(cum_frames_idx[-1])
sampleIdx  =  np.array([21,22,23,24,25,26,27,28,29,30])
testNB = len(sampleIdx)
validFolder = RadarEcho(files, cum_frames_idx,sampleIdx, n_frames_input=n_frames_input,
                      n_frames_output=n_frames_output,
                      num_objects=[3],chanels=args.channel)          ### 加载数据集

validLoader =  DataLoader(validFolder,  batch_size=testNB, shuffle=False)

if args.convlstm:
    encoder_params = convlstm_encoder_params                ### 定义特征提取模型结构
    decoder_params = convlstm_decoder_params                 ### 定义预测模型结构
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
orgINParam = encoder_params[0][0]['conv1_leaky_1']
encoder_params[0][0]['conv1_leaky_1'] = [args.channel] + orgINParam[1:]

    
    
encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()      ### 定义编码器
decoder = Decoder(decoder_params[0], decoder_params[1],n_frames_output).cuda()## 定义解码器
encoder_forecaster = ED(encoder, decoder).to(DEVICE)            ### 组合模型
encoder_forecaster.load_state_dict(torch.load(args.modelFile)['state_dict'])    ### 加载模型参数
print(sampleIdx)
 
_, real_batch,in_batch,_,_  =  iter(validLoader).next()
lossFun = torch.nn.MSELoss().cuda()

# from torchstat import stat
# import torchvision.models as models
# stat(encoder_forecaster, (3, 400, 400))# 3 层数，400这个长宽，8可以随意

with torch.no_grad():
    output = encoder_forecaster(in_batch.cuda())                #### 测试预测结果
output = torch.clamp(output, 0.0, 1.0)
output = output.cpu()
output = output*255
shape = output.shape
outputArray = output.reshape(-1,shape[2],shape[3],shape[4])
outputArray = outputArray.permute(0,2,3,1)
np.save('predictOutput.npy',outputArray.numpy())

base_dir = 'predict'                    ### 预测结果保存文件夹
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

to_pil_image = transforms.ToPILImage()
for bth in range(testNB):      # B,S*1C*H*W
     
    
        prd_seq_img = [to_pil_image(ele) for ele in output[bth]]    ### 将每个样本转化为图片
        tth_seq_img = [to_pil_image(ele) for ele in real_batch[bth]]     ### 将每个样本转化为图片
        
#         image.show()
#         image.save(os.path.join(base_dir,fname[i]  ))  ### 保存gif

        for one in range(len(prd_seq_img)):
            prd_seq_img[one].save(os.path.join(base_dir,f'batch{bth}_{one}_prd.png'  ) )       #### 保存为gif文件
            tth_seq_img[one].save(os.path.join(base_dir,f'batch{bth}_{one}_real.png'  ) )

#         prd_seq_img[0].save(os.path.join(base_dir,f'batch{bth}_prd.gif'  ), save_all=True, append_images=prd_seq_img[1:])       #### 保存为gif文件
#         tth_seq_img[0].save(os.path.join(base_dir,f'batch{bth}_real.gif'  ), save_all=True, append_images=tth_seq_img[1:])
print(' job completed ........')

