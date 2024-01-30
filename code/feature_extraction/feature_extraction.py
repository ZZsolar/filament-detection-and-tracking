# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:32:54 2023

@author: 92875
"""

#输出每轨第一张图的速度暗条速度场

#导入包

import cv2
import glob,os
import numpy as np
from PIL import Image
from astropy.io import fits
from dateutil.parser import parse

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
import skimage.morphology as sm

import re
from datetime import datetime
import warnings

from sklearn.cluster import DBSCAN
import networkx as nx
from skimage import draw
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import matplotlib.gridspec as gridspec

#定义UNet模型
class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers
    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.
    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(LkReLU_num)
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(LkReLU_num)
        # Dropout
        self.drop = nn.Dropout(Drop_num)

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        
        x = self.second(x)
        x = self.act2(x)

        x = self.drop(x)
        return x


class DownSample(nn.Module):
    """
    ### Down-sample
    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample
    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map
    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class UNet(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(512, 1024)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        #activation
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)
            
        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        #activation
        x = self.activation(x)
        return x
    
    
#定义其他函数
#读取文件
def readchase(file):
    hdu = fits.open(file)
    try:
        data = hdu[0].data.astype(np.float32)
        header = hdu[0].header
        
    except:
        header = hdu[1].header
        data = hdu[1].data
        
    if len(data.shape) != 3:
        raise TypeError('file ', str(file), 'is not Chase\'s file, please use other function to read.')
     
    hdu_time = datetime.strptime(header['DATE_OBS'], "%Y-%m-%dT%H:%M:%S")
    if hdu_time < datetime.strptime('2023-04-18', "%Y-%m-%d"):
        cy = header['CRPIX1']
        cx = header['CRPIX2']
    else:
        cx = header['CRPIX1']
        cy = header['CRPIX2']

    #改变数组大小、日心位置
    data = data[:, int(cy-1023):int(cy+1025), int(cx-1023):int(cx+1025)]
    if data.shape != (118,2048,2048):
        raise TypeError('Chase file ', file, 'is corrupted, please check.')
    
    cx = 1023 + cx - int(cx)
    cy = 1023 + cy - int(cy)
    header['CRPIX1'] = 1023 + cx - int(cx)
    header['CRPIX2'] = 1023 + cy - int(cy)
    header['NAXIS1'] = 2048
    header['NAXIS2'] = 2048
    
    return data, header

#unet模型输入输出数据预处理
def input_process(file):
    if isinstance(file, str):
        try:
            img = Image.open(file).convert('L')
        except:
            try:
                hdu_data, hdu_header = readchase(file)
                img = hdu_data[68]
            except:
                hdu_data, hdu_header = fits.open(file)
                img = hdu_data
    elif isinstance(file, np.ndarray):
        img = file
    else:
        raise TypeError('can not recognize file type, only image, fits and numpy.ndarray are supported')

    if img.max() <= 0:
        raise ValueError('image\'s max value is 0')
    elif img.shape != (2048,2048):
        raise ValueError(f'image\'s shape is {img.shape}, not adaptive')
    
    img = img / img.mean()
        
    return transforms.ToTensor()(img).unsqueeze(0).to(device=device, dtype=torch.float32)

def output_process(output):
    output = (output >= 0.5)*1
    return output
        
def model_predict(model, input_img):
    model.eval()
    with torch.no_grad():
        output = model(input_img)
    
    return output.cpu().squeeze().numpy()

#图像处理：标记
def get_labels(loc, result):
    img = np.zeros(shape = (2048,2048))
    for i in range(loc.shape[0]):
        x,y = loc[i]
        img[x,y] = result[i]
    
    return np.array(img)

def mark_connection(img, min_distance=1.5):
    loc = np.array(img.nonzero()).T

    dmodel = DBSCAN(eps = min_distance, min_samples= 2)
    dmodel.fit(loc)
    dresult = dmodel.labels_ + 1
    
    labels = get_labels(loc, dresult)
    
    return labels

def select_img(labels, min_size):
    img = (labels > 0)*1
    N = int(labels.max())
    #print(N)
    from scipy.ndimage import labeled_comprehension
    labelsum = labeled_comprehension(img, labels,index = np.arange(1,N+1), func =  sum, out_dtype = int, default = 0)
    #print(labelsum.shape ,labelsum.max())
    j = 1
    nimg = np.zeros(shape = img.shape)
    for i in range(N):
        if labelsum[i] >= min_size:
            nimg = nimg + (labels == i+1)*j
            j = j + 1
            
    return nimg

def select_filament(img, min_distance = 10, min_size = 100):
    label_filament = mark_connection(img)
    img_filament = select_img(label_filament, min_size = 10)
    label_filament = mark_connection((img_filament>0)*1, min_distance)
    img_filament = select_img(label_filament, min_size)
    
    return (img_filament > 0)*1

def filament2skeleton(filament_img):
    #最长最短路径算法
    #更改数据格式
    if filament_img.any() > 1:
        filament_img = (filament_img > 0)*1
    filament_img= filament_img.astype('uint8')
    if len(filament_img.shape) != 2 and filament_img.max() != 1:
        raise TypeError('image\'s type is error')
    #获取骨架
    from skimage import morphology
    skeleton = morphology.skeletonize(filament_img).astype('uint8')
    
    #获取分叉点和端点
    kernel1 = np.array([[1,0,0],[0,1,1],[0,1,0]])
    kernel2 = np.array([[1,0,1],[0,1,0],[1,0,0]])
    kernel3 = np.array([[1,0,1],[0,1,0],[0,1,0]])
    result = np.zeros(shape = filament_img.shape)
    for i in range(4):
        result = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, np.rot90(kernel1, i)) + result
        result = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, np.rot90(kernel2, i)) + result
        result = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, np.rot90(kernel3, i)) + result
    
    cpts = np.array(np.where(result == 1)).T
    #查找端点
    skeleton1 = cv2.filter2D(skeleton, -1, np.ones(shape = (3,3))) * skeleton
    epts = np.array(np.where(skeleton1 == 2)).T
    
    pts = np.concatenate((cpts, epts), axis = 0)
    
    #寻找最长最短路
    #断开分叉点
    skeleton_ = skeleton
    for cpt in cpts:
        skeleton_[cpt[0]-1:cpt[0]+2, cpt[1]-1:cpt[1]+2] = np.zeros(shape = (3,3))
        
    #连通域标记
    from skimage import measure
    skeleton_label = measure.label(skeleton_)
    #作单连通图
    graph = nx.Graph()
    lines_info = {}
    for i in range(1, skeleton_label.max()+1):
        index = np.array(np.where(skeleton_label == i)).T
        line_length = index.shape[0]
        lines_info[f'line_{i}'] = index
        line_pts = []
        for j in range(pts.shape[0]):
            pt = pts[j].reshape(1,2)
            pt_dist = ((((index - pt)**2).sum(axis = 1))**0.5).min()
            if pt_dist < 3:
                line_pts.append(j)
        pt_num_1, pt_num_2 = line_pts
        graph.add_edge(f'pt_{pt_num_1}', f'pt_{pt_num_2}', weight = line_length, name = f'line_{i}')
    
    #处理较近的分叉点
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=5, min_samples=1)
    dbscan.fit(cpts)
    cpt_labels = dbscan.labels_
    for j in range(cpt_labels.max()+1):
        index = np.array(np.where(cpt_labels == j)).reshape(-1)
        if index.shape[0] == 2:
            i = i+1
            graph.add_edge(f'pt_{index[0]}', f'pt_{index[1]}', weight = 4, name = f'line_{i}')
            lines_info[f'line_{i}'] = np.array(draw.line(cpts[index[0],0],cpts[index[0],1],cpts[index[1],0],cpts[index[1],1])).T
    
    #最短路算法
    dist_matrix = nx.floyd_warshall(graph, weight='weight')
    # 查找距离最远的两个点
    max_distance = 0
    max_nodes = ('', '')
    for source_node, distances in dist_matrix.items():
        for target_node, distance in distances.items():
            if distance > max_distance and source_node != target_node:
                max_distance = distance
                max_nodes = (source_node, target_node)
    #输出最长最短路
    shortest_path = nx.shortest_path(graph, source=max_nodes[0], target=max_nodes[1])
    new_skeleton = np.zeros(shape = skeleton.shape)
    # 输出最短路径的节点和路径名
    for i in range(len(shortest_path)-1):
        start_node = shortest_path[i]
        end_node = shortest_path[i+1]
        edge_attrs = graph.get_edge_data(start_node, end_node)
        edge_name = edge_attrs['name']
        line_index = lines_info[edge_name]
        new_skeleton[line_index[:,0], line_index[:,1]] = 1
        if i > 0:
            node_num = int(start_node.split('_')[1])
            cnode_y = pts[node_num,0]
            cnode_x = pts[node_num,1]
            new_skeleton[cnode_y-1:cnode_y+2, cnode_x-1:cnode_x+2]=1
    
    start_pt = int(shortest_path[0].split('_')[1])
    end_pt = int(shortest_path[-1].split('_')[1])
    
    start_pt = pts[start_pt].tolist()
    end_pt = pts[end_pt].tolist()
    return morphology.skeletonize(new_skeleton).astype('uint8'), start_pt, end_pt

def sort_line(line_index, start_point):
    
    try:
        line_index = line_index.tolist()
    except:
        line_index = list(line_index)
        
    if start_point not in line_index:
        return print("start_point is not in line")
        
    sort_line = []
    start_point = np.array(start_point)

    while len(line_index) > 0:
        start_point = np.array(start_point)
        line_index = np.array(line_index)
        dist_array = (((line_index - start_point)**2).sum(axis = 1))**0.5
        point_index = dist_array.argmin()
        
        start_point = line_index[point_index].tolist()
        line_index = line_index.tolist()
        
        sort_line.append(start_point)
        line_index.remove(start_point)
        
    return sort_line

def straightening_img(image, mask, sorted_points):
    sorted_points = np.array(sorted_points)
    width = int(mask.sum() / sorted_points.shape[0] * 1.5)
    length = 0
    len_list = []
    len_list.append(0)
    for i in range(sorted_points.shape[0]-1):
        length += np.linalg.norm(sorted_points[i+1] - sorted_points[i])
        len_list.append(length)
    len_list = np.array(len_list)
    
    from scipy.interpolate import interp1d
    fx = interp1d(len_list, sorted_points[:,1], kind='cubic', bounds_error = False, fill_value = 'extrapolate')
    fy = interp1d(len_list, sorted_points[:,0], kind='cubic', bounds_error = False, fill_value = 'extrapolate')

    from scipy.misc import derivative
    dx = derivative(fx, len_list, dx = 0.001)
    dy = derivative(fy, len_list, dx = 0.001)
    
    from scipy.ndimage import gaussian_filter1d
    smoothed_dx = gaussian_filter1d(dx, sigma=5)
    smoothed_dy = gaussian_filter1d(dy, sigma=5)
    
    new_image = np.zeros(shape = (int(width)*2 + 1, sorted_points.shape[0]))

    for i in range(sorted_points.shape[0]):#
        dxl = smoothed_dx[i]
        dyl = smoothed_dy[i]
        
        nd = np.array([dxl, -dyl]) / (dxl**2+dyl**2)**0.5
        
        normal = sorted_points[i,:] + np.arange(-int(width), int(width)+1).reshape(-1,1) * nd.reshape(1,2)
        normal = normal.astype('uint')
        new_image[:,i] = image[normal[:,0], normal[:,1]]
        
    return new_image
    

#计算速度，方向：红移为正，蓝移为负
#云模型反演
#数据，背景谱，光谱，标记区域，vc，lamta_range，lamta0，求v,s,tao0,w

def func_cloud_contrast(i0, v,s,tao0,w):
    l0 = lamta0
    l = lamta_range
    
    tao = tao0 * np.exp(-(l - l0 - v / vc)**2/(w / 0.0484867877626943)**2)
    
    return (i0 - s) / i0 * (1 - np.exp(-tao))

def inversion_cloud(i0,ic,qs_center):
    global lamta0
    lamta0 = qs_center
    xdata = i0
    ydata = (i0 - ic) / i0
    
    p0 = (0, 200, 1, 0.5)
    bounds = [[-1e5, 0, 0, 0],[1e5, 1000, 10, 3]]
    
    try:
        popt, pcov = curve_fit(f = func_cloud_contrast, xdata = xdata, ydata = ydata, p0 = p0, bounds = bounds, maxfev=3000)
    except:
        popt=[0,0,0,0]
    
    return np.array(popt)
            
            
#U-Net模型导入
Drop_num = 0.2
LkReLU_num = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    
print('model using', device)
    
model = UNet(1,1)
model.load_state_dict(torch.load('./model/unet_model.pth', map_location=device))

model.to(device)
model.eval()

#忽略warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

#参数定义
min_distance = 1.5#暗条间最小距离
min_size = 64#
#云模型参数
global lamta_range, vc
lamta_range = np.arange(118)
from scipy import constants
vc = constants.c * 0.0484867877626943 / 6562.82
qs_width = 30


global_path = "./"

HA_file_path = global_path + "images/HA/"
filament_file_path = global_path + "images/filament/"
cloud_file_path = global_path + "images/cloud/"
dopmap_file_path = global_path + "images/dopmap/"
submap_file_path = global_path + 'images/submap/'

error_info = []

#遍历文件
folder_path = './chasefile'

file_list = glob.glob(os.path.join(folder_path, "*HA.fits"))


for i in range(len(file_list)):
    
    try:
        hdu_file = file_list[i]
        hdu_data, hdu_header = openchase(hdu_file)
    except:
        i = i+1
        continue
    
    starttime = datetime.now()
    print(hdu_file + ' started at ' + starttime.strftime("%Y-%m-%d %H:%M:%S"))
    
    
    hdu_HA = hdu_data[69]
    hdu_time = parse(hdu_file.split("RSM")[1][:15])
    image_name = hdu_time.strftime("%Y_%m%d_%H%M")

    #Unet模型预测
    hdu_input = input_process(hdu_HA)
    hdu_output = model_predict(model, hdu_input)
    hdu_output = output_process(hdu_output)
    
    
    #图像处理
    try:
        slt_output = select_filament(hdu_output, min_distance=1.5, min_size=min_size)
        mark_output = mark_connection(slt_output, min_distance=1.5)
        mark_output = sm.closing(mark_output,sm.disk(10))
        mask = sm.dilation(mark_output,sm.disk(10))
    except:
        continue
    
    filament_num = int(mark_output.max())
    
    time1 = datetime.now()
    print(f'stage 1 costed: {str(time1 - starttime)}')
    
    #云模型拟合多普勒速度
    map_doppler = np.zeros(shape = (4, 2048,2048))
    
    #创建子文件夹保存暗条多普勒速度图
    submap_name = submap_file_path + hdu_time.strftime("%Y_%m%d_%H%M")

    if not os.path.exists(submap_name):
        os.makedirs(submap_name)
    
    for j in range(1, filament_num+1):
        mark_fila = (mark_output == j)*1
        mark_dila = sm.dilation(mark_fila,sm.disk(10))
        index_filament = np.array(np.where(mark_fila == 1))
        index_qs = np.array(np.where((mark_dila - mark_fila) == 1))
        x1 = index_qs[0].min()-10
        x2 = index_qs[0].max()+10+1
        y1 = index_qs[1].min()-10
        y2 = index_qs[1].max()+10+1
        
        sub_cloud = np.zeros(shape = (4, x2-x1, y2-y1))
        sub_mask = mark_fila[x1:x2, y1:y2]
        
        qs_spec = hdu_data[:, index_qs[0], index_qs[1]].mean(axis = 1)
        flip_qs_spec = (max(qs_spec[69-qs_width], qs_spec[69+qs_width]) - qs_spec[69-qs_width:70+qs_width])
        qs_center = (np.arange(69-qs_width, 70+qs_width) * flip_qs_spec).sum() / flip_qs_spec.sum()
        
        # 并行计算反演云模型
        num_cores = -1  # 使用所有可用核心数，也可以指定具体的核心数
        inversion_results = Parallel(n_jobs=num_cores)(
            delayed(inversion_cloud)(qs_spec, hdu_data[:,index_filament[0,k],index_filament[1,k]], qs_center)
            for k in range(index_filament.shape[1])
        )
        
        inversion_results = np.array(inversion_results)
        sub_cloud[:,index_filament[0,:]-x1,index_filament[1,:]-y1] = inversion_results.T
        sub_cloud = sub_cloud * (sub_cloud[2] >= 0.1)
        map_doppler[:,x1:x2,y1:y2] = sub_cloud
        
        # 创建一个包含四个子图的画布
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
        # 设置子图的标题和内容
        axes[0, 0].set_title('v')
        axes[0, 0].imshow(sub_cloud[0,:,:], vmin = -1e4, vmax = 1e4 ,cmap = 'bwr', origin = 'lower')
    
        axes[0, 1].set_title('s')
        axes[0, 1].imshow(sub_cloud[1,:,:], vmin = 0, vmax = 400 ,cmap = 'afmhot', origin = 'lower')
    
        axes[1, 0].set_title('tao0')
        axes[1, 0].imshow(sub_cloud[2,:,:], vmin = 0, vmax = 3,cmap = 'gray', origin = 'lower')
    
        axes[1, 1].set_title('w')
        axes[1, 1].imshow(sub_cloud[3,:,:], vmin = 0, vmax = 1, cmap = 'gray', origin = 'lower')
    
        # 调整子图的间距
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)
    
        # 保存图像
        fig.savefig(submap_name + f'/submap{image_name}_{j:02d}.png')
        
        try:
            new_skeleton, start_pt, end_pt = filament2skeleton((mark_fila==1)*1)
            line_index = np.array(np.where(new_skeleton == 1)).T
            sorted_points = sort_line(line_index, start_pt)
            sorted_points = np.array(sorted_points)
        
            sq_v = straightening_img(map_doppler[0], (mark_fila==1)*1, sorted_points)
            sq_tao = straightening_img(map_doppler[2], (mark_fila==1)*1, sorted_points)
            
            tao_index = np.array(np.where(sq_tao >= 0.1)).T
            v_vert = sq_v[tao_index[:,0], tao_index[:,1]]
            v_dstc = tao_index[:,0]
        
            map_width = int((sq_v.shape[0]-1)/2 / 5 / 1.04)+1
            map_length = int((sq_v.shape[1]+1) / 50 / 1.04)
        except:
            continue

        # 创建一个10x10大小的画布，并将其分为五个子图
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
    
        # 定义子图的位置
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0])
        ax3_top = fig.add_subplot(ax3_gs[0])
        ax3_bottom = fig.add_subplot(ax3_gs[1])
        ax4 = fig.add_subplot(gs[1, 1])
    
        # 设置子图的标题
        ax1.set_title('H\u03B1')
        ax2.set_title('LOS velocity')
        ax3_top.set_title('Straightened')
        ax3_bottom.set_title('Along the Spine')
        ax4.set_title('Perpendicular to the Spine')
    
        # 在子图上绘制内容
        ax1.imshow(hdu_HA[x1:x2,y1:y2], vmin = 0, vmax = hdu_HA.mean()*3, cmap = 'afmhot', origin = 'lower')
        ax2_img = ax2.imshow(sub_cloud[0,:,:], vmin = -1e4, vmax = 1e4 ,cmap = 'bwr', origin = 'lower')
        ax3_top.imshow(sq_v, vmin = -10000, vmax = 10000, cmap = 'bwr', origin = 'lower')
        ax3_bottom.plot(map_doppler[0, sorted_points[:,0], sorted_points[:,1]], 'k.-')
        ax4.plot(v_dstc-(sq_v.shape[0]-1)/2, v_vert, 'k.', markersize = 1)
    
        #contour、colorbar、坐标轴设置
        ax1.set_xticks([])
        ax1.set_yticks([])
        contour1 = ax1.contour((mark_fila[x1:x2,y1:y2] == 1)*1, colors = 'green' ,linewidths = 1)
        contour2 = ax1.contour((mark_dila[x1:x2,y1:y2] == 1)*1, colors = 'red' ,linewidths = 1)
    
        ax2.set_xticks([])
        ax2.set_yticks([])
        cbar = fig.colorbar(ax2_img, ax=ax2)
        cbar.ax.set_title('(m/s)')
        cbar.set_ticks([-1e4, -7.5e3, -5e3, -2.5e3, 0, 2.5e3, 5e3, 7.5e3, 1e4])  # 设置刻度位置
        cbar.set_ticklabels([i*2.5 for i in range(-4,5)])
    
        ax3_top.set_xticks([i*1.04*50 for i in range(map_length+1)])
        ax3_top.set_xticklabels([i*50 for i in range(map_length+1)]) # 设置x刻度
        ax3_top.set_xlabel('length (")')
        ax3_top.set_ylim((sq_v.shape[0]+1)/2 - map_width*1.04*5, (sq_v.shape[0]+1)/2 + map_width*1.04*5)
        ax3_top.set_yticks([(sq_v.shape[0]+1)/2 + i*10*1.04 for i in range(-int(map_width/2),int(map_width/2)+1)])
        ax3_top.set_yticklabels([i*10 for i in range(-int(map_width/2),int(map_width/2)+1)]) # 设置y刻度
        ax3_top.set_ylabel('width (")')
    
        ax3_bottom.set_xticks([i*1.04*50 for i in range(map_length+1)])
        ax3_bottom.set_xticklabels([i*50 for i in range(map_length+1)]) # 设置x刻度
        ax3_bottom.set_xlabel('length (")')
        ax3_bottom.set_ylim(-2e4, 2e4)
        ax3_bottom.set_yticks([-2e4, -1.5e4, -1e4, -0.5e4, 0, 0.5e4, 1e4, 1.5e4, 2e4])  # 设置y刻度
        ax3_bottom.set_yticklabels([i*5 for i in range(-4,5)]) # 设置x刻度
        ax3_bottom.set_ylabel('LOS velocity (km/s)')
    
        ax4.set_xlim(-(sq_v.shape[0]-1)/2, (sq_v.shape[0]-1)/2)
        ax4.set_xticks([i*5*1.04 for i in range(-map_width,map_width+1)])  # 设置x刻度
        ax4.set_xticklabels([i*5 for i in range(-map_width,map_width+1)]) # 设置x刻度
        ax4.set_xlabel('distance from spine (")')
        ax4.set_ylim(-2e4, 2e4)
        ax4.set_yticks([-2e4, -1.5e4, -1e4, -0.5e4, 0, 0.5e4, 1e4, 1.5e4, 2e4])  # 设置y刻度
        ax4.set_yticklabels([i*5 for i in range(-4,5)]) # 设置x刻度
        ax4.set_ylabel('LOS velocity (km/s)')
    
        # 调整子图的间距
        plt.tight_layout()
    
        fig.savefig(submap_name + f'/dist{image_name}_{j:02d}.png')
        

    time2 = datetime.now()
    print(f'stage 2 costed: {str(time2 - time1)}')
    
    np.save(cloud_file_path + f'/cloud_{image_name}.npy', map_doppler)
    plt.imsave(HA_file_path + f'HA_{image_name}.png',hdu_HA, vmin = 0, vmax = hdu_HA.mean()*3, cmap = 'afmhot', origin = 'lower')
    plt.imsave(filament_file_path + f'filament_{image_name}.png', slt_output, vmin = 0, vmax = 1 ,cmap = 'gray', origin = 'lower')
    
    fig,ax = plt.subplots()
    ax_img = ax.imshow(map_doppler[0], vmin = -10000, vmax = 10000, cmap = 'bwr', origin = 'lower')
    ax_cbar = fig.colorbar(ax_img)
    fig.savefig(dopmap_file_path + f'dopmap_{image_name}.png')
    
    endtime = datetime.now()
    print(hdu_file + ' was completed at ' + endtime.strftime("%Y-%m-%d %H:%M:%S"))
    print(hdu_file + ' coasted ' + str(endtime - starttime))
            
            
            
#输出错误信息
print(error_info)
