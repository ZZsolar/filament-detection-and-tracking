# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:54:14 2023

@author: 92875
"""

#导入包
import random
import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime

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
    
class CustomDataset_train(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # 数据增强操作
        # 上下翻转
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 45°旋转
        if random.random() > 0.2:
            angle = random.uniform(0,45)  # 随机选择一个角度
            image = image.rotate(angle)
            mask = mask.rotate(angle)
            
        image = transforms.ToTensor()(image).to(torch.float32)
        mask = transforms.ToTensor()(mask).to(torch.float32)
        
        image = image / image.mean()
        mask = mask / mask.max() * 1

        return image, mask
    
class CustomDataset_val(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        image = transforms.ToTensor()(image).to(torch.float32)
        mask = transforms.ToTensor()(mask).to(torch.float32)
        
        image = image / image.mean()
        mask = mask / mask.max() * 1

        return image, mask
    
#参数定义
num_epochs = 500
batch_size = 1
num_workers = 1
learning_rate = 1e-5
in_channels = 1
out_channels = 1
FL_gamma = 2
FL_alpha = 4
Drop_num = 0.2
LkReLU_num = 0.1

model_save_dir = './model/unet_model.pth'
model_new_save_dir = './model/unet_model_new.pth'

train_imgs = './data/train/imgs'
train_masks = './data/train/masks'
val_imgs = './data/valid/imgs'
val_masks = './data/valid/masks'
test_imgs = './data/test/imgs'
test_masks = './data/test/masks'

pre_new_dir = './output/pre_new/'
pre_save_dir = './output/predict/'

#创建数据集实例并进行划分
train_set = CustomDataset_train(train_imgs, train_masks)
val_set = CustomDataset_val(val_imgs, val_masks)

#创建数据加载器
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, smoothing=1e-32, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.smoothing=smoothing

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # 计算正样本和负样本的权重
        alpha = torch.tensor([self.alpha, 1]).to(inputs.device)

        # 计算pt函数
        pt = inputs

        # 计算Focal Loss
        loss = (-alpha[0] * targets * (1 - pt) ** self.gamma * torch.log(pt + self.smoothing) 
                -alpha[1] * (1-targets) * pt ** self.gamma * torch.log(1 - pt + self.smoothing))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss 

#模型训练

#选取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using ', device)

#指定GPU
if torch.cuda.is_available():
    torch.cuda.set_device(1)#指定gpu0

#定义模型
model = UNet(in_channels, out_channels)
#model.load_state_dict(torch.load(model_new_save_dir, map_location=device))
model.to(device)

#模型优化器选择（Adam:自适应学习率优化器）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#损失函数选择（FocalLoss）
criterion = FocalLoss(gamma = FL_gamma, alpha = FL_alpha)#alpha = 0.9
#criterion = nn.BCELoss()

best_loss = float('inf')  # 设置一个初始最佳验证集损失值
train_losses = []
val_losses = []
train_acces = []
val_acces = []

data_output = True

for epoch in range(num_epochs):
    starttime = datetime.now()
    model.train()
    train_loss = 0.0
    train_acc = 0.0 
    
    for images, labels in train_loader:
        # 将图像和标签移动到设备（如GPU）上
        images = images.to(device)
        labels = labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播，计算输出和损失
        outputs = model(images)

        loss = criterion(outputs, labels)

        # 反向传播，更新模型参数
        loss.backward()
        optimizer.step()

        # 累计训练损失
        train_loss += loss.item()
        #计算accuracy
        train_acc += torch.mean(outputs * labels + (1-outputs) * (1-labels)).item()
        
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    avg_train_acc = train_acc / len(train_loader)
    train_acces.append(avg_train_acc)

    # 在验证集上评估模型并保存最佳模型
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)
            val_loss += loss_val.item()
            
            val_acc += torch.mean(outputs_val * labels_val + (1-outputs_val) * (1-labels_val)).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        avg_val_acc = val_acc / len(val_loader)
        val_acces.append(avg_val_acc)
        
        torch.save(model.state_dict(), model_new_save_dir)#保存当前模型

        # 如果当前损失值低于之前的最佳损失值，则更新最佳损失值和保存的模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_dir)
            
    endtime = datetime.now()
    print(f'Epoch [{epoch+1}/{num_epochs}], coasted {str(endtime - starttime)}, \n\t\
Train Loss: {avg_train_loss:.8f}, Train Accuracy: {avg_train_acc:.8f}, \n\t\
Validation Loss: {avg_val_loss:.8f}, Validation Accuracy: {avg_val_acc:.8f}')
    
print('Training finished.')

print('Training finished.')

#训练结果可视化
#打印loss即accuracy

print('show model unet_model_test\' loss line')

plt.figure(figsize = (12,6))
plt.subplot(121)
plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(122)
plt.plot(range(1, num_epochs+1), train_acces, 'b-', label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_acces, 'r-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(f'./output/loss_{LkReLU_num}_{Drop_num}.png')
#plt.show()


# 打开文件并写入数组数据
with open('./his/train_loss1.txt', 'w') as f:
    for item in np.array(train_losses):
        formatted_item = "{:.8f}".format(item)
        f.write(formatted_item + '\n')
        
with open('./his/val_loss1.txt', 'w') as f:
    for item in np.array(val_losses):
        formatted_item = "{:.8f}".format(item)
        f.write(formatted_item + '\n')

with open('./his/train_acces1.txt', 'w') as f:
    for item in np.array(train_acces):
        formatted_item = "{:.8f}".format(item)
        f.write(formatted_item + '\n')
        
with open('./his/val_acces1.txt', 'w') as f:
    for item in np.array(val_acces):
        formatted_item = "{:.8f}".format(item)
        f.write(formatted_item + '\n')
        
        
#测试模型
#读取测试文件
img_list = glob.glob(os.path.join(test_imgs, '*.png'))
img_list.sort()

mask_list = glob.glob(os.path.join(test_masks, '*.png'))
mask_list.sort()
        
#输出最新训练模型的结果
model = UNet(in_channels, out_channels)
model.load_state_dict(torch.load(model_new_save_dir, map_location=device))
model.to(device)
model.eval()

iou = 0.0
precision = 0.0 
recall = 0.0 

with torch.no_grad():
    for i in range(len(img_list)):
        # 读取数据
        image = Image.open(img_list[i])
        mask = Image.open(mask_list[i])
        
        image = np.array(image)
        image = image / image.mean()
        image = transforms.ToTensor()(image).unsqueeze(0).to(torch.float32)
        image = image.to(device)
        
        mask = np.array(mask)
        mask = (mask > 0)*1
        # 模型预测
        output = model(image)
        output = output.cpu().squeeze().numpy()
        output = (output > 0.5)*1
        
        # 计算评估参数
        iou1 = (output * mask).sum() / ((output + mask).sum() - (output * mask).sum())
        iou += iou1
        pre1 = (output * mask).sum() / output.sum()
        precision += pre1
        rec1 = (output * mask).sum() / mask.sum()
        recall += rec1
        
        print(f'第{i}张图参数: iou: {iou1:.8f};precision: {pre1:.8f}; recall: {rec1:.8f}')
        
        # 保存预测结果
        output = output * 255
        output = output.astype(np.uint8)
        output = Image.fromarray(output)
        output.save(pre_new_dir + f'pre_{i:02d}.png')

    avg_iou = iou / len(img_list)
    avg_precision = precision / len(img_list)
    avg_recall = recall / len(img_list)
    
print(f'最新训练模型的评估参数: avg_iou: {avg_iou:.8f}; avg_precision: {avg_precision:.8f}; avg_recall: {avg_recall:.8f}')

#输出由验证集选择的模型的结果
model = UNet(in_channels, out_channels)

model.load_state_dict(torch.load(model_save_dir, map_location=device))
model.to(device)
model.eval()

iou = 0.0
precision = 0.0 
recall = 0.0

with torch.no_grad():
    for i in range(len(img_list)):
        # 读取数据
        image = Image.open(img_list[i]).convert('L')
        mask = Image.open(mask_list[i]).convert('L')
        
        image = np.array(image)
        image = image / image.mean()
        image = transforms.ToTensor()(image).unsqueeze(0).to(torch.float32)
        image = image.to(device)
        
        mask = np.array(mask)
        mask = (mask > 0)*1
        
        # 模型预测
        output = model(image)
        output = output.cpu().squeeze().numpy()
        output = (output > 0.5)*1

        # 计算评估参数
        iou1 = (output * mask).sum() / ((output + mask).sum() - (output * mask).sum())
        iou += iou1
        pre1 = (output * mask).sum() / output.sum()
        precision += pre1
        rec1 = (output * mask).sum() / mask.sum()
        recall += rec1
        
        print(f'第{i}张图参数: iou: {iou1:.8f};precision: {pre1:.8f}; recall: {rec1:.8f}')
        
        # 保存预测结果
        output = output * 255
        output = output.astype(np.uint8)
        output = Image.fromarray(output)
        output.save(pre_save_dir + f'pre_{i:02d}.png')

    avg_iou = iou / len(img_list)
    avg_precision = precision / len(img_list)
    avg_recall = recall / len(img_list)
    
print(f'由验证集选择的模型的评估参数: avg_iou: {avg_iou:.8f}; avg_precision: {avg_precision:.8f}; avg_recall: {avg_recall:.8f}')
    

