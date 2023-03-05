#coding=UTF-8
'''
负责load数据
'''
'''
这个文件用来处理cv大作业的数据，将其处理为NeRF需要的格式
大作业数据(handcream)：img，外参，内参
需要得到：images, poses, render_poses, hwf, i_split

不用模仿原版的写法，最好搞清楚作用和功能之后自己写
'''

'''
数据的差别：
cv作业中提供的为w2c外参矩阵，但NeRF中需要的是c2w
cv作业中提供了内参矩阵，直接返回内参矩阵即可
cv中val的数据，test无pose(三个都设成train)
'''

'''
步骤：
将train/test/val的img和pose全都汇总在一起
i_split记录三者的下标（按理说记三个数就行）
算长宽和焦距
生成render_poses
进行half_res处理，暂时不做
'''

import os
import numpy as np
import imageio
import torch
import torch.nn.functional as F
import cv2
import json
from tqdm import tqdm, trange

#不是特别明白这一块在干嘛，先粘过来用一下
#rot_phi和rot_theta对r33的处理是重复的，不知道为什么
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    #根据传入的theta等等生成一个pose
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
c2b = np.linalg.inv(b2c)
def load_cv_data(data_dir='handcream',testskip=1,half_res=False):
    '''
    返回值：imgs, poses, intrins, render_poses, [H,W], i_split
    '''
    types=['images_mask','poses','intrins']
    #先创建list方便append，最后一起转化为数组
    imgs=[]
    poses=[]
    intrins=[]
    count=0 #所有数据都是一样的，所有只需要记一个
    for t in types:
        data_path=os.path.join(data_dir+"/"+t)
        files=sorted(os.listdir(data_path)) #get all file_names
        #for i in trange(0,3):
            #print(i)
        for file in files:
            file_name=data_path+"/"+file
            if t=='images_mask':
                #图片
                imgs.append(imageio.imread(file_name))
                #if i==0: #只在第一遍记录
                count+=1
            elif t=='poses':
                #外参，需要求逆
                w2c=np.loadtxt(file_name)
                c2w=np.linalg.inv(w2c)
                c2w=c2w@c2b
                poses.append(c2w)
            else:
                #内参，直接返回内参，不用算f了
                #print(file_name)
                intrin=np.loadtxt(file_name)
                #因为mask的图像长和宽都缩小了一倍，所以内参也要改变
                intrin[0,:]=intrin[0,:]/2 #第一行
                intrin[1,:]=intrin[1,:]/2 #第二行
                intrins.append(intrin)
    #list→array
    imgs=(np.array(imgs)/255.).astype(np.float32)
    poses=np.array(poses).astype(np.float32)
    intrins=np.array(intrins).astype(np.float32)
    #求focal
    focal=np.mean([[inx[0,0],inx[1,1]] for inx in intrins])

    #生成i_split
    #i_split=[]
    #for i in range(0,3):
    #    i_split.append(np.array(range(count*i,count*i+count)))
    #换一种方法，跳着生成
    i_test=np.arange(imgs.shape[0])[::8]
    i_val=i_test
    i_train=np.array([i for i in np.arange(int(imgs.shape[0])) if
                    (i not in i_test and i not in i_val)])
    i_split=[]
    i_split.append(i_train)
    i_split.append(i_test)
    i_split.append(i_val)
    #H,W
    H,W=imgs[0].shape[:2]

    #生成render_poses
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    return imgs, poses, render_poses, [H,W,focal], i_split



'''
获取图片、位姿、图片的长宽与焦距、划分训练集测试集
RGBA，A是指透明度
'''
def load_blender_data(basedir, half_res=False, testskip=1):
    '''
    basedir:数据集文件夹
    half_res:是否对图像进行半裁剪
    testskip:挑选测试数据集的跳跃步长
    '''
    splits = ['train', 'val', 'test']
    #是个字典
    metas = {}
    for s in splits:
        #fp：filepath，根据三个后缀分别找到相应的json文件
        #basedir：./data/nerf_syntheticc/lego
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        #meta中存的是相应json的数据，也是字典
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
        
        #以指定步长读取内容
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    #给split标序号
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    #（138,800,800,4）
    imgs = np.concatenate(all_imgs, 0)
    #（138,4,4）外参
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    #算焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    #40个4*4，渲染的pose
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        #（138,400,400,4）138是个数
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split
