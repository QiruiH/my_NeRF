#coding=UTF-8
'''
主干文件，负责传参、模型训练
'''
import os,sys
import numpy as np
import time
import torch
import configargparse
import cv2
import imageio
from tqdm import tqdm, trange

from load_data import load_cv_data
from load_data import load_blender_data
from nerf import *
from embedder import *
from render_and_rays import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#对参数进行定义、初始化，返回参数集parser
def config_parser():

    #基本参数
    import configargparse
    parser = configargparse.ArgumentParser()
    #生成config.txt文件
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    #指定实验名称
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    #指定输出目录
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    #指定输入数据目录
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    #训练相关参数
    #网络层数
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    #每层通道数（神经元个数）
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    #fine network和network的区别？
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    #每个梯度阶段随机光束数目
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    #指数学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    #并行处理光束数量，溢出则减少
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    #what is pts？
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    #一次只从一张图片中获取随机光线
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    #不从保存的模型中加载权重
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    #为粗网络重新加载特定权重
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    #渲染参数
    #每条射线的粗样本数
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    #细样本和粗样本的区别？
    #每条射线额外的细样本数
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    #抖动
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    #多分辨率(3D)
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    #(2D)
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    #对输出加上的噪声方差
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    #不优化，重新加载
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    #渲染测试集
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    #降采样加速渲染
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    #训练相关
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    #crop占比
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    #数据集选择 voxel：体素
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    #大数据适用：选取部分数据
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    #降采样因子
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    #记录与存储相关参数
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    #get config parser
    parser=config_parser()
    #get arguments
    args=parser.parse_args()

    K=None

    #load cv data
    if args.dataset_type == 'handcream':
        
        images, poses, render_poses, hwf, i_split = load_cv_data()
        i_train, i_val, i_test = i_split
        i_val = i_val[::5]
        i_test = i_test[::5]

        #near & far
        near=0.1196
        far=0.5572
    
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    #将H,W规范为整数
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    #拼接出内参矩阵K（或许为了精确，可以直接用内参矩阵）
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    #这个参数表示拿测试集的pose来做渲染
    if args.render_test:
        render_poses = np.array(poses[i_test])
    
    #存log的部分并且复制config文件
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # 拼接出参数指定的存储log的路径
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        # config路径
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    #create nerf model
    #生成网络
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    #加入边界
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)


    #render_only
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None
            
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            #直接返回，训练结束
            return

    # Prepare raybatch tensor if batching random rays
    #N_rand表示每批有多大
    N_rand = args.N_rand
    use_batching = not args.no_batching
    #批处理，从一批图像中采取光线
    #采取的光线数目都是一定的，只不过要么从一批中取要么从单张中取

    if use_batching:
        #批处理，对所有图片的所有点都计算get_rays，弄到一起后shuffle
        # For random ray batching
        print('get rays')
        #（138,2,400,400,3）2：前面是rays_o，后面是rays_d
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        #（138,3,400,400,3）3：加了一个rgb，刚好也是（400,400,3）
        #None是一个常用的给array/tensor添加维度的手段
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        #调换一下位置 (138,400,400,3,3)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        #只把train的挑出来，下面就都是只有train的了，只有100个
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        #（100*400*400,3,3）
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

        # Move training data to GPU
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    
    pose = torch.Tensor(poses).to(device)

    #默认迭代200000次
    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    #trange=tqdm(range()),可以打印进度条
    for i in trange(start, N_iters):
        #下面都是每个epoch做的事情

        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            #分批加载rays
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3]
            batch = torch.transpose(batch, 0, 1) #[3,B,3] 第一列和第二列互换
            batch_rays, target_s = batch[:2], batch[2] #把rgb和对应像素点颜色分离

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                #已经过一遍之后再打乱一次
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            #随机选择一张图像进行训练
            img_i = np.random.choice(i_train) #随机选图片的index
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            # 如果要换内参矩阵的话
            # Ks
            # K = Ks[i].copy()
            # K[:2] *= 0.5

            if N_rand is not None:
                #和get_rays_up实现过程类似，不过是对单张图生成rays
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                #precrop这两个参数都和中心裁剪有关
                #计算笛卡尔坐标
                
                if i < args.precrop_iters:
                    #计算图像中心像素的坐标
                    #在lego中precrop_iters=500，procrop_frac=0.5
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    '''
                    补充meshgrid用法：
                    函数输入两个数据类型相同的一维向量
                    输出两个二维向量，行数为第一个向量的元素个数，列数为第二个向量的元素个数
                    第一个输出填充第一个输入向量中的元素，各行元素相同
                    第二个输出填充第二个输入向量中的元素，各列元素相同
                    '''
                    coords = torch.stack(
                        #这个生成的就是这些点的长宽坐标
                        torch.meshgrid(
                            #(H/4,3H/4-1,H/2) 输出H/2个点
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    #计算所有坐标的笛卡尔坐标
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                target = torch.Tensor(target).to(device)
                

                '''
                #先把每张图片的物体挑出来，再在物体所在处采样，定位边框后要扩大范围，注意不要越界
                #首先通过erode和dilate把缝隙删掉
                #消去杂质
                border=11
                kernel=np.ones((border,border),np.uint8)
                #开始框定边界
                #此时msk记的是一个False和True的矩阵，False的地方就是背景
                msk=(target==0).sum(axis=-1)!=3
                msk=msk.astype(np.uint8)
                msk_erode=cv2.erode(msk.copy(),kernel)
                msk_dilate=cv2.dilate(msk_erode,kernel)
                no_mask=msk_dilate.nonzero()
                #开始选边界，直接找出x和y坐标的最大最小值
                x_min,x_max=no_mask[0].min(),no_mask[0].max()
                y_min,y_max=no_mask[1].min(),no_mask[1].max()
                #留出来一些空余
                pudding=40
                mh, mw = target.shape[:2]
                x_min=max(x_min-pudding, 0)
                x_max=min(x_max+pudding, mh-1)
                y_min=max(y_min-pudding, 0)
                y_max=min(y_max+pudding, mw-1)

                #可视化
                #plt.imshow(target[x_min:x_max,y_min:y_max])
                #plt.savefig('test.png')
                target_ = target
                target = torch.Tensor(target).to(device)

                #__import__('ipdb').set_trace()
                #生成坐标，原来是先H后W，结合下面的reshape，这里应该是先y后x
                coords = torch.stack(torch.meshgrid(torch.linspace(x_min, x_max-1, x_max-x_min), torch.linspace(y_min, y_max-1, y_max-y_min)), -1)
                '''

                #coords就是笛卡尔坐标
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)（x，y）
                #选一批光线，也是从400*400个点中选取N_rand个点
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                #取出这些点的坐标
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                #select_coords[:,0]是横坐标，另一个是纵坐标，这样可以取出这些点的rays_o
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                #从先前挑中的图片中选出这些点
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        #__import__('ipdb').set_trace()
        #得到生成的光线后开始对光线进行渲染工作
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        # 优化
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s) #与target图像计算MSE损失
        trans = extras['raw'][...,-1]
        loss = img_loss #所以loss是预测的rgb与实际rgb的MSE
        psnr = mse2psnr(img_loss) #将MSE损失转化为PSNR

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # 反向传播
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        '''
        动态更新学习率
        args.lrate_decay：学习率指数衰减
        '''
        decay_rate = 0.1 #衰减率
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        # 完成一轮
        global_step += 1


if __name__=='__main__':

    #test load
    #imgs, poses, intrins, render_poses, hw, i_split=load_cv_data()
    #
    #i_train,i_val,i_test=i_split
    #print("imgs.shape: {}".format(imgs.shape))
    #print("poses.shape: {}".format(poses.shape))
    #print("intrins.shape: {}".format(intrins.shape))
    #print("render_poses: {}".format(render_poses.shape))
    #print("hw: {}".format(hw))
    
    #print(i_train)
    #print(i_val)
    #print(i_test)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()