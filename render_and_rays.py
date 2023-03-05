'''
进行render及操作rays的相关函数：
+ render_path
+ render
+ get_rays
+ get_rays_np
+ batchify_rays
+ render_rays
+ raw2outputs
+ sample_pdf
'''

import os, sys
import torch
import numpy as np
import time
import imageio
from tqdm import tqdm, trange
import torch.nn.functional as F

DEBUG=False

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

#N_importance用到
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    u.requires_grad = True
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


#将工作分成更小的份来进行，防止显存不足，只是一个空壳，主要调render_rays
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    #存所有结果的字典
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        #渲染光线，得到rgb_map等封装成的字典
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


#返回的是所有点的世界坐标和相机中心的世界坐标
def get_rays(H, W, K, c2w):
    #i为W*H的矩阵，每行相同，为行数
    #j为W*H的矩阵，每列相同，为列数
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    #将其存储在内存中变成连续的
    i = i.t()
    j = j.t()
    '''
    K[0][2]: 水平方向相机中心cx
    K[0][0]: focal*mx
    K[1][2]: 竖直方向相机中心cy
    K[1][1]: focal*my
    stack用法：
    axis=0，在第一维操作
    axis=1，在第二维操作
    axis=-1，在最后一维操作
    操作对象为二维数组时，=1和=-1效果相同
    '''
    #这里就是从像素坐标求出了相机坐标系下的坐标
    #由于列数从上到下越来越大，而blender坐标系是沿y轴向上，所以要反一下，z同理
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    #其实np.newaxis就相当于None，用来增加一个新的维度
    #得到世界坐标
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    #就是相机中心的世界坐标，所有点共用，所以直接expand就行了
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


#和get_rays一样的，只不过没有上torch，方便批处理后续操作
def get_rays_np(H, W, K, c2w):
    '''
    H,W：高和宽
    K：内参矩阵
    c2w：相机坐标系到世界坐标系的转变矩阵
    '''
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    #计算
    #每条ray在相机坐标系下的方向，（400,400,3），（x，y，z）
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    #世界中心点坐标（400,400,3）
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


#对离散的点进行积分得到对应的像素信息
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    #计算相邻两采样点z轴之间的距离（1024,63）
    dists = z_vals[...,1:] - z_vals[...,:-1]
    #转化为用科学计数法表示，提升精度
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    #计算实际距离
    #rays_d_=rays_d[...,None,:]
    #全求二范式，即长度
    #out=torch.norm(rays_d[...,None,:], dim=-1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    #rgb值，把rgb值规范到0~1之间
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    #计算透明度的值
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    #计算分配给每个采样点的权重，用来集合成为预测rgb （1024,64）
    '''
    torch.cumprod是累积乘法，得到的第n项为前n项乘起来
    这里的复现完全遵循论文中的公式，注意此时的alpha是raw2alpha得到的，为1-exp(-aibi)
    weights: Ti(1-exp(-aibi))
    '''
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    #每个ray有个预测的rgb值
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    #（1024）估计每个ray对应的点到物体的距离
    depth_map = torch.sum(weights * z_vals, -1)
    #视差 （1024）
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    #每条ray的权重加和 （1024）
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, network_fn, network_query_fn,
                N_samples, retraw=False, lindisp=False,
                perturb=0., N_importance=0, network_fine=None,
                white_bkgd=False, raw_noise_std=0.,
                verbose=False, pytest=False):
    #将各种参数分离出来
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    #N_samples是每条rays上采样的次数，t_vals就是一个长为N_samples的向量
    t_vals = torch.linspace(0., 1., steps=N_samples)

    #确定z轴的值
    #0 i1 i2 ... 1
    #__import__('ipdb').set_trace()
    # one_vals=torch.tensor(np.ones_like(t_vals.cpu().numpy()))
    # if not lindisp:
    #     #线性采样
    #     z_vals = near * one_vals + (far-near) * t_vals
    #     #换种写法，这样会导致torch和np的矛盾，下同
    # else:
    #     #用深度的inverse来采样，此时近的会采集多一些，远的会少一些
    #     z_vals = 1./(1./near * one_vals + (1./far-1./near) * t_vals)

    #确定z轴的值
    if not lindisp:
        #根据depth线性采样
        #（1024,64） 1024束光线，每个随机采样64个点
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        #按照depth的inverse线性采样
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        #分层随机采样
        # get intervals between samples
        #（1024,63）相当于同一条的ray的前一个采样点与后一个采样点相加*0.5
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        #（1024,64）与最后一个采样点拼接起来，确定上界
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        #（1024,64）与第一个采样点拼接起来，确定下界
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        #从（0,1）的均匀分布中抽取一组随机数，形状为z_vals.shape，即（1024，,6）
        t_rand = torch.rand(z_vals.shape)

        if pytest:
            #采用固定的随机数
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        #所以t_rand相当于在每个区间内的位置比例，这里算出实际采样点的z轴的值
        z_vals = lower + (upper - lower) * t_rand

    #生成光线上每个采样点的位置：pts=o+z*d，所以z解释成绝对长度更合理
    #（1024,64,3）1024个光线每条光线有64个采样点，每个采样点有（x,y,z）坐标
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    #raw = run_network(pts)
    #将参数传给network_fn中前向传播得到每个点对应的RGBA
    raw = network_query_fn(pts, viewdirs, network_fn)
    #计算得到rgb、深度图等等
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    #先不管N_importance，直接复制下来
    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        # raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)


    #把返回值封装为字典
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


#负责渲染的最上层函数
def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    if c2w is not None:
        # special case to render full image
        #渲染图像中的所有点
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        #使用视角信息，将rays_d正则化赋值给viewdirs
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]，rays_d[i]就是第i个点的世界坐标

    #ndc先跳过

    # Create ray batch
    #还是[...,3]
    # 这里的...就是计算了多少rays，不同情况不一样，有时是全部有时是部分
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    #确定边框，乘以1，现在的near和far也都是H*W的
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    #[...,8] 8：3+3+1+1 
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        #如果使用视角就加上视角
        rays = torch.cat([rays, viewdirs], -1)
    
    # Render and reshape
    #开始计算光线属性，返回一堆rgb_map等
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



#render更上层的函数，负责存图片并返回rgb和disp
def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    
    #获得长宽高和focal，如果用内参的话这要改
    H, W, focal = hwf

    #为了加速用的
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        #存图片
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    #返回render到的rgbs和disps
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps