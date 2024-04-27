#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render,render2, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import argparse
import options
import utils
from dataset.dataset_motiondeblur import *
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR,LambdaLR
from timm.utils import NativeScaler
from losses import CharbonnierLoss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
import math
from losses import CharbonnierLoss
from torchvision.transforms import ToPILImage
import Spline
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    addp=0
    ######### U Model ###########
    model_restoration = utils.get_arch(args)
    model_restoration.train()
    ######### Optimizer ###########
    start_epoch = 1
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model_restoration.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
    else:
        raise Exception("Error optimizer...")
    ######### DataParallel ########### 
    model_restoration = torch.nn.DataParallel (model_restoration) 
    model_restoration.cuda() 
    ######### Scheduler ###########
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_3)
    # scheduler.step()
    if args.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = args.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepoch-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()
    loss_scaler = NativeScaler()
    ######### Resume ########### 
    if args.resume: 
        path_chk_rest = args.pretrain_weights 
        print("Resume from "+path_chk_rest)
        utils.load_checkpoint(model_restoration,path_chk_rest) 
        # start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
        # lr = utils.load_optim(optimizer, path_chk_rest) 

        # # for p in optimizer.param_groups: p['lr'] = lr 
        # # warmup = False 
        # # new_lr = lr 
        # # print('------------------------------------------------------------------------------') 
        # # print("==> Resuming Training with learning rate:",new_lr) 
        # # print('------------------------------------------------------------------------------') 
        # for i in range(1, start_epoch):
        #     scheduler.step()
        # new_lr = scheduler.get_lr()[0]
        # print('------------------------------------------------------------------------------')
        # print("==> Resuming Training with learning rate:", new_lr)
        # print('------------------------------------------------------------------------------')

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepoch-start_epoch+1, eta_min=1e-6)   
        ######### Loss ###########
    criterion = CharbonnierLoss().cuda() 
    ######### DataLoader ###########
    print('===> Loading datasets')
    img_options_train = {'patch_size':args.train_ps}
    train_dataset = get_training_data(args.train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, 
            num_workers=args.train_workers, pin_memory=False, drop_last=False)
    len_trainset = train_dataset.__len__()

    print("Sizeof training set: ", len_trainset) 


    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians2 = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    scene2 = Scene(dataset, gaussians2)
    viewpoint_stack0=scene.getTrainCameras().copy()
    gaussians.training_setup(opt)
    gaussians2.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    idx=0
    print("begin restore:")
    with torch.no_grad():
        for traincam in viewpoint_stack0:
            rgb_gt = traincam.restored_image.unsqueeze(0)
            rgb_noisy, mask = expand2square(rgb_gt.cuda(), factor=128)
            
            with torch.cuda.amp.autocast():
                rgb_restored = model_restoration(rgb_noisy)
                rgb_restored = torch.masked_select(rgb_restored, mask.bool()).reshape(1, 3, rgb_gt.shape[2], rgb_gt.shape[3])
                rgb_restored = torch.clamp(rgb_restored, 0, 1).squeeze()
            
            traincam.restored_image = rgb_restored
            to_pil = ToPILImage()
            
            rgb_restored_pil = to_pil(rgb_restored.cpu().detach())
            
            
            result_dir = './restoredimages/'  
            os.makedirs(result_dir, exist_ok=True)
            image_path = os.path.join(result_dir, f'{idx:03d}.png')
            idx=idx+1
            rgb_restored_pil.save(image_path)
        torch.cuda.empty_cache()
    
    print("end restore")
    for iteration in range(first_iter, opt.iterations + 1): 
        
        # original_images=[]      
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        if iteration > args.train_iter:

                gaussians.update_learning_rate(iteration)
                gaussians2.update_learning_rate(iteration)
                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()
                    gaussians2.oneupSHdegree()
                # Pick a random Camera
                # for i in range(args.batch_size):
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                ######Random viewpoint B
                viewpoint_cam.world_view_transform_b=viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.world_view_transform_b_se3=Spline.SE3_to_se3(viewpoint_cam.world_view_transform_b).cuda()
                low, high = 0.0001, 0.005
                rand = (args.high - args.low) * torch.rand(viewpoint_cam.world_view_transform_b_se3.shape[0], 6) + args.low
                rand = rand.to(viewpoint_cam.world_view_transform_b_se3.device)
                viewpoint_cam.world_view_transform_b_se3 = viewpoint_cam.world_view_transform_b_se3 + rand
                viewpoint_cam.world_view_transform_b=Spline.se3_to_SE3(viewpoint_cam.world_view_transform_b_se3).cuda()
                viewpoint_cam.world_view_transform_b_se3 = viewpoint_cam.world_view_transform_b_se3.reshape(-1)[:6]
                viewpoint_cam.world_view_transform_b=viewpoint_cam.world_view_transform_b[0]
                last_row = viewpoint_cam.world_view_transform[-1, :]
                viewpoint_cam.world_view_transform_b = torch.cat([viewpoint_cam.world_view_transform_b, last_row.unsqueeze(0)], dim=0)

                    # original_images.append(viewpoint_cam.original_image.unsqueeze(0))
                    # original_images = torch.cat(original_images, dim=0)
                # A restored
            
                rgb_gt=viewpoint_cam.original_image.unsqueeze(0)
                rgb_noisy, mask = expand2square(rgb_gt.cuda(), factor=128)
                with torch.cuda.amp.autocast():
                    rgb_restored = model_restoration(rgb_noisy)
                #     print(rgb_noisy.size())
                #     print(rgb_restored.size())

                rgb_restored = torch.masked_select(rgb_restored,mask.bool()).reshape(1,3,rgb_gt.shape[2],rgb_gt.shape[3])
                rgb_restored = torch.clamp(rgb_restored,0,1).squeeze()
                # A Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # A' Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg2 = render(viewpoint_cam, gaussians2, pipe, bg)
                image2, viewspace_point_tensor2, visibility_filter2, radii2 = render_pkg2["render"], render_pkg2["viewspace_points"], render_pkg2["visibility_filter"], render_pkg2["radii"]

                #A Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1_A = l1_loss(image, rgb_restored)
                loss1 = (1.0 - opt.lambda_dssim) * Ll1_A + opt.lambda_dssim * (1.0 - ssim(image, rgb_restored))
                gaussians.optimizer.zero_grad(set_to_none = True)
                loss1.backward()
                # A Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # A' Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1_B = l1_loss(image2, gt_image)
                loss2 = (1.0 - opt.lambda_dssim) * Ll1_B + opt.lambda_dssim * (1.0 - ssim(image2, gt_image))
                gaussians2.optimizer.zero_grad(set_to_none = True)
                loss2.backward()
                #  A' Optimizer step
                if iteration < opt.iterations:
                    gaussians2.optimizer.step()
                    gaussians2.optimizer.zero_grad(set_to_none = True)

                iter_end.record()
                    # B Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg_b = render2(viewpoint_cam, gaussians, pipe, bg)
                image_b, viewspace_point_tensor_b, visibility_filter_b, radii_b = render_pkg_b["render"], render_pkg_b["viewspace_points"], render_pkg_b["visibility_filter"], render_pkg_b["radii"]
                    # B' Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg_b2 = render2(viewpoint_cam, gaussians, pipe, bg)
                image_b2, viewspace_point_tensor_b2, visibility_filter_b2, radii_b2 = render_pkg_b2["render"], render_pkg_b2["viewspace_points"], render_pkg_b2["visibility_filter"], render_pkg_b2["radii"]

                # B' restored
                rgb_gt2=image_b2.unsqueeze(0)

                rgb_noisy2, mask2 = expand2square(rgb_gt2.cuda(), factor=128)
                with torch.cuda.amp.autocast():
                    rgb_restored2 = model_restoration(rgb_noisy2)
                # rgb_restored2 = torch.masked_select(rgb_restored2,mask2.bool()).reshape(1,3,rgb_gt2.shape[2],rgb_gt2.shape[3])
                # rgb_restored2 = torch.clamp(rgb_restored2,0,1).squeeze()  
                image_b=image_b.unsqueeze(0)
                rgb_restored_b, mask2 = expand2square(image_b.cuda(), factor=128)

                # U Optimizer
                optimizer.zero_grad()
                # loss_U = criterion(rgb_restored_b, rgb_restored2.cuda())
                L1_U=l1_loss(rgb_restored_b,  rgb_restored2.cuda())
                loss_U = (1.0 - args.lambda_dssim) *  L1_U + args.lambda_dssim * (1.0 - ssim(rgb_restored_b, rgb_restored2))
                # loss_U=L1_U
                if(iteration%100==0):  
                    print("Current Learning Rate:", current_lr)        
                print("loss_U:",loss_U)
                loss_scaler(
                    loss_U, optimizer,parameters=model_restoration.parameters())
                torch.cuda.empty_cache()
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log and save
                training_report(tb_writer, iteration, Ll1_A, loss1, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # # A Densification
                # if iteration < opt.densify_until_iter:
                #     # Keep track of max radii in image-space for pruning
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians.reset_opacity()
                # #  A' Densification
                # if iteration < opt.densify_until_iter:
                #     # Keep track of max radii in image-space for pruning
                #     gaussians2.max_radii2D[visibility_filter2] = torch.max(gaussians2.max_radii2D[visibility_filter2], radii2[visibility_filter2])
                #     gaussians2.add_densification_stats(viewspace_point_tensor2, visibility_filter2)

                #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #         gaussians2.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians2.reset_opacity()

                # U save
                if (iteration in saving_iterations):
                    save_path = os.path.join(args.save_dir, "model_iter_{}.pth".format(iteration))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
                    torch.save({
                        'iteration': iteration,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, save_path)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        else :
            
                gaussians.update_learning_rate(iteration)
                gaussians2.update_learning_rate(iteration)
                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()
                    gaussians2.oneupSHdegree()
                # Pick a random Camera
                # for i in range(args.batch_size):
                if not viewpoint_stack:
                    viewpoint_stack = viewpoint_stack0.copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                    # original_images.append(viewpoint_cam.original_image.unsqueeze(0))
                    # original_images = torch.cat(original_images, dim=0)
                # A restored
            

                # A Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # A' Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg2 = render(viewpoint_cam, gaussians2, pipe, bg)
                image2, viewspace_point_tensor2, visibility_filter2, radii2 = render_pkg2["render"], render_pkg2["viewspace_points"], render_pkg2["visibility_filter"], render_pkg2["radii"]

                #A Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1_A = l1_loss(image, gt_image)
                loss1 = (1.0 - opt.lambda_dssim) * Ll1_A + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss1.backward()
                # A Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # A' Loss
                gt_image2 = viewpoint_cam.original_image.cuda()
                Ll1_B = l1_loss(image2, gt_image2)
                loss2 = (1.0 - opt.lambda_dssim) * Ll1_B + opt.lambda_dssim * (1.0 - ssim(image2, gt_image2))
                loss2.backward()
                #  A' Optimizer step
                if iteration < opt.iterations:
                    gaussians2.optimizer.step()
                    gaussians2.optimizer.zero_grad(set_to_none = True)

                iter_end.record()
                # Log and save
                # training_report(tb_writer, iteration, Ll1_A, loss1, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # A Densification
                if iteration < args.densify_until_iter:
                # if iteration < opt.densify_until_iter:
             
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

#                         stack = []
#                         gaussian_clone = []
#                         gaussian_split = []
#                         clone_vector=[]
#                         split_vector=[]
#                         gaussians_copy=gaussians
#                         with torch.no_grad():
#                             for v in viewpoint_stack0:
#                                 pipe.debug = False
#                                 bg = torch.rand((3), device="cuda") if opt.random_background else background
#                                 render_pkg = render(v, gaussians_copy, pipe, bg)
#                                 image_copy, viewspace_point_tensor_copy, visibility_filter_copy, radii_copy = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
#                                 size_threshold = 20 if iteration > opt.opacity_reset_interval else None
#                                 clone_mask, split_mask = gaussians_copy.densify_and_prune_test(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

#                                 stack.append((v.world_view_transform, clone_mask, split_mask))
#                             for item in stack:
#                                 world_view_transform, clone_mask, split_mask = item
#                                 if len(clone_vector)==0:
#                                     clone_vector=clone_mask
#                                 else:
#                                     clone_vector &= clone_vector
#                                 if len(split_vector)==0:
#                                     split_vector=split_mask
#                                 else:
#                                     split_vector &= split_vector

                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # gaussians.densify_and_prune2(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,clone_vector,split_vector)
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                #  A' Densification
                # if iteration < opt.densify_until_iter:
                if iteration < args.densify_until_iter:
                # if iteration < 3000:
                    # Keep track of max radii in image-space for pruning
                    gaussians2.max_radii2D[visibility_filter2] = torch.max(gaussians2.max_radii2D[visibility_filter2], radii2[visibility_filter2])
                    gaussians2.add_densification_stats(viewspace_point_tensor2, visibility_filter2)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians2.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians2.reset_opacity()


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log1 = 0.4 * loss1.item() + 0.6 * ema_loss_for_log
            ema_loss_for_log2 = 0.4 * loss2.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss1": f"{ema_loss_for_log1:.{7}f}", "Loss2": f"{ema_loss_for_log2:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

                # if (iteration in checkpoint_iterations):
                #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
                    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Add directory to sys.path
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name, './dataset/'))
    sys.path.append(os.path.join(dir_name, '.'))




    # Command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 20000,29000,29100,29200,29300,29400,29500,29600,29700,29800,29900, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--lamda_dssim', type=float, default=0.5, help='lambda_dssim')
    parser.add_argument('--low', type=float, default=0.0001, help='rand low')
    parser.add_argument('--high', type=float, default=0.005, help='rand high')
    parser.add_argument('--train_iter', type=int,default=29000, help='train iter')
    # args = parser.parse_args(sys.argv[1:])
       
    # U parser
    # global settings
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
    parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
    parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
    parser.add_argument('--dataset', type=str, default ='SIDD')
    parser.add_argument('--pretrain_weights',type=str, default='./weights/Uformer_B.pth', help='path of pretrained_weights')
    parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
    parser.add_argument('--lr_initial', type=float, default=1e-7, help='initial learning rate')
    parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--gpu', type=str, default='6,7', help='GPUs')
    parser.add_argument('--arch', type=str, default ='Uformer_B',  help='archtechture')
    parser.add_argument('--mode', type=str, default ='denoising',  help='image restoration mode')
    parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
 
    
    # args for saving 
    parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
    parser.add_argument('--save_images', action='store_true',default=False)
    parser.add_argument('--env', type=str, default ='_',  help='env')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

    # args for Uformer
    parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
    parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
    parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
    parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
    parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')

    # args for vit
    parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
    parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
    parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
    parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
    parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
    parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
    parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
    parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
    
    # args for training
    parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
    parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--train_dir', type=str, default ='./datasets/SIDD/train',  help='dir of train data')
    parser.add_argument('--val_dir', type=str, default ='./datasets/SIDD/val',  help='dir of train data')
    parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
    parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

    # ddp
    parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')#不需要赋值，启动命令 torch.distributed.launch会自动赋值
    parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
    parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")

    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
        # Set GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
