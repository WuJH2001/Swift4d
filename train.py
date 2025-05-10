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
import numpy as np
import random
import os, sys


import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from PIL import Image
import matplotlib.pyplot as plt

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False





def train_dynamic(pipe ,opt, scene,  gaussians, tb_writer, args):

    gaussians.training_setup(opt, args)
    dy_iteration = 3000
    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras()
    temp_list = get_stamp_list(viewpoint_stack,0) # 
    viewpoint_stack = temp_list.copy()
    loss = torch.tensor(0.).cuda()
    progress_bar = tqdm(range(1, dy_iteration), desc="Dynamic Training progress")
    # train dynamic 
    for iteration in range(1,  dy_iteration + 1):

        gaussians.update_learning_dy_rate(iteration) 
        loss = torch.tensor(0.).cuda()

        if not viewpoint_stack:
            viewpoint_stack =  temp_list.copy()
  
        viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, bwd_dynamic=True)
        dynamic_map = render_pkg["dynamic_map"]

        dyn_image = viewpoint_cam.dyn_image.cuda()
        loss = scene.criterion(dynamic_map, dyn_image)

        # dyn_im = dyn_image.tile(3,1,1)
        # show_image(dyn_im)

        loss.backward()



        with torch.no_grad():
            if iteration % 10 == 0:
                # tb_writer.add_scalar("loss", loss.detach(), iteration)

                progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
                progress_bar.update(10)

            if iteration == dy_iteration:
                progress_bar.close()

            if iteration < dy_iteration:
                gaussians.dy_optimizer.step()
                gaussians.dy_optimizer.zero_grad(set_to_none = True)

    # dyn_mask = ( gaussians.get_dynamic >  7.0 ).squeeze(1)  # 
    # image = render(viewpoint_cam, gaussians, pipe, background, dyn_mask2 = dyn_mask)["render"]
    # rgb_image = render_pkg["render"]

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args):
    first_iter = 0

    gaussians.training_setup(opt, args)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    

    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0) # 20
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")

    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        
        # print(iteration)
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        if iteration == 3000:
            print("123")
            pass
        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        last_max_weight_list = []

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii, max_weight_t = \
                        render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"],render_pkg["max_weight_t"]
            images.append(image.unsqueeze(0))

            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            last_max_weight_list.append(max_weight_t)
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        last_max_weight = torch.cat(last_max_weight_list,0).max(dim=0).values
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:]) * (1.0 - opt.lambda_dssim)

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        
        loss = Ll1
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)

        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                
                progress_bar.update(10)
            if iteration == final_iter - 1:
                progress_bar.close()

            # Log and save
            timer.pause()
            if  iteration == final_iter - 1 :
                print("The num of dynamic point: {}".format(torch.sum(gaussians.get_dynamic > gaussians.dynamic_thro)))
            if stage == "fine" : #  or  stage=="coarse":
                training_report(args, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            if stage == "coarse" and iteration % 100 == 0:
                test_view  = scene.getTestCameras()[0]
                image = render(test_view, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)["render"]
                gt_image = test_view.original_image.cuda()   
                psnr_ = psnr(image, gt_image).mean().double()  
                tb_writer.add_scalar(f'{stage}/psnr', psnr_.item(), iteration)

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                gaussians._max_weight = torch.max(gaussians._max_weight, last_max_weight)

                if iteration >= opt.weight_prune_from_iter and iteration % opt.weight_prune_interval == 0 : 
                    gaussians.weight_prune(opt.weight_prune_threshold)
                
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # and gaussians.get_xyz.shape[0]<360000:
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent)

                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 : # and gaussians.get_xyz.shape[0] > 200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)


                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()

                if iteration % 3000 == 0  and stage == "coarse": 
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            
    
            # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.hash_optimizer.step()  # It is only updated during the fine stage.
                gaussians.hash_optimizer.zero_grad(set_to_none = True)


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname,args):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper, opt)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                              checkpoint_iterations, checkpoint, debug_from,
                              gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer,args)
    if opt.train_dynamic:
        train_dynamic ( pipe , opt,  scene,  gaussians , tb_writer ,args)
    
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer, args)

def prepare_output_and_logger(expname):    
    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())

        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
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

def training_report(args , tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):

    # if tb_writer:
    #     tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
    #     tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    psnr_f = open("psnr.txt","a")
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = (
            {'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
            # {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras())) ] },  # test all
            # {'name': 'test', 'cameras' : [scene.getTestCameras()[0]  ] },
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # try:
                    #     if tb_writer and (idx < 5):
                    #         tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #         if iteration == testing_iterations[0]:
                    #             tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # except:
                    #     pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    psnr_ = psnr(image, gt_image, mask=None).mean().double()                    
                    psnr_test += psnr_

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                psnr_f.write( stage +" "+ str (args.hash_init_lr) +"  "+ str(args.hash_final_lr) + "  "+ str(iteration)+" "+ str(psnr_test.item()) +"\n")
                
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)

        psnr_f.close
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 3000, 7000, 10000, 13000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 3000, 7000, 10000, 13000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    # parser.add_argument("--first_frame", type=float, default = 0)
    # parser.add_argument("--final_frame", type=float, default = 300)

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    psnr_f = open("psnr.txt", "a")
    psnr_f.write("\n" + str(args.expname) + "\n")
    psnr_f.close

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args)

    # All done
    print("\nTraining complete.")
