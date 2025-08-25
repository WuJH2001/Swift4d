import concurrent.futures
import gc
import glob
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as Trans
from tqdm import tqdm
import sys
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import torchvision.transforms.functional as TF

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses):

    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    video_images_path = video_path.split('.')[0]
    image_path = os.path.join(video_images_path,"images")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
        while video_frames.isOpened():
            ret, video_frame = video_frames.read()
            if ret:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = Image.fromarray(video_frame)
                if downsample != 1.0:
                    
                    img = video_frame.resize(img_wh, Image.LANCZOS)
                img.save(os.path.join(image_path,"%04d.png"%count))

                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
            else:
                break
      
    else:
        images_path = os.listdir(image_path)
        images_path.sort()
        
        for path in images_path:
            img = Image.open(os.path.join(image_path,path))
            if downsample != 1.0:  
                img = img.resize(img_wh, Image.LANCZOS)
                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
        
    video_frames.release()
    print(f"Video {video_path} processed.")
    return None

# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] , img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def sort_by_image_name( image):
    return int(image.split('.')[0][-2:])
    
class Colmap_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        cam_extrinsics,
        cam_intrinsics,
        downsample,
        split ,
    ):

        self.poses = []
        self.split = split
        self.downsample = downsample
        self.transform = Trans.ToTensor()
        self.focal = []
        self.cameras = len(self.split)
        # If you are training a scene from the MeetRoom dataset, the following settings are sufficient. 
        #If you are training with the VRU dataset, you need to train the model every 20 frames, and you can divide the frames into segments such as 0–20, 20–40, and so on.
        self.first_frame = 0
        self.final_frame = 300

        for idx, key in enumerate(cam_extrinsics):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
            sys.stdout.flush()

            extr = cam_extrinsics[idx]

            intr = cam_intrinsics[extr.camera_id]


            height = intr.height
            width = intr.width

            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)

            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)

            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            self.poses.append( ( R , T ) )
            self.focal.append([focal_length_x, focal_length_y])

            self.FovX = FovX
            self.FovY = FovY
            self.height = height
            self.width  = width



        self.img_wh = ( width , height ) 
        self.root_dir = datadir

        

        self.load_meta()
        print(f"    meta data loaded, total image:{len(self)}")



    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.

        frame_paths = glob.glob(os.path.join(self.root_dir, "frame*"))  # root_dir 是数据集
        frame_paths = sorted(frame_paths)
        # videos = [videos[i] for i in range(len(videos)) if i in self.split] 
        # breakpoint()
        self.N_frames =   self.final_frame - self.first_frame  # len(frame_paths)
        self.val_poses = None
        self.image_paths , self.image_poses, self.image_times  = self.load_images_path(frame_paths)
        self.std_frames = []

        self.calc_std(self.image_paths, os.path.join(self.root_dir, f"stds_{self.first_frame}_{self.final_frame-1}") )  # 


    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
    

    def calc_std(self, video_path_root, std_path_root):

        if os.path.exists(std_path_root):
            std_files = glob.glob(os.path.join(std_path_root, "*.npy"))
            std_files = sorted(std_files)

            for i, std_file in enumerate(std_files):
                std_frame = np.load(std_file)
                std_frame = torch.from_numpy(std_frame)
                # std_frame = torch.nn.functional.interpolate(std_frame.unsqueeze(0).unsqueeze(0), size=(self.height, self.width), mode='bilinear', align_corners=False)
                self.std_frames.append(std_frame)
                
            return
        os.makedirs(std_path_root)
        # print(video_path_root)
        print("-----------------caculating std-------------------")
        progress_bar = tqdm(range(0,  self.cameras), desc="caculating std")

        for i in range( self.cameras):
            progress_bar.update(1)
            # images_path = video_path.split('.')[0]
            # frame_paths = glob.glob(os.path.join(os.path.join(images_path, "images"), "*.png"))
            # frame_paths.sort()
            frames = []
            frame_paths = self.image_paths[i*self.N_frames: (i+1) * self.N_frames]
            for j, fp in enumerate(frame_paths):

                frame = Image.open(fp).convert('RGB')
                frame_tensor = TF.to_tensor(frame).unsqueeze(0).cuda()   # Convert to tensor and move to device
                frames.append(frame_tensor)

            frames_tensor = torch.cat(frames, dim=0)  # Stack frames [300,h,w,3] into tensor [300,3,h,w]
            std_map = frames_tensor.float().std(dim=0).mean(dim=0)  # Calculate std_map on GPU
            std_map_blur = cv2.GaussianBlur(std_map.cpu().numpy(), (31, 31), 0).astype(np.float32)  # Gaussian blur on CPU
            std_name = "%03d_std.npy" % i
            std_path = os.path.join(std_path_root, std_name)


            self.std_frames.append(torch.from_numpy(std_map_blur))
            np.save(std_path, std_map_blur)

            # 清理 GPU 缓存
            del frames_tensor, std_map
            torch.cuda.empty_cache()


        # for i , video_path in enumerate(video_path_root):

        #     progress_bar.update(1)
        #     images_path = video_path.split('.')[0]
        #     frame_paths = glob.glob(os.path.join(os.path.join(images_path,"images"), "*.png"))
        #     frames = []
        #     for j , fp in enumerate(frame_paths):
        #         frame = Image.open( fp ).convert('RGB')
        #         frame = np.array(frame, dtype=np.float32) / 255.
        #         frames.append(frame)
        #     frame = np.stack(frames, axis=0)  # [300,h,w,3]
        #     std_map = frame.std(axis=0).mean(axis=-1)  # [h,w]
        #     std_map_blur = (cv2.GaussianBlur(std_map, (31, 31), 0)).astype(np.float32) # [1014,1352] 高斯模糊
        #     std_path = os.path.join(std_path_root, os.path.basename(video_path[:-4])+'_std.npy')

        #     self.std_frames.append(std_map_blur)
        #     np.save(std_path, std_map_blur)


        # progress_bar.close()




    def load_images_path(self,frames_paths):

        image_paths = []
        image_poses = []
        image_times = []


        for index, frame_path in enumerate(frames_paths):
            if index < self.final_frame and index >= self.first_frame:
                image_path = os.path.join( frame_path, "images" )
                images_path = os.listdir(image_path)
                images_path.sort()

                for idx, path in enumerate(images_path):
                    if idx in self.split:
                        image_paths.append(os.path.join(image_path,path))
                    
        image_paths.sort(key= sort_by_image_name)

        for idx in range ( self.cameras):  # 12
            for i in range(self.N_frames):   # 300
                image_poses.append(self.poses[idx])
                image_times.append(i/self.N_frames)

        return image_paths , image_poses, image_times
    

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self,index):

        img = Image.open(self.image_paths[index])
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)

        return img, self.image_poses[index], self.image_times[index] , self.std_frames[index//self.N_frames]
    

    def load_pose(self,index):
        return self.image_poses[index]




