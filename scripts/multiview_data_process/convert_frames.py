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
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--skip_undistortion", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--last_frame_id", default=249, type=int)

args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

for id in range(1,args.last_frame_id+1):
# for id in range(0,1):
    frame_id = f'{id:0>6}'
    inputDir='/frame'+frame_id
    
    print("Processing "+inputDir)
                
    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + args.source_path + inputDir + " \
        --input_path " + args.source_path + "/distorted/sparse/0 \
        --output_path " + args.source_path + inputDir + " \
        --output_type COLMAP" + "--min_scale 1 --max_scale 1"      )  #  --min_scale 1 --max_scale 1        
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(args.source_path + inputDir + "/sparse")
    os.makedirs(args.source_path + inputDir + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(args.source_path, inputDir[1:], "sparse", file)
        destination_file = os.path.join(args.source_path, inputDir[1:], "sparse", "0", file)
        shutil.move(source_file, destination_file)

    if(args.resize):
        print("Copying and resizing...")
        # Resize images.
        os.makedirs(os.path.join(args.source_path, inputDir[1:]) + "/images_2", exist_ok=True)
        
        # Get the list of files in the source directory
        files = os.listdir(args.source_path + inputDir + "/images")
        # Copy each file from the source directory to the destination directory
        for file in files:
            source_file = os.path.join(args.source_path, inputDir[1:], "images", file)
            destination_file = os.path.join(args.source_path, inputDir[1:], "images_2", file)
            shutil.copy2(source_file, destination_file)
            print("Resizing " + source_file + " to " + destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)


print("Done.")