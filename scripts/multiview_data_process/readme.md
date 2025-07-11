

The scene requires undistortion processing using the provided camera parameters, undistortion parameters, and scripts for frame 0. Use the script to copy parameters to other frames and call COLMAP for undistortion. The steps are as follows:

1. Convert the multi-view video (.mp4) into images (.png). Make sure to modify the video path (line 34) and the number of viewpoints (line 32):

   ```bash
   python video2stream.py
   ```

2. Next, you need to perform undistortion on **frame 0**:

   ```bash
   python convert.py -s /amax/dataset/VRU_gz/frame000000
   ```

3. Copy the camera parameters from frame 000000 to all other frames, and copy the undistortion parameters to the parent directory:

   ```bash
   python copy_cams.py --source /amax/dataset/VRU_gz/frame000000 --scene /amax/dataset/VRU_gz
   ```

4. Perform undistortion for other frames based on the copied parameters, and store the results in the `images` folder:

   ```bash
   python convert_frames.py -s /amax/dataset/VRU_gz
   ```



