import os, glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import join as osp
from pathlib import Path

def read_poses(pose_folder:str,
                stride:int=1,
                verbose:bool=False):
    pose_files = glob.glob(os.path.join(pose_folder, "*.txt"))
    pose_files = sorted(pose_files)
    min_frame_id = int(os.path.basename(pose_files[0]).split(".")[0][6:])
    if min_frame_id>0:
        print('Start frame id: {}'.format(min_frame_id))
    
    poses = {}
    for i in range(0,len(pose_files),stride):
        pose_file = pose_files[i]
        frame_name = os.path.basename(pose_file).split(".")[0]
        # frame_id = int(frame_name.split("-")[-1])
        T_wc = np.loadtxt(pose_file)
        # calibrated_frame_id = frame_id - frame_offset
        # poses['frame-{:06d}'.format(calibrated_frame_id)] = T_wc
        poses[frame_name] = T_wc
    
    if verbose:
        print(f"Loaded {len(poses)} poses from {pose_folder}")
    return poses, min_frame_id

def read_euro_gt_file(gt_file:str):
    """
    gt file format: 
        # timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
    Only read the first 7 columns: 
        timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z
    """
    
    transforms = {} # (timestamp_ns, T_wc)
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split(',')
            timestamp_ns = items[0]
            p_RS_R = np.array([float(x) for x in items[1:4]])
            q_RS = np.array([float(x) for x in items[4:8]])
            T_wc = np.eye(4)
            T_wc[:3, :3] = R.from_quat(q_RS).as_matrix()
            T_wc[:3, 3] = p_RS_R
            transforms[timestamp_ns] = T_wc
    
        min_frame_name = min(transforms.keys())
        min_frame_name = int(min_frame_name)
    if len(transforms) == 0:
        raise ValueError(f"No valid poses found in {gt_file}")
    return transforms, min_frame_name

def load_sequence_poses(sequence_folder:str,
                        verbose:bool=False):
    
    if 'sgslam' in sequence_folder:
        poses, min_frame_id = read_poses(osp(sequence_folder, 'pose'), 
                                            stride=1, 
                                            verbose=verbose)        
    elif 'euroc' in sequence_folder:
        poses, min_frame_t_ns = read_euro_gt_file(osp(sequence_folder, 'state_groundtruth_estimate0','data.csv'))        
        min_frame_id = min_frame_t_ns
    else:
        raise ValueError(f"Unsupported sequence folder: {sequence_folder}")
    if verbose:
        print(f"Loaded {len(poses)} poses from {sequence_folder}")

    return poses, min_frame_id

def read_euro_images(session_dir:Path,
                     verbose:bool=False):
    
    frame_list = [p.relative_to(session_dir).as_posix() for p in (session_dir/'cam0'/'data').iterdir()]
    frame_list = sorted(frame_list)
    timestamp = [float(os.path.basename(p).split('.')[0]) for p in frame_list] # in ns
    timestamp = [t/1e6 for t in timestamp] # convert to ms
    t_start = timestamp[0]
    timestamp = [t - t_start for t in timestamp] # make it start from 0
    
    if verbose:
        print('Load {} images from {}. t in ({:.1f},{:.1f})'.format(len(frame_list), 
                                                            session_dir,
                                                            timestamp[0],
                                                            timestamp[-1]))    
    return timestamp, frame_list

def load_euro_images(session_dir:Path,
                     frame_gap:int=1,
                    verbose:bool=False):
    """
    Load images from a Euroc session directory.
    
    Args:
        session_dir (Path): Path to the Euroc session directory.
        verbose (bool): If True, print loading information.
        
    Returns:
        timestamp (list): List of timestamps in ms.
        frame_list (list): List of image file paths relative to the session directory.
    """
    timestamp, frame_list = read_euro_images(session_dir, verbose)
    if frame_gap > 1:
        frame_list = frame_list[::frame_gap]
        timestamp = timestamp[::frame_gap]
        if verbose:
            print('Reduced frame list to every {}-th frame.'.format(frame_gap))
    
    rgb_images = {}
    
    for frame in frame_list:
        img = cv2.imread(osp(session_dir, frame))
        frame_name = os.path.basename(frame).split('.')[0]
        rgb_images[frame_name] = img
    if verbose:
        print('Loaded {} RGB images from {}'.format(len(rgb_images), session_dir))
    return rgb_images, timestamp


def align_euro_keyframes(poses:dict,
                         rgb_images:dict,
                         verbose:bool=False):
    '''
    Align RGB images with the provided poses.
    Args:
        poses (dict): Dictionary of poses {frame_name: T_wc}.
        rgb_images (dict): Dictionary of RGB images {frame_name: image}.
        Returns:
        rgb_poses (dict): Aligned poses for RGB images {frame_name: T_wc}.
    '''
    rgb_poses = {} # {frame_name: T_wc}
    for frame_name, _ in rgb_images.items():
        if frame_name in poses:
            rgb_poses[frame_name] = poses[frame_name]
        else:
            print(f"Warning: Frame {frame_name} not found in poses.")
    if len(rgb_poses) == 0:
        raise ValueError("No valid poses found for the provided RGB images.")
    if verbose:
        print(f"Aligned {len(rgb_poses)} RGB poses with images.")
    # Ensure the poses are in the same order as the images
    rgb_poses = {k: rgb_poses[k] for k in sorted(rgb_poses.keys())}

    return rgb_poses
    
