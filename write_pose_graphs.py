import os, glob 
import numpy as np
from scipy.spatial.transform import Rotation as R
from ri_looper import read_loop_pairs, read_loop_transformations, save_loop_transformation
from ri_looper import read_frame_list

def read_scene_pose(scene_path, frame_list):
    pose_folder = os.path.join(scene_path,'pose')
    poses = {}
    for frame in frame_list:
        pose_file = os.path.join(pose_folder,frame+'.txt')
        T_wc = np.loadtxt(pose_file)
        poses[frame] = T_wc
        # poses.append(T_wc)
        
    return poses

def write_scene_poses(outfile_dir:str, frame_poses:dict):
    with open(outfile_dir,'w') as f:
        f.write('# frame x y z qx qy qz qw\n')
        for frame in frame_poses:
            T_wc = frame_poses[frame]
            R_wc = R.from_matrix(T_wc[:3,:3])
            t_wc = T_wc[:3,3]
            q_wc = R_wc.as_quat()
            f.write('{} {:.3f} {:.3f} {:.3f} '.format(frame,t_wc[0],t_wc[1],t_wc[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(q_wc[0],q_wc[1],q_wc[2],q_wc[3]))
        f.close()

# def extract_src_ref_frames(loop_pairs:list):
#     src_frames = []
#     ref_frames = []
    
#     for pair in loop_pairs:
#         if pair[0] not in src_frames:
#             src_frames.append(pair[0])
#         if pair[1] not in ref_frames:
#             ref_frames.append(pair[1])
#     return src_frames, ref_frames

def write_loop_poses(outfile_dir:str, src_poses, ref_poses):
    assert(len(src_poses)==len(ref_poses))
    with open(outfile_dir,'w') as f:
        f.write('# src_x, src_y, src_z, src_qx, src_qy, src_qz, src_qw, ref_x, ref_y, ref_z, ref_qx, ref_qy, ref_qz, ref_qw\n')
        for i in range(len(src_poses)):
            T_src = src_poses[i]
            T_ref = ref_poses[i]
            R_src = R.from_matrix(T_src[:3,:3])
            R_ref = R.from_matrix(T_ref[:3,:3])
            t_src = T_src[:3,3]
            t_ref = T_ref[:3,3]
            q_src = R_src.as_quat()
            q_ref = R_ref.as_quat()
            f.write('{:.3f} {:.3f} {:.3f} '.format(t_src[0],t_src[1],t_src[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f} '.format(q_src[0],q_src[1],q_src[2],q_src[3]))
            f.write('{:.3f} {:.3f} {:.3f} '.format(t_ref[0],t_ref[1],t_ref[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(q_ref[0],q_ref[1],q_ref[2],q_ref[3]))
            
            # f.write(' '.join(
            #     [str(x) for x in t_src])+' '+' '.join([str(x) for x in q_src])+' '+' '.join([str(x) for x in t_ref])+' '+' '.join([str(x) for x in q_ref])+'\n')
        f.close()

def write_scene_loop_poses(dataroot, sfm_dataroot, src_scan, ref_scan):
    sfm_scene = os.path.join(sfm_dataroot,'{}-{}'.format(src_scan,ref_scan))
    pgo_folder = os.path.join(sfm_scene,'pose_graph')
    if os.path.exists(pgo_folder)==False:
        os.makedirs(pgo_folder)

    # loop_pairs, loop_transformations = read_loop_transformations(os.path.join(sfm_scene,'loop_pairs.txt'))
    # loop_pairs = read_loop_pairs(os.path.join(sfm_dataroot,'{}-{}'.format(src_scan,ref_scan),'loop_pairs.txt'))
    src_frames = read_frame_list(os.path.join(sfm_scene,'src_frames.txt'))
    ref_frames = read_frame_list(os.path.join(sfm_scene,'ref_frames.txt'))
    
    # src_frames, ref_frames = extract_src_ref_frames(loop_pairs)
    # M = len(loop_transformations)
    print('{} src frames, {} ref frames'.format(len(src_frames), len(ref_frames)))

    src_frame_poses = read_scene_pose(os.path.join(dataroot,'scans',src_scan), 
                                src_frames)
    ref_frame_poses = read_scene_pose(os.path.join(dataroot,'scans',ref_scan),
                                ref_frames)
                                    
    write_scene_poses(os.path.join(pgo_folder,'src_poses.txt'), src_frame_poses)
    write_scene_poses(os.path.join(pgo_folder,'ref_poses.txt'), ref_frame_poses)
    
    # assert(len(src_poses)==len(ref_poses))
    # write_loop_poses(os.path.join(sfm_dataroot,'{}-{}'.format(src_scan,ref_scan),'loop_poses.txt'), 
    #                  src_poses, ref_poses)

if __name__=='__main__':
    dataroot = '/data2/sgslam'
    sfm_dataroot = '/data2/sfm'
    src_scan = 'uc0204_00a'
    ref_scan = 'uc0204_00b'
    
    write_scene_loop_poses(dataroot, sfm_dataroot, src_scan, ref_scan)
    