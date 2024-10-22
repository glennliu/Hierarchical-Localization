import os, glob 
import numpy as np
from scipy.spatial.transform import Rotation as R
from ri_looper import read_loop_pairs, read_loop_transformations, save_loop_transformation
from ri_looper import read_frame_list
from timing import SequenceTimingRecord
from bandwidth import BandwidthSummary

def read_scene_pose(scene_path, frame_list):
    pose_folder = os.path.join(scene_path,'pose')
    poses = {}
    for frame in frame_list:
        pose_file = os.path.join(pose_folder,frame+'.txt')
        if os.path.exists(pose_file):
            T_wc = np.loadtxt(pose_file)
            poses[frame] = T_wc
        else:
            print('Pose file {} not found'.format(pose_file))
        # poses.append(T_wc)
    print('Find {}/{} poses'.format(len(poses),len(frame_list)))
    
    return poses

def read_feature_timing(dir):
    with open(dir,'r') as f:
        lines = f.readlines()
        netvlad = float(lines[1].split(':')[1].strip())
        superpoint = float(lines[2].split(':')[1].strip())
        
        f.close()
        return netvlad, superpoint

def read_timing_record(timing_file:str):
    with open(timing_file,'r') as f:
        lines = f.readlines()
        header = lines[0].split()[1:]
        timing = {}
        time_array = []
        frame_number = 0
        for line in lines[1:]:
            key, val = line.split(':')
            if 'LMatch' in key:
                frame_number = int(val.split(',')[0])
            time_array.append(float(val.split(',')[1]))
            
        f.close()
        return frame_number, np.array(time_array)

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

def write_scene_loop_poses(dataroot, sfm_dataroot, src_scan, ref_scan, multi_session=False):
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
    
    if multi_session:
        import shutil
        shutil.copyfile(os.path.join(sfm_scene,'loop_transformations.txt'), 
                        os.path.join(pgo_folder,'loop_transformations.txt'))


if __name__=='__main__':
    ########### Set Parameters ################
    dataroot = '/data2/sgslam'
    sfm_dataroot = '/data2/sfm'
    WRITE_PG_FOLDER = False
    SUMMARY_MS_TIMING = False
    SUMMARY_MA_TIMING = True
    SUMMARY_MA_BANDWIDTH = True
    ###########################################

    
    scene_pairs =[
                # ['uc0110_00a','uc0110_00b'],
                ['uc0110_00a','uc0110_00c'],
                ['uc0115_00a','uc0115_00b'],
                ['uc0115_00a','uc0115_00c'],
                ['uc0204_00a','uc0204_00c'],
                ['uc0204_00a','uc0204_00b'],
                ['uc0111_00a','uc0111_00b'],
                ['ab0201_03c','ab0201_03a'],
                ['ab0302_00a','ab0302_00b'],
                ['ab0403_00c','ab0403_00d'],
                ['ab0401_00a','ab0401_00b']
                ]
    ms_frame_number = 0
    ma_frame_numer = 0
    summary_time_array = []
    summary_ma_time_array = SequenceTimingRecord()
    summary_bandwidth = BandwidthSummary()
    
    for pair in scene_pairs:
        pair_folder = os.path.join(sfm_dataroot, 'multi_agent','{}-{}'.format(pair[0],pair[1]))
        
        if WRITE_PG_FOLDER:
            write_scene_loop_poses(dataroot, os.path.join(sfm_dataroot,'multi_agent'), pair[0], pair[1])
    
        if SUMMARY_MS_TIMING:    
            frame_number, scene_time_array = read_timing_record(
                                            os.path.join(sfm_dataroot,'{}-{}'.format(pair[0],pair[1]),'timing.txt'))
            # print(frame_number, scene_time_array)
            ms_frame_number += frame_number
            summary_time_array.append(scene_time_array)
        
        if SUMMARY_MA_TIMING:
            scene_timing = np.loadtxt(os.path.join(sfm_dataroot,'multi_agent','{}-{}'.format(pair[0],pair[1]),
                                    'match_timing.txt'))    
            pnp_timing = np.loadtxt(os.path.join(sfm_dataroot,'multi_agent','{}-{}'.format(pair[0],pair[1]),
                                    'pnp_timing.txt'))
            netvlad, superpoint = read_feature_timing(os.path.join(sfm_dataroot,'multi_agent','{}-{}'.format(pair[0],pair[1]),
                                    'features_timing.txt'))
                        
            summary_ma_time_array.netvlad += netvlad
            summary_ma_time_array.superpoint += superpoint
            summary_ma_time_array.db_frames.append(scene_timing[:,0])
            summary_ma_time_array.gmatch_times.append(scene_timing[:,2])
            summary_ma_time_array.lightglue_times.append(scene_timing[:,3])
            summary_ma_time_array.pnp_times.append(pnp_timing)
            ma_frame_numer += scene_timing.shape[0]
            num_images = scene_timing.shape[0]
        else:
            num_images = 0

        if SUMMARY_MA_BANDWIDTH:
            import h5py
            from hloc.utils.io import list_h5_names
            summary_bandwidth.image_number += num_images
            
            superpoint_feature_dir= os.path.join(pair_folder,'{}_superpoint.h5'.format(pair[0]))
            image_names = list_h5_names(superpoint_feature_dir)

            with h5py.File(str(superpoint_feature_dir), "r", libver="latest") as hfile:
                for image_name in image_names:
                    dset = hfile[image_name]
                    kpts = dset['keypoints'].__array__()
                    # descriptors = dset['descriptors'].__array__()
                    # scores = dset['scores'].__array__()         
                    num_features = kpts.shape[0]
                    summary_bandwidth.sp_number.append(num_features) # Sum all of the sp features
            
            # print(data_names)
            
    
    if ms_frame_number>0:
        summary_time_array = np.array(summary_time_array)
        mean_time_array = np.sum(summary_time_array,axis=0)/ms_frame_number
        print('Total frames:',ms_frame_number)
        print('Header: netvlad, gmatch, superpoint, lmatch, RANSAC+PNP')
        print('Avg(ms): {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(mean_time_array[0]*1000,
                                                                    mean_time_array[1]*1000,
                                                                    mean_time_array[2]*1000,
                                                                    mean_time_array[3]*1000,
                                                                    mean_time_array[4]*1000))
        print('Total query time(ms): {:.3f}'.format( mean_time_array.sum()*1000))

    if ma_frame_numer>0:
        summary_ma_time_array.analysis()
        summary_ma_time_array.export(os.path.join(sfm_dataroot,'multi_agent','timing.npy'))
    
    if SUMMARY_MA_BANDWIDTH:
        summary_bandwidth.analysis()