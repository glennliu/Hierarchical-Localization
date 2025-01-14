import os, glob 
import numpy as np
from scipy.spatial.transform import Rotation as R
from ri_looper import read_loop_pairs, read_loop_transformations, save_loop_transformation
from ri_looper import read_frame_list
from timing import SequenceTimingRecord
from bandwidth import BandwidthSummary

def read_all_poses(pose_folder):
    pose_files = glob.glob(os.path.join(pose_folder, '*.txt'))
    pose_map = {}
    for pose_f in sorted(pose_files):
        frame_name = os.path.basename(pose_f).split('.')[0]
        pose = np.loadtxt(pose_f)
        pose_map[frame_name] = pose
    
    return pose_map

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

def read_pnp_folder(dir):
    files = glob.glob(os.path.join(dir,'*.txt'))
    files = sorted(files)
    pnp_predictions = []
    
    for f in files:
        pairs, transformations = read_loop_transformations(f)
        if len(pairs)>0:
            src_frame = pairs[0][0]
            ref_frame = pairs[0][1]
            pnp_predictions.append({'src_frame':src_frame, 
                                    'ref_frame':ref_frame, 
                                    'pose':transformations[0]})
        
    return pnp_predictions

def write_scene_poses(frame_poses:dict, outfile_dir:str):
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

def compute_gt_pnp(pnp_predictions:list,
                   src_poses:list,
                   ref_poses:list,
                   gt_pose:np.array):
    ''' Compute the ground-truth transformation between images T_c1_c0.
        Return: list of dict {'src_frame':str, 'ref_frame':str, 'pose':np.array}
    '''
    
    pnp_gt_lists = []
    gt_pnp_map = {}
    T_ref_src = gt_pose # (4,4)
    
    for pred in pnp_predictions:
        src_frame = pred['src_frame']
        ref_frame = pred['ref_frame']
        T_ref_c1 = ref_poses[ref_frame]
        T_src_c0 = src_poses[src_frame]
        T_gt_ref_c0 = T_ref_src @ T_src_c0
        T_gt_c1_c0 = np.linalg.inv(T_ref_c1) @ T_gt_ref_c0
        pnp_gt_lists.append({'src_frame':src_frame, 
                             'ref_frame':ref_frame, 
                             'pose':T_gt_c1_c0})
        gt_pnp_map['{}-{}'.format(src_frame,ref_frame)] = T_gt_c1_c0
    
    return gt_pnp_map
 
def compute_relative_predictions(pnp_predictions:list,
                          src_frame_poses:list,
                          ref_frame_poses:list):
    ''' Compute the registration between two agents T_ref_src
        return: list of dict {'src_frame':str, 'ref_frame':str, 'pose':np.array}
    '''    
    predictions = []
    for i, pnp_dict in enumerate(pnp_predictions):
        T_c1_c0 = pnp_dict['pose']
        src_frame = pnp_dict['src_frame']
        ref_frame = pnp_dict['ref_frame']
        assert src_frame in src_frame_poses, 'src {} not found'.format(src_frame)
        assert ref_frame in ref_frame_poses, 'ref {} not found'.format(ref_frame)
        
        T_src_c0 = src_frame_poses[src_frame]
        T_ref_c1 = ref_frame_poses[ref_frame]
        T_ref_c0 = T_ref_c1 @ T_c1_c0
        T_ref_src = T_ref_c0 @ np.linalg.inv(T_src_c0)
        predictions.append({'src_frame':src_frame, 'ref_frame':ref_frame, 'pose':T_ref_src})
        
    return predictions
    
    if multi_session:
        import shutil
        shutil.copyfile(os.path.join(sfm_scene,'loop_transformations.txt'), 
                        os.path.join(pgo_folder,'loop_transformations.txt'))

def write_frames_transfromation(transformation_list, output_file):
    with open(output_file,'w') as f:
        f.write('# src_frame ref_frame tx ty tz qx qy qz qw\n')
        for pred in transformation_list:
            T_ref_src = pred['pose']
            src_frame = pred['src_frame']
            ref_frame = pred['ref_frame']
            quat = R.from_matrix(T_ref_src[:3,:3]).as_quat()
            t = T_ref_src[:3,3]
            
            f.write('{} {} '.format(src_frame, ref_frame))
            f.write('{:.3f} {:.3f} {:.3f} '.format(t[0],t[1],t[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(quat[0],quat[1],quat[2],quat[3]))
        f.close()
        print('Write {} transformation to {}'.format(len(transformation_list), output_file))
    
def process_image_matching(pnp_predictions:list,
                        src_scene_dir:str,
                        ref_scene_dir:str,
                        src_feature_dir:str,
                        ref_feature_dir:str,
                        match_file_dir:str,
                        T_gt_c1_c0:dict,
                        output_folder:str):
    from hloc.utils.io import list_h5_names, read_image, get_matches, get_keypoints
    from hloc.utils.viz import plot_images, plot_keypoints, add_text, plot_matches, save_plot
    from third_party.SuperGluePretrainedNetwork.models.utils import compute_epipolar_error
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder)
    K0 = np.loadtxt(os.path.join(src_scene_dir,
                                 'intrinsic', 
                                 'intrinsic_depth.txt'))
    K1 = np.loadtxt(os.path.join(ref_scene_dir,
                                'intrinsic', 
                                'intrinsic_depth.txt'))
    EPIPOLOR_THED = 1e-5 # 5e-4
    HARDCODE_FRAME = None
    HARDCODE_FRAME = 'frame-000949'
    
    for pred in pnp_predictions:
        src_frame = pred['src_frame']
        ref_frame = pred['ref_frame']
        if HARDCODE_FRAME is not None and src_frame!=HARDCODE_FRAME:
            continue
        
        print('--- {}-{} ---'.format(src_frame, ref_frame))
        
        rgb0 = read_image(os.path.join(src_scene_dir, 'rgb', '{}.png'.format(src_frame)))
        rgb1 = read_image(os.path.join(ref_scene_dir, 'rgb', '{}.png'.format(ref_frame)))
        kpts0 = get_keypoints(src_feature_dir, 
                              'rgb/{}.png'.format(src_frame))
        kpts1 = get_keypoints(ref_feature_dir, 
                              'rgb/{}.png'.format(ref_frame))
        
        matches, scores = get_matches(match_file_dir, 
                                      'rgb-{}.png'.format(src_frame), 
                                      'rgb-{}.png'.format(ref_frame))
        if matches.shape[0]==0:
            print('Skip invalid matches')
            continue
        
        # compute epipolar errors
        gt_pose = T_gt_c1_c0['{}-{}'.format(src_frame,ref_frame)]
        epi_errs = compute_epipolar_error(kpts0[matches[:, 0], :2],
                                      kpts1[matches[:, 1], :2],
                                      gt_pose,
                                      K0, K1)
        correct = epi_errs < EPIPOLOR_THED
        print('{}/{} correct'.format(np.sum(correct), len(correct)))
        
        # color correct match in green, incorrect in red
        colors = [[0,1.0,0.0,1.0] if c else [1.0,0.0,0.0,1.0] for c in correct]

        plot_images([rgb0, rgb1], 
                    [src_frame,ref_frame])
        plot_matches(kpts0[matches[:, 0], :2], 
                    kpts1[matches[:, 1], :2], 
                    color=colors,
                    lw=1.5, a=0.5)
        add_text(0, '{}/{} correct'.format(np.sum(correct), len(correct)))        
        save_plot(os.path.join(output_folder,'{}_{}.pdf'.format(src_frame,ref_frame)))

if __name__=='__main__':
    ########### Set Parameters ################
    dataroot = '/data2/sgslam'
    sfm_dataroot = '/data2/sfm'
    OUTPUT_FOLDER = os.path.join(dataroot, 'output', 'hloc')
    WRITE_PG_FOLDER = False
    EVAL_MATCHES = True
    SUMMARY_MS_TIMING = False
    SUMMARY_MA_TIMING = False
    SUMMARY_MA_BANDWIDTH = False
    ###########################################

    scene_pairs =[
                # ['uc0110_00a','uc0110_00b'],
                # ['uc0110_00a','uc0110_00c'],
                # ['uc0115_00a','uc0115_00b'],
                # ['uc0115_00a','uc0115_00c'],
                # ['uc0204_00a','uc0204_00c'],
                ['uc0204_00a','uc0204_00b'],
                # ['uc0111_00a','uc0111_00b'],
                # ['ab0201_03c','ab0201_03a'],
                # ['ab0302_00a','ab0302_00b'],
                # ['ab0403_00c','ab0403_00d'],
                # ['ab0401_00a','ab0401_00b']
                ]
    ms_frame_number = 0
    ma_frame_numer = 0
    summary_time_array = []
    summary_ma_time_array = SequenceTimingRecord()
    summary_bandwidth = BandwidthSummary()
    
    for pair in scene_pairs:
        print('************* {}-{} *************'.format(pair[0],pair[1]))
        
        pair_folder = os.path.join(sfm_dataroot, 'multi_agent','{}-{}'.format(pair[0],pair[1]))
        pair_output_folder = os.path.join(OUTPUT_FOLDER,'{}-{}'.format(pair[0],pair[1]))
        gt_pose = np.loadtxt(os.path.join(dataroot,'gt','{}-{}.txt'.format(pair[0],pair[1])))
        if os.path.exists(pair_output_folder)==False:
            os.makedirs(pair_output_folder)
        
        pnp_predictions = read_pnp_folder(os.path.join(pair_folder,'pnp'))
        src_frame_poses = read_all_poses(os.path.join(dataroot,'scans',pair[0],'pose'))
        ref_frame_poses = read_all_poses(os.path.join(dataroot,'scans',pair[1],'pose'))
                    
        if WRITE_PG_FOLDER:
            pgo_folder = os.path.join(pair_folder,'pose_graph')
            if os.path.exists(pgo_folder)==False:
                os.makedirs(pgo_folder)
            
            predictions = compute_relative_predictions(pnp_predictions, 
                                                src_frame_poses,
                                                ref_frame_poses) # T_ref_src
            
            write_scene_poses(src_frame_poses,
                            os.path.join(pgo_folder,'src_poses.txt'))
            write_scene_poses(ref_frame_poses,
                            os.path.join(pgo_folder,'ref_poses.txt'))
            write_frames_transfromation(predictions, 
                            os.path.join(pair_output_folder,'predictions.txt'))
        
        if EVAL_MATCHES:
            import h5py
            MATCH_FILE = 'matches-superpoint-lightglue.h5'
            FEATURE_FILE = 'superpoint.h5'
            
            gt_pnp_map = compute_gt_pnp(pnp_predictions,
                            src_frame_poses,
                            ref_frame_poses,
                            gt_pose)
            
            process_image_matching(pnp_predictions,
                                   os.path.join(dataroot,'scans',pair[0]),
                                   os.path.join(dataroot,'scans',pair[1]),
                                   os.path.join(pair_folder,'{}_{}'.format(pair[0],FEATURE_FILE)),
                                   os.path.join(pair_folder,'{}_{}'.format(pair[1],FEATURE_FILE)),
                                   os.path.join(pair_folder,MATCH_FILE),
                                   gt_pnp_map,
                                   os.path.join(pair_folder,'eval'))
            # write_frames_transfromation(pnp_gt_lists,
            #                             os.path.join(pair_folder,'pnp_gt.txt'))
            
            
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