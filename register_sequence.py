import os, glob
from pathlib import Path
import numpy as np
import open3d as o3d
from estimator import HybridReg
from tools import save_loop_transformation, read_all_poses

def relative_rotation_error(gt_rotations: np.ndarray, 
                            rotations: np.ndarray):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    
    mat = np.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre

def realtive_translation_error(gt_translations: np.ndarray,
                               translations: np.ndarray):
    r"""Relative Translation Error.
    """
    rte = np.linalg.norm(gt_translations - translations, axis=-1)
    return rte

def eval_loop(src_pose,
              ref_pose,
              T_pred_ref_src,
              ROT_THRESHOLD = 5,
              POS_THRESHOLD = 0.5):
    
    pred_ref_pose = T_pred_ref_src @ src_pose
    
    rre = relative_rotation_error(ref_pose[:3,:3], pred_ref_pose[:3,:3])
    rte = realtive_translation_error(ref_pose[:3,3], pred_ref_pose[:3,3])
    
    true_positive = (rre < ROT_THRESHOLD) & (rte < POS_THRESHOLD)
    true_positive = int(true_positive)
    
    return rre, rte, true_positive

if __name__=='__main__':
    ############# SET PARAMS #############
    DATAROOT = Path('/data2/sgslam/scans')
    SFM_DATAROOT = Path('/data2/sfm/single_session')  
    SEQUENCE_NAME = 'vins_reverse_loop'
    NOISE_BOUND = 0.1
    REFINE = 'icp' # 'icp'
    EVAL= True
    #####################################
    
    frame_pair_folders = glob.glob(str(SFM_DATAROOT/SEQUENCE_NAME/'tmp/*'))
    frame_pair_folders = sorted(frame_pair_folders)
    output_folder = SFM_DATAROOT/SEQUENCE_NAME/'teaser'
    os.makedirs(output_folder, exist_ok=True)
    
    frame_poses = read_all_poses(os.path.join(DATAROOT, SEQUENCE_NAME, 'pose'))
    output_result = '# src_frame, ref_frame, tp, rre(deg), rte(m)\n'
    count_tp = 0
    count_pairs = 0

    for frame_pair_folder in frame_pair_folders:
        frame_pair = os.path.basename(frame_pair_folder)
        src_frame = frame_pair[:12]
        ref_frame = frame_pair[13:]
        print('-------------- {} --------------'.format(frame_pair))
        if not os.path.exists(frame_pair_folder+'/src.ply'):
            continue
        
        src_pcd = o3d.io.read_point_cloud(frame_pair_folder+'/src.ply')
        ref_pcd = o3d.io.read_point_cloud(frame_pair_folder+'/ref.ply')
        corr_A = np.load(frame_pair_folder+'/corr_A.npy')
        corr_B = np.load(frame_pair_folder+'/corr_B.npy')
        
        hybrid_reg = HybridReg(src_pcd, ref_pcd, refine=REFINE)
        tf = hybrid_reg.solve_by_teaser(corr_A, corr_B, noise_bound=NOISE_BOUND)
        
        frame_pairs = [[src_frame, ref_frame]]
        transformations = [tf]
        
        save_loop_transformation(output_folder/'{}.txt'.format(src_frame), 
                                 frame_pairs, 
                                 transformations,
                                 False)
        
        if EVAL:
            pose_src = frame_poses[src_frame]
            pose_ref = frame_poses[ref_frame]
            
            rre, rte, tp = eval_loop(pose_src, pose_ref, tf)
            output_result += '{} {} '.format(src_frame, ref_frame)
            output_result += '{} {:.3f} {:.3f}\n'.format(tp, rre, rte)
            
            count_tp += tp
            
        count_pairs += 1
        break
    #
    print('************ Finished ************')
    print('Recall {}/{}'.format(count_tp, count_pairs))
    with open(SFM_DATAROOT/SEQUENCE_NAME/'teaser_result.txt', 'w') as f:
        f.write(output_result)
    

