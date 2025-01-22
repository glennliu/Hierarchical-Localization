import os, sys
import numpy as np
from pathlib import Path
from ri_looper import single_session_lcd
# from estimator import computeLoopTransformation
from write_pose_graphs import read_all_poses, read_pnp_folder


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
              ROT_THRESHOLD = 10,
              POS_THRESHOLD = 1.0):
    pred_ref_pose = src_pose @ np.linalg.inv(T_pred_ref_src)
    
    rre = relative_rotation_error(ref_pose[:3,:3], pred_ref_pose[:3,:3])
    rte = realtive_translation_error(ref_pose[:3,3], pred_ref_pose[:3,3])
    
    true_positive = (rre < ROT_THRESHOLD) & (rte < POS_THRESHOLD)
    true_positive = int(true_positive)
    
    return rre, rte, true_positive
    

if __name__=='__main__':
    
    DATAROOT = Path('/data2/sgslam/scans')
    SFM_DATAROOT = Path('/data2/sfm/single_session')  
    RUN = True  
    EVAL = False
    
    #     
    session = 'vins_reverse_loop'
    # session = 'vins_hard_loop'
    tp_array = []

    if RUN:
        single_session_lcd(DATAROOT/session,SFM_DATAROOT/session)

    if EVAL:
        output_result = '# src_frame, ref_frame, tp, rre(deg), rte(m)\n'
        print('Evaluate the session: {}'.format(session))
        frame_poses = read_all_poses(os.path.join(DATAROOT, session, 'pose'))
        pnp_predictions = read_pnp_folder(os.path.join(SFM_DATAROOT, session, 'pnp'))
        
        for pred in pnp_predictions:
            pose_src = frame_poses[pred['src_frame']]
            pose_ref = frame_poses[pred['ref_frame']]
            T_pnp = pred['pose']
            
            rre, rte, tp = eval_loop(pose_src, pose_ref, T_pnp)
            output_result += '{} {} '.format(pred['src_frame'], 
                                                pred['ref_frame'])
            output_result += '{} {:.3f} {:.3f}\n'.format(tp, rre, rte)
            tp_array.append(tp)
        
        #
        tp_array = np.array(tp_array)
        print('TP {}/{}'.format(tp_array.sum(), len(tp_array)))
        
        with open(os.path.join(SFM_DATAROOT, session, 'loop_result.txt'), 'w') as f:
            f.write(output_result)