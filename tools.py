import os, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

def save_loop_transformation(out_dir:str, 
                             loop_pairs:list, 
                             loop_transformations:list, valid_only:bool):
    with open(out_dir,'w') as f:
        count = 0
        f.write('# src_frame ref_frame tx ty tz qx qy qz qw\n')
        for pair, T in zip(loop_pairs, loop_transformations):
            fail_pnp = np.allclose(T, np.eye(4))
            if fail_pnp and valid_only:
                continue
            
            translation = T[:3,3]
            quaternion = R.from_matrix(T[:3,:3]).as_quat()
            f.write('{} {} '.format(pair[0],pair[1]))
            f.write('{:.3f} {:.3f} {:.3f} '.format(translation[0],translation[1],translation[2]))
            f.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(quaternion[0],quaternion[1],quaternion[2],quaternion[3]))
            count+=1
        f.close()
        print('Save {}/{} loop transformations.'.format(count,len(loop_transformations)))


def read_all_poses(pose_folder):
    pose_files = glob.glob(os.path.join(pose_folder, '*.txt'))
    pose_map = {}
    for pose_f in sorted(pose_files):
        frame_name = os.path.basename(pose_f).split('.')[0]
        pose = np.loadtxt(pose_f)
        pose_map[frame_name] = pose
    
    return pose_map