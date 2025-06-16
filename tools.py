import os, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

def save_loop_pairs(out_dir:str, 
                    loop_pairs:list):
    with open(out_dir,'w') as f:
        f.write('# src_frame ref_frame\n')
        for pair in loop_pairs:
            f.write('{} {}\n'.format(pair[0],pair[1]))
        f.close()

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

def save_loop_true_masks(out_dir:str,
                         loop_eval_list:list):
    with open(out_dir,'w') as f:
        f.write('# src_frame ref_frame true_positive R_err(deg) t_err(m) \n')
        for loop_eval_dict in loop_eval_list:
            f.write('{} {} '.format(loop_eval_dict['src_frame'],
                                    loop_eval_dict['ref_frame']))
            f.write('{} {:.3f} {:.3f}\n'.format(loop_eval_dict['true_positive'],
                                        loop_eval_dict['R_err'],
                                        loop_eval_dict['t_err']))
        f.close()
        print('Save {} loop with evaluation masks to {}'.format(len(loop_eval_list),
                                                                out_dir))

def load_loop_true_masks(f_dir:str):
    loop_eval_list = []
    count = 0
    with open(f_dir,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split(' ')
            loop_eval_dict = {}
            loop_eval_dict['src_frame'] = items[0]
            loop_eval_dict['ref_frame'] = items[1]
            loop_eval_dict['true_positive'] = int(items[2])
            if loop_eval_dict['true_positive']>0:
                count += 1
            loop_eval_dict['R_err'] = float(items[3])
            loop_eval_dict['t_err'] = float(items[4])
            loop_eval_list.append(loop_eval_dict)
        f.close()
    
        print('{}/{} true positive loop pairs'.format(count,
                                                      len(loop_eval_list)))
    return loop_eval_list
 
def read_all_poses(pose_folder):
    pose_files = glob.glob(os.path.join(pose_folder, '*.txt'))
    pose_map = {}
    for pose_f in sorted(pose_files):
        frame_name = os.path.basename(pose_f).split('.')[0]
        pose = np.loadtxt(pose_f)
        pose_map[frame_name] = pose
    
    return pose_map

def read_loop_transformations(in_dir:str):
    loop_pairs = []
    loop_transformations = []
    with open(in_dir,'r') as f:
        for line in f.readlines():
            if '#' in line: continue
            elements = line.strip().split()
            src_frame = elements[0]
            ref_frame = elements[1]
            tvec = np.array([float(x) for x in elements[2:5]])
            quat = np.array([float(x) for x in elements[5:9]])
            T_ref_src = np.eye(4)
            T_ref_src[:3,:3] = R.from_quat(quat).as_matrix()
            T_ref_src[:3,3] = tvec
            
            loop_pairs.append([src_frame, ref_frame])
            loop_transformations.append(T_ref_src)
            # loop_transformations.append({'src_frame':src_frame, 
            #                             'ref_frame':ref_frame, 
            #                             'T_ref_src':T_ref_src})
        f.close()
        return loop_pairs, loop_transformations
    
def read_pnp_folder(dir:str, 
                    verbose=False):
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
    
    if verbose:
        print('Load {} pnp constraints from {}'.format(len(pnp_predictions), dir))
    return pnp_predictions

def read_loop_pair_file(file_dir:str):
    """
    Read loop pairs from a file.
    File format:
        # src_frame ref_frame
        query_f match_f0
        query_f match_f1
        ...
    """
    loop_pairs = []
    with open(file_dir, 'r') as f:
        for line in f.readlines():
            if '#' in line: continue
            elements = line.strip().split()
            if len(elements) != 2:
                continue
            src_frame = elements[0]
            ref_frame = elements[1]
            loop_pairs.append([src_frame, ref_frame])
        f.close()
    
    return loop_pairs

def read_loop_pairs(folder_dir:str,
                    verbose:bool=False):
    loop_pairs = {}
    loop_pair_files = glob.glob(os.path.join(folder_dir, '*.txt'))
    loop_pair_files = sorted(loop_pair_files)
    for loop_pair_file in loop_pair_files:
        pairs = read_loop_pair_file(loop_pair_file)
        # loop_pairs.extend(pairs)
        query_frame = pairs[0][0]
        loop_info = {'ref_frame': pairs[0][1]}
        loop_pairs[query_frame] = loop_info
    if verbose:
        print('Read {} loop pairs from {}'.format(len(loop_pairs), folder_dir))
    return loop_pairs
