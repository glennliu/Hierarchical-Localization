import os, sys
from pathlib import Path
import numpy as np
import cv2 as cv
import time
from third_party.SuperGluePretrainedNetwork.models.utils import compute_epipolar_error
from hloc.utils import viz, io
from hloc import extract_features, match_features

def kpt2np(kpts):
    return np.array([k.pt for k in kpts])

def read_gt_pose(src_frame_pose:str, 
                 ref_frame_pose: str,
                 gt_ref_src_pose: str):
    
    T_gt_ref_src = np.loadtxt(gt_ref_src_pose)
    T_ref_c1 = np.loadtxt(ref_frame_pose)
    T_src_c0 = np.loadtxt(src_frame_pose)
    
    T_ref_c0 = T_gt_ref_src @ T_src_c0
    T_c1_c0 = np.linalg.inv(T_ref_c1) @ T_ref_c0
    
    return T_c1_c0

def write_pairs(pair_file_dir:str,
                src_frame:str, 
                ref_frame:str):

    # write the pair frame to the file. If the file exists, append to it.
    mode = 'a' if os.path.exists(pair_file_dir) else 'w'
    with open(pair_file_dir, mode) as f:
        f.write('{} {}\n'.format(src_frame, ref_frame))
        f.close()


''' Brute-force match two images using ORB features'''
if __name__=='__main__':
    ############ Configuration ############
    DATAROOT = '/data2/mower_data'
    SRC_SCENE = 'random'
    REF_SCENE = 'random'
    src_frame = 'IMG_20250516_141040.jpg'
    ref_frame = 'IMG_20250516_141040.jpg' 
    
    FEAT_TYPE = 'superpoint'
    # OUTPUT_FOLDER = os.path.join(DATAROOT, 'output','hydra_lcd',
    #                              '{}-{}/'.format(SRC_SCENE, REF_SCENE),
    #                              'eval')       
    OUTPUT_FOLDER = os.path.join(DATAROOT, 'output')
    ######################################
    src_scene_dir = os.path.join(DATAROOT, SRC_SCENE)
    ref_scene_dir = os.path.join(DATAROOT, REF_SCENE)
    src_frame_dir = os.path.join(src_scene_dir, src_frame)
    ref_frame_dir = os.path.join(ref_scene_dir, ref_frame)
    src_frame_pose_dir = os.path.join(src_scene_dir, 'pose', src_frame[:-4]+'.txt')
    ref_frame_pose_dir = os.path.join(ref_scene_dir, 'pose', ref_frame[:-4]+'.txt')

    EPIPLOAR_THRESHOLD = 1e-3
    
    if os.path.exists(OUTPUT_FOLDER) is False:
        os.makedirs(OUTPUT_FOLDER)
    
    TOPK = 5000
    DISTANCE_THD = 80.0

    img1 = cv.imread(src_frame_dir, cv.IMREAD_COLOR)          # queryImage
    img2 = cv.imread(ref_frame_dir, cv.IMREAD_COLOR) # trainImage
    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        sys.exit(0)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    
    # Initiate ORB detector
    t0 = time.time()
    if FEAT_TYPE=='orb':
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kpts1, des1 = orb.detectAndCompute(img1,None)
        t_orb = time.time()-t0
        kpts2, des2 = orb.detectAndCompute(img2,None)
        print('descriptor shape:', des1.shape)
        
        # create BFMatcher object
        t0 = time.time()
        bf = cv.BFMatcher(cv.NORM_HAMMING, 
                        crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # matches = bf.knnMatch(des1,des2)
        t_match = time.time()-t0
        
        print('ORB time: {:.3f}ms, Matching time: {:.3f}ms'.format(t_orb*1000, t_match*1000))
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        scores = [m.distance for m in matches]
        matches = [m for m in matches if m.distance<DISTANCE_THD]
        
        if len(matches)<TOPK:
            viz_matches = matches
        else:   
            viz_matches = matches[:TOPK]
        print('Find {} matches'.format(len(viz_matches)))
        
    elif FEAT_TYPE=='superpoint':
        OVERWRITE = True
        MATCH_MIN_SCORE = 0.02
        sp_feat_conf = extract_features.confs['superpoint_inloc']
        matcher_conf = match_features.confs['superpoint+lightglue']
        # matcher_conf = match_features.confs['NN-superpoint']
        # matcher_conf["model"]["do_mutual_check"] = False
        # matcher_conf["model"]["distance_threshold"] = 0.1
        global_pair_dir = Path(src_scene_dir+'/global_pairs.txt')
        sp_feat_dir = Path(src_scene_dir+'/superpoint.h5')
        dense_match_dir = Path(src_scene_dir+'/matches_superpoint.h5')
        durations = []
        
        write_pairs(global_pair_dir, src_frame, ref_frame)
        extract_features.main(conf=sp_feat_conf,
                        image_dir=Path(src_scene_dir),
                        image_list=[Path(src_frame), Path(ref_frame)],
                        feature_path=sp_feat_dir,
                        overwrite=OVERWRITE,
                        duration_list=durations) # local features

        match_features.main(conf=matcher_conf, 
                    pairs = global_pair_dir, 
                    features=sp_feat_dir, 
                    matches=dense_match_dir, 
                    overwrite=OVERWRITE,
                    duration_list=durations)

        kpts1 = io.get_keypoints(path=sp_feat_dir, 
                                 name=src_frame)
        kpts2 = io.get_keypoints(path=sp_feat_dir, 
                                 name=ref_frame)
        matches, scores = io.get_matches(dense_match_dir, src_frame, ref_frame)
        # mask = scores>MATCH_MIN_SCORE
        # matches = matches[mask]
        # scores = scores[mask]
        if False:
            select_pt_index = 0
            select_mask = matches[:,0]==select_pt_index
            matches = matches[select_mask]
            scores = scores[select_mask]
            print('select pt {} has {} matches'.format(select_pt_index, len(matches)))
        
        tmp_indices = matches[:, 0]
        unique_indices = np.unique(tmp_indices)
        print('total {} matches, {} unique pts in src image'.format(len(matches), len(unique_indices)))
        
           
        
        viz_matches = matches

        print('kpts shape: ', kpts1.shape)
        print('matches shape:', matches.shape)

    # exit(0)
    # convert
    if not isinstance(kpts1, np.ndarray):
        kpts1 = kpt2np(kpts1) # (N1,2)
    if not isinstance(kpts2, np.ndarray):
        kpts2 = kpt2np(kpts2)  # (N2,2)
    if not isinstance(viz_matches, np.ndarray):
        viz_matches = np.array([(m.queryIdx, m.trainIdx) for m in viz_matches]) # (M,2)
    raw_pts1 = kpts1
    raw_pts2 = kpts2
    kpts1 = kpts1[viz_matches[:,0]]
    kpts2 = kpts2[viz_matches[:,1]]    
    
    # Compute epipolar geometry

    # Viz
    viz.plot_images([img1, img2], 
                    [src_frame, ref_frame])
    viz.plot_keypoints([raw_pts1, raw_pts2],
                       ps=3, colors=[[1.0,0.0,0.0,1.0],[0.0,1.0,0.0,1.0]])
    
    if os.path.exists(src_frame_pose_dir) and os.path.exists(ref_frame_pose_dir):
        T_gt_c1_c0 = read_gt_pose(src_frame_pose_dir, 
                                  ref_frame_pose_dir,
                                  os.path.join(DATAROOT,'gt','{}-{}.txt'.format(SRC_SCENE, REF_SCENE)))
        K0 = np.loadtxt(os.path.join(src_scene_dir,
                                    'intrinsic',
                                    'intrinsic_depth.txt'))
        K1 = np.loadtxt(os.path.join(ref_scene_dir,
                                    'intrinsic',
                                    'intrinsic_depth.txt'))
        epipolar_errs = compute_epipolar_error(kpts1,kpts2,T_gt_c1_c0,K0,K1)
        correct = epipolar_errs<EPIPLOAR_THRESHOLD
        colors = [[0,1.0,0.0,1.0] if c else [1.0,0.0,0.0,1.0] for c in correct]
        print('{}/{} correct'.format(np.sum(correct), len(correct)))
        
        viz.add_text(0,
                    '{}/{} correct'.format(np.sum(correct), len(correct)))
    else:
        colors = [[0,0.7,1.0,1.0] for _ in range(len(kpts1))]
    viz.plot_matches(kpts1,kpts2,colors,lw=1.5,a=0.5)
    # viz.save_plot(OUTPUT_FOLDER+'/{}-{}.pdf'.format(src_frame,ref_frame))
    viz.save_plot(OUTPUT_FOLDER+'/{}-{}.png'.format(src_frame,ref_frame))


