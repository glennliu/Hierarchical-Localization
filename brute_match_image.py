import os, sys
import numpy as np
import cv2 as cv
import time
from third_party.SuperGluePretrainedNetwork.models.utils import compute_epipolar_error
from hloc.utils import viz

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

''' Brute-force match two images using ORB features'''
if __name__=='__main__':
    ############ Configuration ############
    DATAROOT = '/data2/sgslam'
    SRC_SCENE = 'uc0204_00a'
    REF_SCENE = 'uc0204_00b'
    src_frame = 'frame-000299'
    ref_frame = 'frame-000262'    
    ######################################
    OUTPUT_FOLDER = os.path.join(DATAROOT, 'output','hydra_lcd',
                                 '{}-{}/'.format(SRC_SCENE, REF_SCENE),
                                 'eval')
    EPIPLOAR_THRESHOLD = 1e-3
    
    if os.path.exists(OUTPUT_FOLDER) is False:
        os.makedirs(OUTPUT_FOLDER)
    
    src_scene_dir = os.path.join(DATAROOT, 'scans', SRC_SCENE)
    ref_scene_dir = os.path.join(DATAROOT, 'scans', REF_SCENE)
    TOPK = 5000
    DISTANCE_THD = 80.0

    img1 = cv.imread(os.path.join(src_scene_dir, 'rgb', src_frame+'.png'),
                     cv.IMREAD_COLOR)          # queryImage
    img2 = cv.imread(os.path.join(ref_scene_dir, 'rgb', ref_frame+'.png'),
                     cv.IMREAD_COLOR) # trainImage
    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        sys.exit(0)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    
    # Initiate ORB detector
    t0 = time.time()
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kpts1, des1 = orb.detectAndCompute(img1,None)
    t_orb = time.time()-t0
    kpt2, des2 = orb.detectAndCompute(img2,None)

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
    
    # convert
    kpts1 = kpt2np(kpts1) # (N1,2)
    kpts2 = kpt2np(kpt2) # (N2,2)
    viz_matches = np.array([(m.queryIdx, m.trainIdx) for m in viz_matches]) # (M,2)
    kpts1 = kpts1[viz_matches[:,0]]
    kpts2 = kpts2[viz_matches[:,1]]    
    
    # Compute epipolar geometry
    T_gt_c1_c0 = read_gt_pose(os.path.join(src_scene_dir, 'pose', src_frame+'.txt'),
                              os.path.join(ref_scene_dir, 'pose', ref_frame+'.txt'),
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
    
    # Viz
    viz.plot_images([img1, img2], 
                    [src_frame, ref_frame])
    viz.plot_matches(kpts1,kpts2,colors,lw=1.5,a=0.5)
    viz.add_text(0,
                 '{}/{} correct'.format(np.sum(correct), len(correct)))
    viz.save_plot(OUTPUT_FOLDER+'/{}-{}.pdf'.format(src_frame,ref_frame))
    viz.save_plot(OUTPUT_FOLDER+'/{}-{}.png'.format(src_frame,ref_frame))


