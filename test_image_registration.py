import os, sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from hloc.visualization import read_image
from hloc.utils import viz, io

''' Brute-force match two images using ORB features'''
if __name__=='__main__':
    ############ Configuration ############
    DATAROOT = '/data2/sgslam'
    SRC_SCENE = 'ab0201_03c'
    REF_SCENE = 'ab0201_03a'
    src_frame = 'frame-000403.png'
    ref_frame = 'frame-000338.png'    
    ######################################
    
    src_folder = os.path.join(DATAROOT, 'scans', SRC_SCENE, 'rgb')
    ref_folder = os.path.join(DATAROOT, 'scans', REF_SCENE, 'rgb')
    TOPK = 5000
    DISTANCE_THD = 52.0

    img1 = cv.imread(os.path.join(src_folder, src_frame),
                     cv.IMREAD_COLOR)          # queryImage
    img2 = cv.imread(os.path.join(ref_folder, ref_frame),
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
    kp1, des1 = orb.detectAndCompute(img1,None)
    t_orb = time.time()-t0
    kp2, des2 = orb.detectAndCompute(img2,None)
    

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

    # concatenate images with a white margin between them
    white_margin = np.ones((img1.shape[0], 13, 3), np.uint8)*255
    img3 = np.hstack((img1, white_margin, img2))    
    
    # draw matches
    kpts0 = []
    kpts1 = []
    for m in viz_matches:
        # Get the matching keypoints for each of the images
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        kpts0.append([x1, y1])
        kpts1.append([x2, y2])
        
        random_color = np.random.randint(0,255,3).tolist()
        
        cv.circle(img3, (int(x1),int(y1)), 3, random_color, -1)
        cv.circle(img3, (int(x2)+img1.shape[1]+10,int(y2)), 3, random_color, -1)
        cv.line(img3, (int(x1),int(y1)), (int(x2)+img1.shape[1]+10,int(y2)), random_color, 1, cv.LINE_AA)
    img3 = cv.cvtColor(img3, cv.COLOR_RGB2BGR)
    # cv.imwrite('outputs/{}-{}'.format(src_frame.split('.')[0],ref_frame), img3)

    # HLoc visualization
    kpts0 = np.array(kpts0)
    kpts1 = np.array(kpts1)
    img0 = read_image(os.path.join(src_folder, src_frame))
    img1 = read_image(os.path.join(ref_folder, ref_frame))
    inlier_text = '{} correspondences'.format(len(viz_matches))
    viz.plot_images([img0, img1], dpi=75)
    viz.plot_matches(kpts0=kpts0, 
                     kpts1=kpts1)
    viz.save_plot('outputs/hydra_low_corrs.png')
    

    # img3 = cv.drawMatches(img1,kp1,img2,kp2,viz_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imsave('images/{}-{}.png'.format(src_frame,ref_frame), img3)


