import os
import tqdm
from pathlib import Path
import numpy as np
import cv2

from hloc.visualization import read_image
from hloc.utils import viz, io


if __name__=='__main__':
    ####### SET ARGS #######
    DATAROOT = '/data2/sgslam/scans'
    SCENE_RESULT_FOLDER = Path('/data2/sfm/multi_agent/uc0151_02-uc0151_00')
    MATCHING_DIR = SCENE_RESULT_FOLDER / 'matches-superpoint-lightglue.h5'
    SRC_FRAME = 'frame-000550'
    REF_FRAME = 'frame-000490'
    DRAW_MATCHES = True
    ########################
    
    pair_name = os.path.basename(SCENE_RESULT_FOLDER)
    src_scene = pair_name.split('-')[0]
    ref_scene = pair_name.split('-')[1]
    src_frame_dir = os.path.join(DATAROOT, src_scene, 'rgb', SRC_FRAME + '.png')
    ref_frame_dir = os.path.join(DATAROOT, ref_scene, 'rgb', REF_FRAME + '.png')
    superpoint_src = SCENE_RESULT_FOLDER / '{}_superpoint.h5'.format(src_scene)
    superpoint_ref = SCENE_RESULT_FOLDER / '{}_superpoint.h5'.format(ref_scene)

    # load
    src_image = read_image(src_frame_dir)
    ref_image = read_image(ref_frame_dir)
    kpts_src = io.get_keypoints(superpoint_src, 
                                'rgb/{}.png'.format(SRC_FRAME))
    kpts_ref = io.get_keypoints(superpoint_ref,
                                'rgb/{}.png'.format(REF_FRAME))
    if DRAW_MATCHES:
        matches, scores = io.get_matches(MATCHING_DIR,
                                    'rgb/{}.png'.format(SRC_FRAME),
                                    'rgb/{}.png'.format(REF_FRAME))
        kpts0 = kpts_src[matches[:, 0]]
        kpts1 = kpts_ref[matches[:, 1]]
    else:
        kpts0 = kpts_src
        kpts1 = kpts_ref
    
    # plot
    fake_image = np.zeros_like(src_image)
    fake_image[:, :, 0] = 255
    fake_image[:, :, 1] = 255
    fake_image[:, :, 2] = 255
    viz.plot_images([src_image,ref_image], dpi=75)
    # viz.plot_keypoints([kpts_src],
    #                    colors='lime',
    #                    ps=1)
    if DRAW_MATCHES:
        inliner_text = '{} correspondences'.format(matches.shape[0])
        viz.plot_matches(kpts0=kpts0,
                        kpts1=kpts1,
                        lw=1,
                        )
    # inliner_text = '{}/{} correct'.format(0,
    #                                     matches.shape[0])
    
    if False:
        inliner_text = '{} features'.format(kpts_src.shape[0])    
        viz.add_text(0, inliner_text)
                   
    viz.save_plot('outputs/image_matching.png')
    src_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('outputs/{}.png'.format(SRC_FRAME), src_image)
    cv2.imwrite('outputs/{}.png'.format(REF_FRAME), ref_image)   

