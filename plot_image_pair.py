import os
import tqdm
from pathlib import Path
import numpy as np

from hloc.visualization import read_image
from hloc.utils import viz, io


if __name__=='__main__':
    ####### SET ARGS #######
    DATAROOT = '/data2/sgslam/scans'
    SCENE_RESULT_FOLDER = Path('/data2/sfm/multi_agent/uc0204_00a-uc0204_00b')
    MATCHING_DIR = SCENE_RESULT_FOLDER / 'matches-superpoint-lightglue.h5'
    SRC_FRAME = 'frame-000449'
    REF_FRAME = 'frame-000250'
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
    matches, scores = io.get_matches(MATCHING_DIR,
                                'rgb/{}.png'.format(SRC_FRAME),
                                'rgb/{}.png'.format(REF_FRAME))
    print(scores.min())
    kpts0 = kpts_src[matches[:, 0]]
    kpts1 = kpts_ref[matches[:, 1]]
    
    # plot
    # inliner_text = '{}/{} correct'.format(0,
    #                                     matches.shape[0])
    inliner_text = '{} correspondences'.format(matches.shape[0])
    print(inliner_text)
    viz.plot_images([src_image, ref_image], dpi=75)
    # viz.plot_keypoints([kpts_src, kpts_ref])
    viz.plot_matches(kpts0=kpts0,
                     kpts1=kpts1,
                     lw=1,
                    #  color='red'
                     )
    # viz.add_text(0, inliner_text)
                   
    viz.save_plot('outputs/hloc_low_corrs.png')

