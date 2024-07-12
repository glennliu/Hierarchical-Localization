import os
import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path
import numpy as np

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils import viz


if __name__=='__main__':
    scan = 'sacre_coeur'
    images = Path('datasets/')/scan
    outputs = Path('outputs')/scan
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    
    if os.path.exists(outputs)==False:
        os.mkdir(outputs)

    feature_conf = extract_features.confs['disk']
    matcher_conf = match_features.confs['disk+lightglue']

    references = [p.relative_to(images).as_posix() for p in (images/'mapping').iterdir()]
    print(len(references), "mapping images")
    plot_images([read_image(images / r) for r in references], dpi=25)
    # viz.plot_matches(read_image(images / references[0]), read_image(images / references[1]), np.loadtxt(loc_pairs), color='lime')
    viz.save_plot(outputs / 'mapping.png')

    # Extract local features and match them
    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # sfm
    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
    fig.show()
    visualization.visualize_sfm_2d(model, images, color_by='visibility', n=2)
    
    # Export 
    viz.save_plot(sfm_dir / 'reconstruction.png')
    
    ## Localization
    print('------ relocalization test -----------')
    query = 'query/night.jpg'
    plot_images([read_image(images / query)], dpi=75)
    
    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)
    
    import pycolmap
    from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

    camera = pycolmap.infer_camera_from_image(images / query)
    ref_ids = [model.find_image_with_name(r).image_id for r in references]
    conf = {
        'estimation': {'ransac': {'max_error': 12}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
    visualization.visualize_loc_from_log(images, query, log, model) # visualize the matched pair with the most inliers
        
    # Export
    pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
    viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
    # visualize 2D-3D correspodences
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
    fig.show()    
    
    viz.save_plot(outputs / 'localization.png')