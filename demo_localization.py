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
    #
    images = Path('datasets/samples')
    output = Path('outputs/samples')
    ref_frame = Path('night.jpg')
    features = output / 'features.h5'
    matches = output / 'matches.h5'
    
    query = images/ref_frame

    # settings
    feature_conf = extract_features.confs['disk']
    matcher_conf = match_features.confs['disk+lightglue']

    plot_images([read_image(images / ref_frame)], dpi=75)
    
    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True);
    
    
    
    
    
    viz.save_plot(output / 'test.png')

    