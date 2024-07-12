import os, glob 
import time
from pathlib import Path
from hloc import extract_features, match_features, localize_sfm
from hloc import visualization, colmap_from_nvm
import hloc.pairs_from_retrieval as pairs_from_retrieval
from hloc.utils import read_write_model, viz


if __name__=='__main__':
    # Setup the paths
    dataset = Path('datasets/aachen')
    outputs = Path('outputs/aachen')
    images = dataset / 'images_upright/'
    sift_sfm = dataset / 'sfm_sift'  # from which we extract the reference poses

    reference_sfm = outputs / 'sfm_superpoint+superglue'  # the built SfM model
    loc_pairs = outputs / 'pairs-query-netvlad50.txt'  # top-k retrieved by NetVLAD
    results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad50.txt'

    # sfm with pose priors
    colmap_from_nvm.main(
        dataset / '3D-models/aachen_cvpr2018_db.nvm',
        dataset / '3D-models/database_intrinsics.txt',
        dataset / 'aachen.db',
        sift_sfm)

    exit(0)
    # 
    feature_conf = extract_features.confs['superpoint_aachen']
    retrieval_conf = extract_features.confs['netvlad']
    matcher_conf = match_features.confs['superglue']
    features = extract_features.main(feature_conf, images, outputs)


    t0 = time.time()
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    print('load previous descriptors from ', global_descriptors)
    t1 = time.time()
    pairs_from_retrieval.main(global_descriptors, loc_pairs, 50,
                            query_prefix='query', db_model=reference_sfm) # global match 
    
    t2 = time.time()
    print('extract_features time:{:.3f} sec'.format(t1-t0))
    print('pairs_from_retrieval time:{:.3f} sec'.format(t2-t1))
    
    exit(0)
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf['output'], outputs) # local match
    t3 = time.time()
    print('flag')
    
    exit(0)
    # visualize the local matches 
    print('visualize some queries')
    images = read_write_model.read_images_binary(reference_sfm / 'images.bin')
    db_names = [i.name for i in images.values()]
    for db_name in db_names:
        print(db_name)

    exit(0)    
    queries = ['query/day/milestone/2010-10-30_17-47-33_586.jpg',
             'query/day/nexus4/IMG_20130210_163239.jpg',
             'query/day/nexus5x/IMG_20161227_160238.jpg']
    for query in queries:
        visualization.visualize_loc(results=results,image_dir=images,reconstruction=reference_sfm,selected=[query])
        out_name = os.path.basename(query).split('.')[0]+'.png'
        viz.save_plot(outputs /'matches'/ out_name)
    
    # localize_sfm.main(
    #     reference_sfm,
    #     dataset / 'queries/*_time_queries_with_intrinsics.txt',
    #     loc_pairs,
    #     features,
    #     loc_matches,
    #     results,
    #     covisibility_clustering=False)  # not required with SuperPoint+SuperGlue

    
    