import os, glob 
import time
import numpy as np
from pathlib import Path
from hloc import extract_features, match_features, localize_sfm
from hloc import visualization
import hloc.pairs_from_retrieval as pairs_from_retrieval
from hloc.utils import read_write_model, viz
from hloc.utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array

def get_image_idxs(db_dir):
    from scipy.spatial.transform import Rotation as R

    db = COLMAPDatabase.connect(db_dir)
    # temp = db.execute("SELECT * FROM images")
    rows = db.execute("SELECT image_id, name, prior_qw, prior_qx, prior_qy, prior_qz FROM images")
    image_id = {}
    image_names = {}
    image_poses = {}
    for idx, name,qw,qx,qy,qz in rows:
        rot = R.from_quat([qx,qy,qz,qw]).as_matrix()
        image_poses[name] = rot
        image_id[name] = idx
        image_names[str(idx)] = name
    db.close()
    print('read {} images from database'.format(len(image_id)))
    return image_id, image_names, image_poses

def update_3d_keypoints(db_dir):
    '''
    update 3d keypoints from depth maps
    '''
    db = COLMAPDatabase.connect(db_dir)
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))
    for image_id, keypoint in keypoints.items():
        print(image_id, keypoint.shape)

    db.close()
    

def find_top_k_matches(frames:list,matches:list,k:int):
    '''
    find the top k matches
    '''
    m = len(frames)
    if m<k:
        return frames
    else:
        points_number = [match.shape[0] for match in matches]
        selected = np.argsort(points_number)[::-1][:k]
        frame_array = np.array(frames)
        return selected

def read_verified_matches(db_dir:str):
    db = COLMAPDatabase.connect(db_dir)
    original_matches = db.execute("SELECT pair_id, rows FROM matches")
    two_view_geometries = db.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries")
    geometry_matches = []
    original_matches_points = {}
    
    for pair_id, rows in original_matches:
        if rows<1: continue
        id_a, id_b = pair_id_to_image_ids(pair_id)
        original_matches_points[pair_id] = rows
    
    for pair_id, rows, cols, data in two_view_geometries:
        if rows<1: continue
        # if pair_id not in original_matches_points:
        #     print('!!! pair_id {} not in original_matches_points'.format(pair_id))
        id_a, id_b = pair_id_to_image_ids(pair_id)
        matches = blob_to_array(data, np.uint32, (rows, cols)) # (N,2), np.uint32
        geometry_matches.append({
            'db_id':int(id_a),
            'query_id':int(id_b),
            'original_matched':original_matches_points[pair_id], # int
            'verified_matches':matches
        })
    
    print('{} two_view_geometries in the database'.format(len(geometry_matches)))
    db.close()
    
    return geometry_matches

def create_queries_match(geometry_matches):
    queries_match = {} # {'query_name':'db_frames':[db_name_0, ...],'matches':[match_0, ...]}
    
    for verified_match_info in geometry_matches:
        count = 0
        query_id = verified_match_info['query_id']
        tar_id = verified_match_info['db_id']
        query_name = image_names[str(query_id)]
        tar_name = image_names[str(tar_id)]
        matches = verified_match_info['verified_matches']
        # print('{}, {}'.format(query_name,tar_name))
        if query_name not in queries_match:
            queries_match[query_name] = {'db_frames':[tar_name],
                                         'matches':[matches],
                                         'raw_matched':[verified_match_info['original_matched']]}
        else:
            queries_match[query_name]['db_frames'].append(tar_name)
            queries_match[query_name]['matches'].append(matches)
            queries_match[query_name]['raw_matched'].append(verified_match_info['original_matched'])
    return queries_match
    
if __name__=='__main__':
    # Setup the paths
    dataset = Path('/data2/sfm/uc0101b')
    query_split = '05'
    global_pairs = dataset/ 'pairs-query-netvlad50.txt'
    viz_outputs = dataset/'viz'
    if not os.path.exists(viz_outputs): 
        os.mkdir(viz_outputs)

    images = dataset / 'images_upright'
    
    # 
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superpoint+lightglue']
    
    # global match
    t0 = time.time()
    global_features_dir = extract_features.main(retrieval_conf, images, dataset) # global features
    t1 = time.time()
    pairs_from_retrieval.main(global_features_dir, global_pairs, 50,
                        query_prefix='query',db_prefix='db') # global match 
    t2 = time.time()
    print('query global match in {:.3f} sec'.format(t2-t1))

    # local match
    local_features_dir = extract_features.main(feature_conf, images, dataset)
    t3 = time.time()
    loc_matches = match_features.main(
        matcher_conf, global_pairs, feature_conf['output'], dataset) # local match
    t4 = time.time()
    print('query local match in {:.3f} sec'.format(t4-t3))
    # exit(0)
    
    # export database 
    from hloc.triangulation import import_features, import_matches, estimation_and_geometric_verification
    db_dir = dataset/'database.db' #.format(os.path.basename(dataset))
    print('export database to ', db_dir)
    image_ids, image_names, image_poses = get_image_idxs(db_dir)
    
    # exit(0)
    print('read local features from ', local_features_dir)
    import_features(image_ids,db_dir,local_features_dir)
    import_matches(image_ids,db_dir,global_pairs,loc_matches)
    estimation_and_geometric_verification(db_dir,global_pairs,verbose=True)
    
    # todo: import 3d keypoints from depth maps
    
    
    # exit(0)
    # read matches from database
    geometry_matches = read_verified_matches(db_dir) # [{'db_id':int,'query_id':int,'original_matched':int,'verified_matches':np.array(N,2)}]
    queries_match    = create_queries_match(geometry_matches) # {'query_name':'db_frames':[db_name_0, ...],'matches':[match_0, ...]}
    
    # visualize verified matches   
    from hloc.utils import io
    MIN_MATCHES = 50
    MIN_SCORES = 0.2
    TOPK = 10
    
    for query_frame, match_info in queries_match.items():
        print('{} has {} matches'.format(query_frame,len(match_info['db_frames'])))
        db_frames = match_info['db_frames']
        raw_matched = match_info['raw_matched']
        matches = match_info['matches']
        selection = find_top_k_matches(db_frames,matches,TOPK)
        
        for db_frame, verified_matches, raw_matched in zip(np.array(db_frames)[selection],np.array(matches)[selection],np.array(raw_matched)[selection]):
            if verified_matches.shape[0]<MIN_MATCHES:
                break
            query_img = io.read_image(images/query_frame)
            db_img = io.read_image(images/db_frame)
            viz.plot_images([query_img,db_img], dpi=75)
            
            kpts0 = io.get_keypoints(path=local_features_dir, name=query_frame)
            kpts1 = io.get_keypoints(path=local_features_dir, name=db_frame)
            
            mpt0 = kpts0[verified_matches[:,1],:2]
            mpt1 = kpts1[verified_matches[:,0],:2]
            viz.plot_matches(mpt0, mpt1,lw=4.0,a=0.5)
            inliner_text = '{}/{} verified fts'.format(verified_matches.shape[0],raw_matched)
            viz.add_text(0,inliner_text)
            output_name = query_frame.split('/')[-1][:-4]+'-'+db_frame.split('/')[-1]
            viz.save_plot(viz_outputs/output_name)

        # break
            