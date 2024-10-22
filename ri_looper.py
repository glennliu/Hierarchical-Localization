import os, glob 

# add current path to sys.path
import sys
sys.path.append('/home/cliuci/code_ws/Hierarchical-Localization')

import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from hloc import extract_features, match_features, localize_sfm
from hloc import visualization
import hloc.pairs_from_retrieval as pairs_from_retrieval
import hloc.pairs_from_exhaustive as pairs_from_exhaustive
from hloc.utils import read_write_model, viz
from hloc.utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
from hloc.utils import io
from hloc.utils.parsers import parse_retrieval
from hloc.utils.io import get_matches

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

def save_loop_pairs(out_dir:str, loop_pairs:list):
    with open(out_dir,'w') as f:
        f.write('# src_frame ref_frame\n')
        for pair in loop_pairs:
            f.write('{} {}\n'.format(pair[0],pair[1]))
        f.close()
        
def save_loop_transformation(out_dir:str, loop_pairs:list, loop_transformations:list, valid_only:bool):
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

def save_frame_list(frames:list, out_dir:str):
    with open(out_dir,'w') as f:
        for frame in frames:
            f.write('{}\n'.format(frame))
        f.close()

def read_frame_list(filedir:str):
    frames = []
    print(filedir)
    with open(filedir,'r') as f:
        for line in f.readlines():
            frames.append(line.strip())
        f.close()
        return frames
    
def read_loop_pairs(in_dir:str):
    loop_pairs = []
    with open(in_dir,'r') as f:
        for line in f.readlines():
            if '#' in line: continue
            loop_pairs.append(line.strip().split())
        f.close()
    return loop_pairs

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
            
def get_ordered_matches(dense_match_dir:str, query_name, ref_candidate_names):
    candidate_names = []
    candidate_match_numbers = []
    
    
    for candidate in ref_candidate_names:
        matches, scores = get_matches(dense_match_dir, query_name, candidate)        
        candidate_names.append(candidate)
        candidate_match_numbers.append(matches.shape[0])
    
    candidate_match_numbers = np.array(candidate_match_numbers)
    rank_selections = np.argsort(candidate_match_numbers)[::-1]
    rank_names = []
    
    for idx in rank_selections:
        rank_names.append(candidate_names[idx])
        
    return rank_names
        

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

def prepare_candidate_images(ref_image_list, src_frame_dir):
    src_frame_name = src_frame_dir.split('/')[-1].split('.')[0]
    src_frame_id = int(src_frame_name.split('-')[-1])
    candidate_frames = []
    
    for ref_frame in ref_image_list:
        ref_frame_name = ref_frame.split('/')[-1].split('.')[0]
        ref_frame_id = int(ref_frame_name.split('-')[-1])
        if(ref_frame_id<=src_frame_id):
            candidate_frames.append(ref_frame)
            
    return candidate_frames

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

def construct_3d_keypoints(kpts, depth, K, depth_scale=1000.0):
    assert depth is not None
    pts3d = np.zeros((kpts.shape[0],3))
    for i in range(kpts.shape[0]):
        x,y = kpts[i,:2]
        z = depth[int(y),int(x)] / depth_scale
        if(z>0.01):
            pts3d[i] = np.dot(np.linalg.inv(K),np.array([x*z,y*z,z]))
        
    return pts3d
    

def computeLoopTransformation(src_folder, src_frame, ref_folder, ref_frame, kpts_src, kpts_ref, matches):
    src_depth = cv2.imread('{}/depth/{}.png'.format(src_folder,src_frame),cv2.IMREAD_UNCHANGED) 
    if src_depth is None:
        return False, np.eye(4)
    # ref_depth = cv2.imread('{}/depth/{}.png'.format(ref_folder,ref_frame),cv2.IMREAD_UNCHANGED)
    K = np.loadtxt('{}/intrinsic/intrinsic_depth.txt'.format(src_folder))
    K = K[:3,:3]
    
    pts_m_src = kpts_src[matches[:,0],:2] # (M,2)
    pts_m_ref = kpts_ref[matches[:,1],:2] # (M,2)
    pts_m_ref = pts_m_ref.astype(np.float64)
    pts3d_m_src = construct_3d_keypoints(pts_m_src, src_depth, K)    
    
    mask = pts3d_m_src[:,2]>1e-3
    pts3d_m_src = pts3d_m_src[mask]
    pts_m_ref = pts_m_ref[mask]
    
    centroid3d = np.mean(pts3d_m_src, axis=0)
    centroid2d = np.mean(pts_m_ref, axis=0)
    # print('centroid3d: ', centroid3d)
    # print('centroid2d: ', centroid2d)
    print('ref_frame: ', ref_frame)
    if pts3d_m_src.shape[0]<5:
        return False, np.eye(4)
    
    # print('object points: ', pts3d_m_src.shape, pts3d_m_src.dtype)
    # print('image points: ', pts_m_ref.shape, pts_m_ref.dtype)
    rtval, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d_m_src, pts_m_ref, K, None) 
    T_ref_src = np.eye(4)
    if rtval:
        # print('{} matches'.format(matches.shape[0]))
        # print('rvec:{},{},{}, tvec:{},{},{}'.format(rvec[0],rvec[1],rvec[2],tvec[0],tvec[1],tvec[2]))
        inliers = inliers.squeeze()
        R = cv2.Rodrigues(rvec)[0]
        T_ref_src[:3,:3] = R
        T_ref_src[:3,3] = tvec.squeeze()
    else:
        print('Fail PnP. {} input matched features'.format(pts3d_m_src.shape[0]))
        
    
    return rtval, T_ref_src
    
class TicToc:
    def __init__(self):
        self.tic()
    def tic(self):
        self.t0 = time.time()
    def toc(self):
        return time.time()-self.t0
        
def multi_session_slam(dataroot, sfm_output_dataroot,src_scan, ref_scan):
    from hloc.utils import io

    # Setup the paths
    # dataroot = Path('/data2/sgslam/scans')
    # sfm_dataroot = Path('/data2/sfm')
    # src_scan = 'ab0201_03a'
    # ref_scan = 'ab0201_03c'
    overwrite = True
    VIZ = True
    SOLVEPNP = True
    SAMPLE_GAP = 100
    MIN_MATCHES = 30
    
    # Initialization
    output_folder = sfm_output_dataroot/'{}-{}'.format(src_scan,ref_scan)
    global_pairs = output_folder/ 'pairs-query-netvlad50.txt'
    loop_folder = output_folder/'loop_pairs'
    dense_matches = output_folder/ 'matches-superpoint-lightglue.h5'
    viz_outputs = output_folder/'viz'
    # pnp_outputs = output_folder/'pnp'
    global_feat_name = 'netvlad.h5'
    local_feat_name = 'superpoint.h5'
    
    src_global_feat_dir = output_folder/'{}_{}'.format(src_scan,global_feat_name)
    ref_global_feat_dir = output_folder/'{}_{}'.format(ref_scan,global_feat_name)
    src_local_feat_dir = output_folder/'{}_{}'.format(src_scan,local_feat_name)
    ref_local_feat_dir = output_folder/'{}_{}'.format(ref_scan,local_feat_name)
    global_topk = 10
    dense_topk = 3

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(viz_outputs):
        os.mkdir(viz_outputs)
    if not os.path.exists(loop_folder):
        os.mkdir(loop_folder)

    src_image_list = [p.relative_to(dataroot/src_scan).as_posix() for p in (dataroot/src_scan/'rgb').iterdir()]
    ref_image_list = [p.relative_to(dataroot/ref_scan).as_posix() for p in (dataroot/ref_scan/'rgb').iterdir()]
    src_image_list = sorted(src_image_list)[::SAMPLE_GAP]
    ref_image_list = sorted(ref_image_list)[::SAMPLE_GAP]
    
    # src_image_list = src_image_list[:1]
    save_frame_list([img.split('/')[-1].split('.')[0] for img in src_image_list], 
                    output_folder/'src_frames.txt')
    save_frame_list([img.split('/')[-1].split('.')[0] for img in ref_image_list],
                    output_folder/'ref_frames.txt')
    
    # 
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superpoint+lightglue']
    from timing import TimingRecord
    timing_record = TimingRecord()
    
    # 1. global features
    tictoc = TicToc()
    extract_features.main(retrieval_conf, 
                        dataroot/src_scan, 
                        image_list=src_image_list, 
                        feature_path=src_global_feat_dir, 
                        overwrite=overwrite,
                        duration_list = timing_record.netvlad) # global features         
    print('Extract global features in {:.3f} sec'.format(tictoc.toc())) #41.1 sec
    extract_features.main(retrieval_conf, 
                          dataroot/ref_scan, 
                          image_list=ref_image_list, 
                          feature_path=ref_global_feat_dir, 
                          overwrite=overwrite) # global features

    for query_frame in src_image_list:
        global_loop_dir = loop_folder/'{}.txt'.format(query_frame.split('/')[-1].split('.')[0])
   
        
        pairs_from_retrieval.main(src_global_feat_dir, global_loop_dir, global_topk,
                                query_list=[query_frame],
                                db_descriptors=ref_global_feat_dir,
                                duration_list=timing_record.gmatch)
    print('{} query frames. Global pairs from retrieval in {:.3f} sec'.format(
        len(src_image_list), tictoc.toc())) # 42.6 sec    
        
    # 2. local match
    tictoc.tic()
    extract_features.main(feature_conf, 
                          dataroot/src_scan, 
                          image_list=src_image_list,
                          feature_path=src_local_feat_dir, 
                          overwrite=overwrite,
                          duration_list=timing_record.superpoint) # local features
    print('Extract superpoint features takes {:.3f} sec'.format(tictoc.toc())) # 59.8s
    extract_features.main(feature_conf, dataroot/ref_scan, image_list=ref_image_list,feature_path=ref_local_feat_dir, overwrite=overwrite)
    
    for query_frame in src_image_list:
        global_loop_dir = loop_folder/'{}.txt'.format(query_frame.split('/')[-1].split('.')[0])
        match_features.main(matcher_conf, 
                            global_loop_dir, 
                            src_local_feat_dir, 
                            matches= dense_matches, 
                            features_ref=ref_local_feat_dir, 
                            overwrite=overwrite,
                            duration_list=timing_record.lmatch) # local match
    t_dense_match = tictoc.toc()
    print('Local matches {} query frames. It takes {:.3f} sec. {:.3f}s/frame'.format(
        len(src_image_list),t_dense_match,t_dense_match/len(src_image_list))) # 209.1s

    # 
    # timing_record.analysis()

    # 3. Estimate relative poses
    # global_pair_candidates = parse_retrieval(global_pairs)
    loop_pairs = [] # [query_frame, ref_frame]
    loop_transformation = [] # T_ref_query
    print('******** Estimate relative poses **********')
    
    for query_frame in src_image_list:
        global_loop_dir = loop_folder/'{}.txt'.format(query_frame.split('/')[-1].split('.')[0])
        
        # if query_frame not in global_pair_candidates:
        if not os.path.exists(global_loop_dir):
            print('no matches for {}'.format(query_frame))
            continue
        print('--- {} ---'.format(query_frame))
        kpts0 = io.get_keypoints(path=src_local_feat_dir, name=query_frame)
        rgb0 = io.read_image(dataroot/src_scan/query_frame)
        candidate_ref_frames = parse_retrieval(global_loop_dir)[query_frame]

        rank_candidate_frames = get_ordered_matches(dense_matches, query_frame, candidate_ref_frames)
        
        for ref_candidate in rank_candidate_frames[:dense_topk]:
            kpts1 = io.get_keypoints(path=ref_local_feat_dir, name=ref_candidate)
            rgb1 = io.read_image(dataroot/ref_scan/ref_candidate)
            matches, scores = get_matches(dense_matches, query_frame, ref_candidate)
            assert(matches.shape[0]==scores.shape[0])
            if((matches[:,0].max()>=kpts0.shape[0]) or 
               (matches[:,1].max()>=kpts1.shape[0])): 
                print('skip invalid matches for {}-{}'.format(query_frame,ref_candidate))
                continue
            if(matches.shape[0]<MIN_MATCHES):continue
                 
            if VIZ:
                viz.plot_images([rgb0, rgb1], dpi=75)
                viz.plot_matches(kpts0[matches[:, 0], :2], kpts1[matches[:, 1], :2], lw=1.5, a=0.5)
                inliner_text = '{}/{} matched pts'.format(matches.shape[0],kpts0.shape[0])
                viz.add_text(0,inliner_text)
                output_name = query_frame.split('/')[-1][:-4]+'-'+ref_candidate.split('/')[-1][:-4]
                viz.save_plot(viz_outputs/output_name)

            tictoc.tic()
            if SOLVEPNP:
                rtval, T_ref_src = computeLoopTransformation(dataroot/ src_scan, 
                                          query_frame.split('/')[-1].split('.')[0],
                                          dataroot/ ref_scan,
                                          ref_candidate.split('/')[-1].split('.')[0],
                                          kpts0, kpts1,
                                          matches)
            else: T_ref_src = np.eye(4)

            timing_record.pnp.append(tictoc.toc())
            loop_pairs.append([query_frame.split('/')[-1].split('.')[0], 
                               ref_candidate.split('/')[-1].split('.')[0]])
            loop_transformation.append(T_ref_src)

    save_loop_transformation(output_folder/'loop_transformations.txt', loop_pairs, loop_transformation, False)
    # save_loop_pairs(output_folder/'loop_pairs.txt', loop_pairs)
    timing_record.analysis()
    timing_record.write_to_file(output_folder/'timing.txt')
    
    return True
    exit(0)
    
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
            query_img = io.read_image(image_dir/query_frame)
            db_img = io.read_image(image_dir/db_frame)
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

def multi_agent_slam(dataroot, sfm_output_folder,src_scan, ref_scan):
    from hloc.utils import io
    overwrite = False
    VIZ = False
    SOLVEPNP = True
    SAMPLE_GAP = 10
    MIN_MATCHES = 3
    
    # Initialization
    output_folder = sfm_output_folder/'{}-{}'.format(src_scan,ref_scan)
    # global_pairs = output_folder/ 'pairs-query-netvlad50.txt'
    loop_folder = output_folder/'loop_pairs'
    dense_matches = output_folder/ 'matches-superpoint-lightglue.h5'
    viz_outputs = output_folder/'viz'
    global_feat_name = 'netvlad.h5'
    local_feat_name = 'superpoint.h5'
    
    src_global_feat_dir = output_folder/'{}_{}'.format(src_scan,global_feat_name)
    ref_global_feat_dir = output_folder/'{}_{}'.format(ref_scan,global_feat_name)
    src_local_feat_dir = output_folder/'{}_{}'.format(src_scan,local_feat_name)
    ref_local_feat_dir = output_folder/'{}_{}'.format(ref_scan,local_feat_name)
    global_topk = 10
    global_min_score = 0.01
    dense_topk = 1

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(viz_outputs):
        os.mkdir(viz_outputs)
    if not os.path.exists(loop_folder):
        os.mkdir(loop_folder)

    src_image_list = [p.relative_to(dataroot/src_scan).as_posix() for p in (dataroot/src_scan/'rgb').iterdir()]
    ref_image_list = [p.relative_to(dataroot/ref_scan).as_posix() for p in (dataroot/ref_scan/'rgb').iterdir()]
    src_image_list = sorted(src_image_list)[::SAMPLE_GAP]
    ref_image_list = sorted(ref_image_list)[::SAMPLE_GAP]
    # src_image_list = ['rgb/frame-000671.png']
    print('src frames: {}, ref frames: {}'.format(len(src_image_list), len(ref_image_list)))
    
    save_frame_list([img.split('/')[-1].split('.')[0] for img in src_image_list], 
                    output_folder/'src_frames.txt')
    save_frame_list([img.split('/')[-1].split('.')[0] for img in ref_image_list],
                    output_folder/'ref_frames.txt')

    # 
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superpoint+lightglue']
    from timing import TimingRecord
    timing_record = TimingRecord()
    match_timing_record = []

    # 1. global features
    tictoc = TicToc()
    extract_features.main(retrieval_conf, 
                        dataroot/src_scan, 
                        image_list=src_image_list, 
                        feature_path=src_global_feat_dir, 
                        overwrite=overwrite,
                        duration_list = timing_record.netvlad) # global features         
    extract_features.main(retrieval_conf, 
                          dataroot/ref_scan, 
                          image_list=ref_image_list, 
                          feature_path=ref_global_feat_dir, 
                          overwrite=overwrite) # global features

    # 2. Local features
    tictoc.tic()
    extract_features.main(feature_conf,
                            dataroot/src_scan,
                            image_list=src_image_list,
                            feature_path=src_local_feat_dir,
                            overwrite=overwrite,
                            duration_list=timing_record.superpoint) # local features
    print('Extract superpoint features takes {:.3f} sec'.format(tictoc.toc())) 
    extract_features.main(feature_conf, 
                          dataroot/ref_scan, 
                          image_list=ref_image_list,
                          feature_path=ref_local_feat_dir, 
                          overwrite=overwrite)
    if overwrite:
        timing_record.src_frames = len(src_image_list)
        timing_record.write_to_file_new(output_folder/'features_timing.txt')
    
    # 3. Hierachical matching
    for query_id, query_frame in enumerate(src_image_list):
        # if query_id<1: continue
        query_frame_name = query_frame.split('/')[-1].split('.')[0]
        query_frame_id = int(query_frame_name.split('-')[-1])
        if query_frame_id<1: continue
        # if query_id>5:break
        global_loop_dir = loop_folder/'{}.txt'.format(query_frame.split('/')[-1].split('.')[0])
        # db_frames = ref_image_list[:query_frame_id]
        db_frames = prepare_candidate_images(ref_image_list, query_frame)
        if len(db_frames)<1:continue
        print('query {} in {} db frames'.format(query_frame, len(db_frames)))
        
        # global match
        tictoc.tic()
        duration_gmatch = []
        global_match_pairs = pairs_from_retrieval.main(src_global_feat_dir, 
                                                    global_loop_dir, 
                                                    num_matched = min(global_topk,len(db_frames)),
                                                    query_list=[query_frame],
                                                    db_list = db_frames,
                                                    db_descriptors=ref_global_feat_dir,
                                                    duration_list= duration_gmatch,
                                                    min_score=global_min_score)
        duration_gmatch = 1000 * duration_gmatch[0]
        # print('find {} global matches in {:.3f} sec'.format(global_match_pairs,duration_gmatch))

        # local match
        tictoc.tic()
        duration_lmatch = []
        match_features.main(matcher_conf, 
                            global_loop_dir, 
                            src_local_feat_dir, 
                            matches= dense_matches, 
                            features_ref=ref_local_feat_dir, 
                            overwrite=overwrite,
                            duration_list=duration_lmatch)
        if len(duration_lmatch)>0:
            duration_lmatch = 1000 * duration_lmatch[0]
        else:
            duration_lmatch = 0
        print('Global match takes {:.3f} ms, local match takes {:.3f} ms'.format(duration_gmatch,duration_lmatch))
        frame_time = np.array([len(db_frames), global_match_pairs, duration_gmatch, duration_lmatch])
        match_timing_record.append(frame_time)

    #
    if overwrite:
        match_timing_record = np.array(match_timing_record)
        np.savetxt(output_folder/'match_timing.txt',match_timing_record, fmt='%.3f')
    
    # 4. Loop closure
    # loop_pairs = [] # [query_frame, ref_frame]
    # loop_transformation = [] # T_ref_query
    print('******** Estimate relative poses **********')
    pnp_folder = output_folder/'pnp'
    if os.path.exists(pnp_folder)==False:
        os.mkdir(pnp_folder)
    pnp_timing_record = []
    
    for query_frame in src_image_list:
        pnp_duration = 0.0
        global_loop_dir = loop_folder/'{}.txt'.format(query_frame.split('/')[-1].split('.')[0])
        
        # if query_frame not in global_pair_candidates:
        if not os.path.exists(global_loop_dir):
            print('no matches for {}'.format(query_frame))
            continue
        print('--- {}-{} ---'.format(src_scan, query_frame))
        kpts0 = io.get_keypoints(path=src_local_feat_dir, name=query_frame)
        rgb0 = io.read_image(dataroot/src_scan/query_frame)
        candidate_ref_frames = parse_retrieval(global_loop_dir)
        if query_frame not in candidate_ref_frames:
            continue
        candidate_ref_frames = candidate_ref_frames[query_frame]

        tictoc.tic()
        rank_candidate_frames = get_ordered_matches(dense_matches, query_frame, candidate_ref_frames)
        pnp_duration += tictoc.toc()
        print('{} ref frames'.format(len(rank_candidate_frames)))
        if(len(rank_candidate_frames)<1): continue
        select_candidates = min(dense_topk,len(rank_candidate_frames))

        # compute PnP for all the loops
        frame_pairs = []
        loop_transformatins = []
        query_frame_name = query_frame.split('/')[-1].split('.')[0]

        for ref_candidate in rank_candidate_frames[:select_candidates]:
            ref_frame_name = ref_candidate.split('/')[-1].split('.')[0]
            kpts1 = io.get_keypoints(path=ref_local_feat_dir, name=ref_candidate)
            rgb1 = io.read_image(dataroot/ref_scan/ref_candidate)
            matches, scores = get_matches(dense_matches, query_frame, ref_candidate)
            assert(matches.shape[0]==scores.shape[0])
            if(matches is None):continue
            if(matches.shape[0]<1): continue
            if((matches[:,0].max()>=kpts0.shape[0]) or 
               (matches[:,1].max()>=kpts1.shape[0])): 
                print('skip invalid matches for {}-{}'.format(query_frame,ref_candidate))
                continue
            if(matches.shape[0]<MIN_MATCHES):continue
                 
            if VIZ:
                viz.plot_images([rgb0, rgb1], dpi=75)
                viz.plot_matches(kpts0[matches[:, 0], :2], kpts1[matches[:, 1], :2], lw=1.5, a=0.5)
                # inliner_text = '{}/{} matched pts'.format(matches.shape[0],kpts0.shape[0])
                # viz.add_text(0,inliner_text)
                output_name = query_frame_name+'-'+ref_frame_name
                # query_frame.split('/')[-1][:-4]+'-'+ref_candidate.split('/')[-1][:-4]
                viz.save_plot(viz_outputs/output_name)

            tictoc.tic()
            if SOLVEPNP:
                rtval, T_ref_src = computeLoopTransformation(dataroot/ src_scan, 
                                          query_frame_name,
                                          dataroot/ ref_scan,
                                          ref_frame_name,
                                          kpts0, kpts1,
                                          matches)
            else: T_ref_src = np.eye(4)
            pnp_duration += tictoc.toc()
            
            frame_pairs.append([query_frame_name, ref_frame_name])
            loop_transformatins.append(T_ref_src)

        # Save loop transformations
        save_loop_transformation(pnp_folder/'{}.txt'.format(query_frame_name), frame_pairs, loop_transformatins, False)
        pnp_timing_record.append(pnp_duration)

    pnp_timing_record = 1000 * np.array(pnp_timing_record)
    np.savetxt(output_folder/'pnp_timing.txt',pnp_timing_record, fmt='%.3f')
    print('Finished')

if __name__=='__main__':
    
    dataroot = Path('/data2/sgslam/scans')
    sfm_dataroot = Path('/data2/sfm')    
    
    #     
    scene_pairs =[
                #   ['uc0110_00a','uc0110_00b'],
                ['uc0110_00a','uc0110_00c'],
                ['uc0115_00a','uc0115_00b'],
                ['uc0115_00a','uc0115_00c'],
                ['uc0204_00a','uc0204_00c'],
                ['uc0204_00a','uc0204_00b'],
                ['uc0111_00a','uc0111_00b'],
                ['ab0201_03c','ab0201_03a'],
                ['ab0302_00a','ab0302_00b'],
                ['ab0403_00c','ab0403_00d'],
                ['ab0401_00a','ab0401_00b']
                  ]
    
    for pair in scene_pairs:
        print('Processing {}-{}'.format(pair[0],pair[1]))
        # multi_session_slam(dataroot, sfm_dataroot/'multi_session', pair[0], pair[1])
        multi_agent_slam(dataroot, sfm_dataroot/'multi_agent', pair[0], pair[1])
