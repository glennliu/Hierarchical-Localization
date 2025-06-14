import os, glob 
import open3d as o3d
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import cv2
from scipy.spatial.transform import Rotation as R
# from write_pose_graphs import load_frame_transformations
from tools import load_loop_true_masks

def read_poses(pose_folder:str,
                stride:int=1,
                verbose:bool=False):
    pose_files = glob.glob(os.path.join(pose_folder, "*.txt"))
    pose_files = sorted(pose_files)
    min_frame_id = int(os.path.basename(pose_files[0]).split(".")[0][6:])
    if min_frame_id>0:
        print('Start frame id: {}'.format(min_frame_id))
    
    poses = {}
    for i in range(0,len(pose_files),stride):
        pose_file = pose_files[i]
        frame_name = os.path.basename(pose_file).split(".")[0]
        # frame_id = int(frame_name.split("-")[-1])
        T_wc = np.loadtxt(pose_file)
        # calibrated_frame_id = frame_id - frame_offset
        # poses['frame-{:06d}'.format(calibrated_frame_id)] = T_wc
        poses[frame_name] = T_wc
    
    if verbose:
        print(f"Loaded {len(poses)} poses from {pose_folder}")
    return poses, min_frame_id

def read_rgb_images(rgb_folder:str,
                    pose_map:dict,
                    verbose:bool=False):
    rgb_images = {}
    if len(pose_map)>0:
        for frame_name, _ in pose_map.items():
            rgb_file = os.path.join(rgb_folder, f"{frame_name}.png")
            if os.path.exists(rgb_file):
                kf = cv2.imread(rgb_file)
                rgb_images[frame_name] = kf
    else:
        print('Not implemented yet')
    
    if verbose:
        print(f"Loaded {len(rgb_images)} rgb images from {rgb_folder}")
    return rgb_images

def load_single_agent(scene_dir:str,
                      stride:int=1,
                      max_frames:int=-1,
                      verbose:bool=False):
    poses, start_frame_id = read_poses(os.path.join(scene_dir, "pose"),
                       stride,
                       verbose)
    
    
    if max_frames>0 and len(poses)>max_frames:
        poses = dict(list(poses.items())[:max_frames])

    rgb_images = read_rgb_images(os.path.join(scene_dir, "rgb"),
                                 poses,
                                 verbose)
    
    global_map = o3d.io.read_point_cloud(os.path.join(scene_dir, "mesh_o3d.ply"))
    print('Load {} poses and {} points '.format(len(poses),
                                                len(global_map.points)))
    
    
    return {'poses':poses,
            'rgb_images':rgb_images,
            'global_map':global_map,
            'start_frame_id':start_frame_id}


def render_point_cloud(pcd:o3d.geometry.PointCloud,
                       name:str="map",
                       point_size:float=0.05,
                       color=None):
    """
    Renders a point cloud using Open3D and Rerun.
    """
    # Convert Open3D point cloud to Rerun format
    if color is None:
        color = np.asarray(pcd.colors)
    else:
        color = np.asarray(color)
        
    rr.log(name,
           rr.Points3D(np.asarray(pcd.points),
                       colors=color,
                       radii=point_size),
           static=True)
    
def render_a_camera_pose(name:str,
                         T_wc:np.ndarray):
    '''
    Renders a camera pose using Rerun.
    '''
    q = R.from_matrix(T_wc[:3, :3]).as_quat()
    t = T_wc[:3, 3]
    w,h = 640,480
    fx,fy,cx,cy = 619, 618, 336, 246
    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
    
    rr.log(name,
           rr.Transform3D(translation=t,
                          rotation=rr.Quaternion(xyzw=q)))
    # rr.log(name,
    #        rr.Pinhole(image_from_camera=intrinsic,
    #                   resolution=[w,h]))

def render_agent_poses(agentName:str,
                      pose_dict:dict,
                      frame_gap:float=0.1,
                      color:list=[0,0,255],
                      path_width:float=0.02,
                      frame_offset:int=0):
    
    w,h = 640,480
    fx,fy,cx,cy = 619, 618, 336, 246
    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
    points3d = []

    for frame_name, pose in pose_dict.items():
        frame_id = int(frame_name.split('-')[-1])
        timestamp = frame_id * frame_gap
        # rr.set_time_seconds('time', timestamp)
        rr.set_time_sequence('frame', frame_id-frame_offset)
        q = R.from_matrix(pose[:3,:3]).as_quat()
        t = pose[:3,3]
        rr.log('{}/pose'.format(agentName),
               rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q)))
        rr.log('{}/pose'.format(agentName),
               rr.Pinhole(focal_length=3,
                          width=3,
                          height=3))
        # path
        points3d.append(t)        
        rr.log('{}/path'.format(agentName),
            rr.LineStrips3D(points3d,
                            colors=color,
                            radii=path_width))
    
def render_rgb_sequence(entity_name:str,
                        rgb_images:dict,
                        frame_offset:int=0):
    
    for frame_name, rgb in rgb_images.items():
        frame_id = int(frame_name.split('-')[-1])
        cv2.imwrite('temp_image.png', rgb)  # Save the image temporarily
        rr.set_time_sequence('frame', frame_id-frame_offset)
        rr.log('{}/rgb'.format(entity_name),
               rr.Image(rgb, color_model=rr.ColorModel.BGR,))

def render_loop_edges(name:str,
                    src_poses:dict,
                    ref_poses:dict,
                    loop_transformations:dict,
                    verbose:bool=False):
    for src_frame, loop_info in loop_transformations.items():
        frame_id = int(src_frame.split('-')[-1])
        ref_frame = loop_info['ref_frame']
        # loop_name = '{}-{}'.format(src_frame, ref_frame)
        if src_frame in src_poses and ref_frame in ref_poses:
            rr.set_time_sequence('frame', frame_id)
            src_pose = src_poses[src_frame]
            ref_pose = ref_poses[ref_frame]
            end_points = [src_pose[:3,3].reshape(3),
                        ref_pose[:3,3].reshape(3)]
            
            rr.log('{}'.format(name),
                            rr.LineStrips3D(end_points,
                                            colors=[0,255,0],
                                            radii=0.02))

def render_loop_edges_2(entityName:str,
                    src_poses:dict,
                    ref_poses:dict,
                    loop_eval_list:list,
                    frame_offset:int=0,
                    verbose:bool=False):
    
    loop_edges = []
    loop_colors = []
    
    for loop_eval_dict in loop_eval_list:
        src_frame = loop_eval_dict['src_frame']
        ref_frame = loop_eval_dict['ref_frame']
        tp = loop_eval_dict['true_positive']
        if tp == 0:color = [255,0,0]
        else:color = [0,255,0]
        loop_name = '{}-{}'.format(src_frame, ref_frame)
        if src_frame in src_poses and ref_frame in ref_poses:
            src_pose = src_poses[src_frame]
            ref_pose = ref_poses[ref_frame]
            end_points = [src_pose[:3,3].reshape(3),
                        ref_pose[:3,3].reshape(3)]
            loop_edges.append(end_points)
            loop_colors.append(color)
            frame_id = int(src_frame.split('-')[-1])
            rr.set_time_sequence('frame', frame_id-frame_offset)
            rr.log('{}/loops'.format(entityName),
                            rr.LineStrips3D(loop_edges,
                                               colors=loop_colors,
                                               radii=0.015))

def transform_scene(scene_dict:dict,
                    T_ref_src:np.ndarray):
    """
    Transforms the scene to the reference frame.
    """
    scene_dict['global_map'].transform(T_ref_src)
    for frame_name, pose in scene_dict['poses'].items():
        tmp_pose = T_ref_src @ pose
        scene_dict['poses'][frame_name] = tmp_pose
   

if __name__=='__main__':
    ################# SETTINGS #################
    print('SET ARGS HERE')
    VERBOSE = True
    MAX_FRAME = 5000
    FRAME_STRIDE = 10
    VIZ_MODE = 1 # 0: spawn, 1: conect_tcp, 2: serve_web, 3: save
    FRAME_DURATION = 0.1
    SRC_NAME = 'uc0110_00a'
    REF_NAME = 'uc0110_00c'
    EXT_NAME = 'uc0101_00c'
    
    SRC_SCENE_DIR = 'datasets/'+SRC_NAME
    REF_SCENE_DIR = 'datasets/'+REF_NAME
    EXT_SCENE_DIR = 'datasets/'+EXT_NAME
    
    RESULT_FOLDER = None # '/data2/sgslam/output/hloc/'+SRC_NAME+'-'+REF_NAME
    GT_FILE = '/Users/liuchuhao/dataset/sgslam/gt/{}-{}.txt'.format(SRC_NAME,REF_NAME)
    REMOTE_ADDRESS = "143.89.38.169:9876"
    ############################################
    
    # 1. Load
    print('----- Loading -----')
    src_scene=load_single_agent(SRC_SCENE_DIR,FRAME_STRIDE,MAX_FRAME,VERBOSE)
    ref_scene=load_single_agent(REF_SCENE_DIR,FRAME_STRIDE,MAX_FRAME,VERBOSE)
    if EXT_SCENE_DIR is not None:
        ext_scene=load_single_agent(EXT_SCENE_DIR,FRAME_STRIDE,MAX_FRAME,VERBOSE)
        T_ref_ext = np.loadtxt(os.path.join('/Users/liuchuhao/dataset/sgslam/gt/uc0101_00c-uc0110_00c.txt'))
        transform_scene(ext_scene, T_ref_ext)

    if GT_FILE is not None:
        print('----- Load GT -----')
        T_ref_src = np.loadtxt(GT_FILE)
        transform_scene(src_scene, T_ref_src)
        print('Load gt pose from {}'.format(GT_FILE))
        
    if RESULT_FOLDER is not None:
        loop_eval_list = load_loop_true_masks(os.path.join(RESULT_FOLDER, "eval.txt"))
        print('Load {} loop masks from {}'.format(len(loop_eval_list),
                                                   os.path.join(RESULT_FOLDER, "eval.txt")))
        
    # 2. Visualize
    print('----- Visualizing -----')
    rr.init("multiAgent")
    rr.set_time_sequence('frame',0)
    
    rr.log("world",rr.Transform3D(axis_length=20,scale=1.0),
                static=False
            )
    rr.log("world",rr.Transform3D(translation=[0,0,-3]))
    
    render_point_cloud(src_scene['global_map'],
                       'agentA/global_map',
                       0.03,
                    #    [180,180,0]
                       )
    render_point_cloud(ref_scene['global_map'],
                       'agentB/global_map',
                       0.03,
                    #    [0,180,180]
                       )
    if EXT_SCENE_DIR is not None:
        render_point_cloud(ext_scene['global_map'],
                           'agentC/global_map',
                           0.03,
                           )
        render_agent_poses('agentC',
                           ext_scene['poses'],
                       FRAME_DURATION,
                       [0,200,0],
                       0.03)
    
    
    render_agent_poses('agentA', 
                       src_scene['poses'],
                       FRAME_DURATION,
                       [180,0,180],
                       0.03,
                       )
    render_agent_poses('agentB', 
                       ref_scene['poses'],
                       FRAME_DURATION,
                       [0,0,255],
                       0.03)
    render_rgb_sequence('agentA',
                        src_scene['rgb_images'],
                        FRAME_DURATION,
                        )
    render_rgb_sequence('agentB',
                        ref_scene['rgb_images'],
                        FRAME_DURATION)
    
    if RESULT_FOLDER is not None:
        render_loop_edges_2('loop_edges',
                            src_scene['poses'],
                            ref_scene['poses'],
                            loop_eval_list,
                            0)
    #

    if VIZ_MODE==0:
        rr.spawn()
    elif VIZ_MODE==1:    
       rr.connect_tcp()
    elif VIZ_MODE==2:
        rr.serve_web(web_port=0,ws_port=0,open_browser=False)
    elif VIZ_MODE==3:
        rr.save(os.path.join(RESULT_FOLDER, "rerun.rrd"))
        print('Saved rerun file to {}'.format(os.path.join(RESULT_FOLDER, 
                                                           "rerun.rrd")))
    else:
        print('No visualization mode selected')
    