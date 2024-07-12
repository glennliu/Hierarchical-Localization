# Convert the recorded dataset to hfnet format.

import os, glob 
import numpy as np

def read_realsense_intrinsics(intrinsic_file_dir):
    with open(intrinsic_file_dir,'r') as f:
        model_name = 'PINHOLE'
        width = 0
        height = 0
        fx = 0.0
        fy = 0.0
        cx = 0.0
        cy = 0.0
        for line in f.readlines():
            parts = line.strip().split('=')
            if 'color_width' in parts[0]: 
                width = int(parts[1].strip())
            elif 'color_height' in parts[0]:
                height = int(parts[1].strip())
            elif 'color_fx' in parts[0]:
                fx = float(parts[1].strip())
            elif 'color_fy' in parts[0]:
                fy = float(parts[1].strip())
            elif 'color_cx' in parts[0]:
                cx = float(parts[1].strip())
            elif 'color_cy' in parts[0]:
                cy = float(parts[1].strip())
            else:
                continue
        params = np.array([fx,fy,cx,cy])
        print('-Read camera intrinsic')
        print('  model_name:',model_name)
        print('  width:',width,'height:',height)
        print('  params:',params)
        return model_name, width, height, params
    return None

def read_scannet_intrinsic(intrinsic_folder):
    width, height = 640, 480
    fx, fy, cx, cy = None, None, None, None
    model_name = 'PINHOLE'

    # with open(os.path.join(intrinsic_folder,'sensor_shapes.txt'),'r') as shape_file:
    #     for line in shape_file.readlines():
    #         parts = line.strip().split(':')
    #         if parts[0]=='color_width':
    #             width = int(parts[1])
    #         elif parts[0]=='color_height':
    #             height = int(parts[1])
    #         else:
    #             continue    
    
    with open(os.path.join(intrinsic_folder,'intrinsic_depth.txt'),'r') as intrin_file:
        intrinsic_mat = np.loadtxt(intrin_file) # (4,4), np.float64
        fx = intrinsic_mat[0,0]
        fy = intrinsic_mat[1,1]
        cx = intrinsic_mat[0,2]
        cy = intrinsic_mat[1,2]
    
    assert width is not None and height is not None and fx is not None and fy is not None and cx is not None and cy is not None
    print('read scannet intrinsic, image dimension {} x {}'.format(width,height))
    print('    fx: {}, fy: {}, cx: {}, cy: {}'.format(fx,fy,cx,cy))
    return model_name,width, height, np.array([fx,fy,cx,cy])

if __name__=='__main__':
    dataroot = '/data2/sgslam/scans'
    sfm_folder = '/data2/sfm/uc0101b'
    database_scan = 'uc0101_04'
    query_scan = 'uc0101_05'
    query_split = query_scan.split('_')[-1].strip()
    DATASET = 'realsense' # 'scannet' or 'realsense'
    if DATASET=='scannet':
        IMAGE_FOLDER = 'color'
        IMAGE_POSFIX = '.jpg'
        DEPTH_POSFIX = '.png'
    elif DATASET=='realsense':
        IMAGE_FOLDER = 'rgb'
        IMAGE_POSFIX = '.png'
        DEPTH_POSFIX = '.png'
    else:
        raise NotImplementedError
    print('reading from {} with postfix: {}'.format(IMAGE_FOLDER,IMAGE_POSFIX))

    images = os.path.join(sfm_folder,'images_upright')
    depth_images = os.path.join(sfm_folder,'depth_upright')
    db_image_list = {} # eg. {'db/000000.png':{'transformation':np.array(4,4)}}
    query_image_list = {}
    
    # build database from target scan
    db_dir = os.path.join(images,'db') 
    query_dir = os.path.join(images,'query')
    db_depth_dir = os.path.join(depth_images,'db')
    query_depth_dir = os.path.join(depth_images,'query')
    if not os.path.exists(sfm_folder):
        os.makedirs(sfm_folder)    
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    if not os.path.exists(depth_images):
        os.makedirs(depth_images)
    if not os.path.exists(db_depth_dir):
        os.makedirs(db_depth_dir)
    if not os.path.exists(query_depth_dir):
        os.makedirs(query_depth_dir)

    # build database
    source_folder = os.path.join(dataroot,database_scan)
    source_rgb_frames = glob.glob(os.path.join(source_folder,IMAGE_FOLDER,'*{}'.format(IMAGE_POSFIX)))  
    for src_rgb in source_rgb_frames:
        src_rgb_name = os.path.basename(src_rgb)
        src_pose_name = src_rgb_name.replace(IMAGE_POSFIX,'.txt')
        src_depth_name = src_rgb_name.replace(IMAGE_POSFIX,DEPTH_POSFIX)
        src_camera_pose = np.loadtxt(os.path.join(dataroot,database_scan,'pose',src_pose_name))

        if not os.path.exists(os.path.join(db_dir,src_rgb_name)):
            os.symlink(src_rgb,os.path.join(db_dir,src_rgb_name))
        if not os.path.exists(os.path.join(db_depth_dir,src_depth_name)):
            os.symlink(os.path.join(dataroot,database_scan,'depth',src_depth_name),os.path.join(db_depth_dir,src_depth_name))
        # db_image_list.append(os.path.join('db',src_rgb_name))
        db_image_list[os.path.join('db',src_rgb_name)] = {'transformation':src_camera_pose}
    print('{} frames in database scan'.format(len(source_rgb_frames)))

    # exit(0)
    # build query folder
    query_folder = os.path.join(query_dir,query_split)
    query_depth_folder = os.path.join(query_depth_dir,query_split)
    FRAME_GAP = 5
    if not os.path.exists(query_folder): 
        os.mkdir(query_folder)
    if not os.path.exists(query_depth_folder):
        os.mkdir(query_depth_folder)

    count_queries = 0
    target_rgb_frames = glob.glob(os.path.join(dataroot,query_scan,IMAGE_FOLDER,'*{}'.format(IMAGE_POSFIX)))
    for idx, tar_rgb in enumerate(target_rgb_frames):
        tar_rgb_name = os.path.basename(tar_rgb)
        tar_pose_name = tar_rgb_name.replace(IMAGE_POSFIX,'.txt')
        tar_depth_name = tar_rgb_name.replace(IMAGE_POSFIX,DEPTH_POSFIX)
        tar_camera_pose = np.loadtxt(os.path.join(dataroot,query_scan,'pose',tar_pose_name))
        if idx%FRAME_GAP!=0:continue
        if not os.path.exists(os.path.join(query_folder,tar_rgb_name)):
            os.symlink(tar_rgb,os.path.join(query_folder,tar_rgb_name))
        if not os.path.exists(os.path.join(query_depth_folder,tar_depth_name)):
            os.symlink(os.path.join(dataroot,query_scan,'depth',tar_depth_name),os.path.join(query_depth_folder,tar_depth_name))
        # query_image_list.append(os.path.join('query',query_split,tar_rgb_name))
        query_image_list[os.path.join('query',query_split,tar_rgb_name)] = {'transformation':tar_camera_pose}
        count_queries += 1
    print('{} frames in query scan'.format(count_queries))
    # exit(0)
    #
    from hloc import colmap_from_nvm
    from hloc.utils.database import COLMAPDatabase, blob_to_array
    from hloc.utils.read_write_model import Camera, Image, CAMERA_MODEL_NAMES 
    
    if DATASET=='scannet':
        model_name, width, height, params = read_scannet_intrinsic(os.path.join(dataroot,database_scan,'intrinsic'))
    elif DATASET=='realsense':    
        model_name, width, height, params = read_realsense_intrinsics(os.path.join(dataroot,query_scan,'intrinsic.txt'))
    else:
        raise NotImplementedError
    # exit(0)
    camera_model = CAMERA_MODEL_NAMES[model_name]
    database_dir = os.path.join(sfm_folder,'database.db')
    
    # exit(0)
    # all_image_list = db_image_list.copy()
    # all_image_list.extend(query_image_list)
    
    db = COLMAPDatabase.connect(database_dir)
    db.create_tables()
    rgb_camera_id = db.add_camera(
        camera_model.model_id, width, height, params,
        camera_id=0, prior_focal_length=True)    
    print('camera {} is written to database'.format(rgb_camera_id))
    
    from scipy.spatial.transform import Rotation as R
    idx = 0
    # for idx, img_dir in enumerate(all_image_list):
    for img_dir, camera_pose in db_image_list.items():
        q = R.from_matrix(camera_pose['transformation'][:3,:3]).as_quat() # (x,y,z,w)
        q_colmap = np.array([q[3],q[0],q[1],q[2]]) # (w,x,y,z)
        t = camera_pose['transformation'][:3,3]
        img_id = db.add_image(
            img_dir, rgb_camera_id, 
            prior_q=q_colmap, prior_t=t, image_id = idx)
        
        # print('image: {}, t:{}'.format(img_dir,t))
        # print('r:{}'.format(R.from_quat(q).as_matrix()))
        idx += 1
        
    
    for img_dir, camera_pose in query_image_list.items():
        q = R.from_matrix(camera_pose['transformation'][:3,:3]).as_quat() # (x,y,z,w)
        q_colmap = np.array([q[3],q[0],q[1],q[2]]) # (w,x,y,z)
        t = camera_pose['transformation'][:3,3]
        img_id = db.add_image(
            img_dir, rgb_camera_id, 
            prior_q=q_colmap, prior_t=t, image_id = idx)
        # print('image: {}, t:{}'.format(img_dir,t))
        # print('r:{}'.format(R.from_quat(q).as_matrix()))
        idx +=1
    
    db.commit()
    
    # test read
    cam_rows = db.execute("SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras")
    for cam_id,model,w_,h_,params,focal_length in cam_rows:
        intrinsic = blob_to_array(params, np.float64, (4,))
        print(intrinsic)
    
    # rows = db.execute("SELECT image_id, name, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images")
    # for idx, name,qw,qx,qy,qz,tx,ty,tz in rows:
    #     rot = R.from_quat([qx,qy,qz,qw]).as_matrix()
    #     t = np.array([tx,ty,tz])
    #     print('image:{}, t:{} \n rot:{}'.format(name,t,rot))
    db.close()
    
    print('write {} images to databset'.format(len(db_image_list)+len(query_image_list)))
    
    
    