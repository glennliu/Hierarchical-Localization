import os 
import argparse
from os.path import join as osp
import rerun
import open3d as o3d
from rerun_viz import read_poses, read_rgb_images
from rerun_viz import render_agent_poses, render_point_cloud, render_loop_edges, render_rgb_sequence
from tools import read_pnp_folder

def argparse_args():
    parser = argparse.ArgumentParser(description='Render RGB-D sequence and HLoc results in Rerun.')
    parser.add_argument('--rgbd_dataset', type=str,
                        default='/Users/liuchuhao/dataset/sgslam/scans/uc0110_00a',
                        help='Path to the RGB-D dataset folder.')
    parser.add_argument('--sfm_folder', type=str,
                        default='/Users/liuchuhao/dataset/sfm/single_session/uc0110_00a/pnp',
                        help='Path to the SfM folder containing PnP results.')
    parser.add_argument('--rerun_mode', type=int, default=2,
                        help='Rerun mode: 1 for spawn, 2 for connect_tcp.')
    return parser.parse_args()

if __name__=='__main__':
    print('Load single session RGB-D sequence \
        and render its HLoc results.')
    args = argparse_args()
    rgbd_dataset = args.rgbd_dataset
    #############################################
    
    # 1. Load                  
    poses, min_frame = read_poses(osp(rgbd_dataset,'pose'))
    rgb_images = read_rgb_images(osp(rgbd_dataset, 'rgb'),
                    pose_map=poses,
                    verbose=True)

    pcd = o3d.io.read_point_cloud(osp(rgbd_dataset,'mesh_o3d.ply'))
    pnp_predictions = read_pnp_folder(args.sfm_folder, False)
    pnp_predictions = {p['src_frame']:p for p in pnp_predictions}
    
    # exit(0)
    #
    rerun.init('RGBHLoc', spawn=False)
    if pcd is None: print('No point cloud found.')
    else: render_point_cloud(pcd=pcd, name='src/map')
        
    render_agent_poses(agentName='src/camera',pose_dict=poses)
    render_rgb_sequence(entity_name='src/camera', 
                        rgb_images=rgb_images)
    render_loop_edges(name='src/loop_edges', 
                      src_poses=poses,
                      ref_poses=poses,
                      loop_transformations=pnp_predictions)                      
    
    if args.rerun_mode==1: 
        rerun.spawn()
    elif args.rerun_mode==2:
        rerun.connect_tcp()

