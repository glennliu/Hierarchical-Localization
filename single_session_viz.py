import os 
import argparse
from os.path import join as osp
from pathlib import Path
import rerun
import open3d as o3d
from rerun_viz import read_rgb_images
from rerun_viz import render_agent_poses, render_point_cloud, render_loop_edges, render_rgb_sequence, render_loop_images
from dataloader import load_euro_images, read_poses, load_sequence_poses, align_euro_keyframes
from tools import read_pnp_folder, read_loop_pairs

def argparse_args():
    parser = argparse.ArgumentParser(description='Render RGB-D sequence and HLoc results in Rerun.')
    parser.add_argument('--rgbd_dataset', type=str,
                        default='/data2/euroc/MH_01_easy',
                        help='Path to the RGB-D dataset folder.')
    parser.add_argument('--sfm_folder', type=str,
                        default='/data2/sfm/single_session/MH_01_easy',
                        help='Path to the SfM folder containing PnP results.')
    parser.add_argument('--rerun_mode', type=int, default=2,
                        help='Rerun mode: 1 for spawn, 2 for connect_tcp.')
    parser.add_argument('--frame_gap', type=int, default=10,
                        help='Frame gap for loading images, default is 1 (load all frames).')
    return parser.parse_args()

if __name__=='__main__':
    print('Load single session RGB-D sequence \
        and render its HLoc results.')
    args = argparse_args()
    rgbd_dataset = args.rgbd_dataset
    #############################################
    
    # 1. Load        
    poses, min_frame = load_sequence_poses(rgbd_dataset,
                                               verbose=True)       
    # rgb_images = read_rgb_images(osp(rgbd_dataset, 'rgb'),
    #                 pose_map=poses,
    #                 verbose=True)
    rgb_images, timestamp = load_euro_images(Path(rgbd_dataset), 
                                  frame_gap=args.frame_gap,
                                  verbose=True)
    
    poses = align_euro_keyframes(poses=poses,
                                     rgb_images=rgb_images,
                                     verbose=True)
    
    # pcd = o3d.io.read_point_cloud(osp(rgbd_dataset,'mesh_o3d.ply'))
    # pnp_predictions = read_pnp_folder(osp(args.sfm_folder, 'pnp'), 
    #                                   verbose=True)
    # pnp_predictions = {p['src_frame']:p for p in pnp_predictions}
    loop_pairs = read_loop_pairs(osp(args.sfm_folder, 'loop_pairs'),
                                 verbose=True)
    pcd = None
    pnp_predictions = {}
    
    # exit(0)
    #
    rerun.init('RGBHLoc', spawn=False)
    if pcd is None: print('No point cloud found.')
    else: render_point_cloud(pcd=pcd, name='src/map')
    
    render_agent_poses(agentName='src',pose_dict=poses)
    render_rgb_sequence(entity_name='src', 
                        rgb_images=rgb_images)
    render_loop_edges(entityName='src', 
                      src_poses=poses,
                      ref_poses=poses,
                      loop_transformations=loop_pairs)                      
    render_loop_images(entityName='src',
                       loop_pairs=loop_pairs,
                       sfm_folder=args.sfm_folder)
    
    if args.rerun_mode==1: 
        rerun.spawn()
    elif args.rerun_mode==2:
        # If run the program in remote server, set the IP to the local machine,
        # after launch rerun in the local machine. 
        # Then, you can visualize the same results at your local machine.
        rerun.connect_tcp() 

