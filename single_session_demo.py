import os 
import argparse
from pathlib import Path
from dataloader import read_euro_images
from ri_looper import single_session_lcd

def parse_args():
    parser = argparse.ArgumentParser(description='Run visual hierarchical localization in a RGB sequence.')
    parser.add_argument('--data_dir', type=str, default='/data2/euroc/MH_01_easy',
                        help='Path to the data directory containing the RGB sequence.')
    parser.add_argument('--output_folder', type=str, default='/data2/sfm/single_session/MH_01_easy',
                        help='Path to the output folder for results.')
    return parser.parse_args()

if __name__=='__main__':
    
    print('Run visual hierarchical localization in a RGB sequence.')
    args = parse_args()
    session_dir = Path(args.data_dir)
    
    # 
    timestamp, frame_list = read_euro_images(session_dir, verbose=True)
    
    #
    single_session_lcd(session_dir=session_dir,
                       timestamp=timestamp,
                       src_image_list=frame_list,
                       output_folder=Path(args.output_folder))
    
    
    
