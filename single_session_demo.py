import os 
from pathlib import Path
from ri_looper import single_session_lcd



if __name__=='__main__':
    
    print('Run visual hierarchical localization in a RGB sequence.')
    data_dir = '/data2/sgslam/scans/uc0110_00a'
    output_folder = '/data2/sfm/single_session/uc0110_00a'
    
    single_session_lcd(session_dir=Path(data_dir),
                       output_folder=Path(output_folder))


