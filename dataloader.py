import os 
from os.path import join as osp
from pathlib import Path


def read_euro_images(session_dir:Path,
                     verbose:bool=False):
    
    frame_list = [p.relative_to(session_dir).as_posix() for p in (session_dir/'cam0'/'data').iterdir()]
    frame_list = sorted(frame_list)
    timestamp = [float(os.path.basename(p).split('.')[0]) for p in frame_list] # in ns
    timestamp = [t/1e6 for t in timestamp] # convert to ms
    t_start = timestamp[0]
    timestamp = [t - t_start for t in timestamp] # make it start from 0
    
    if verbose:
        print('Load {} images from {}. t in ({:.1f},{:.1f})'.format(len(frame_list), 
                                                            session_dir,
                                                            timestamp[0],
                                                            timestamp[-1]))    
    return timestamp, frame_list