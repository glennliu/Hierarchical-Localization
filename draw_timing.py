import os, glob 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection


if __name__=='__main__':
    sfm_dataroot = '/data2/sfm'
    
    timing = np.load(os.path.join(sfm_dataroot,'multi_agent','timing.npy'))
    # sort the timing by db size
    timing = timing[:,np.argsort(timing[0,:])]
    
    db_size = timing[0,:]
    gmatch_timing = timing[1,:]
    lightglue_timing = timing[2,:]
    print('Loaded {} frames of timinig'.format(timing.shape[1]))    
    
    # plot the timing data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(db_size, gmatch_timing, color='tab:blue')
    ax.plot(db_size, lightglue_timing, color='tab:orange')

    # create the events marking the x data points
    xevents1 = EventCollection(db_size, color='tab:red', linelength=0.05)

    # create the events marking the y data points
    yevents1 = EventCollection(gmatch_timing, color='tab:blue', linelength=0.05,
                            orientation='vertical')
    yevents2 = EventCollection(lightglue_timing, color='tab:orange', linelength=0.05,
                            orientation='vertical')

    ax.add_collection(xevents1)
    ax.add_collection(yevents1)
    ax.add_collection(yevents2)
    
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 300])
    # ax.set_title('Timing Analysis')
    
    ax.set_xlabel('Database Frame Number')
    ax.set_ylabel('Time (ms)')
    ax.legend(['Place Recognize','LightGlue'])
    ax.grid()
    
    plt.savefig(os.path.join(sfm_dataroot,'multi_agent','timing.png'))