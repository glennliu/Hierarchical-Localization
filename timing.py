import numpy as np 

class TimingRecord:
    def __init__(self):
        self.header = ['netvlad', 'gmatch', 'superpoint', 'lmatch']
        self.netvlad = []
        self.gmatch = []
        self.superpoint = []
        self.lmatch = []
        self.pnp = []

    def analysis(self):
        self.netvlad = np.array(self.netvlad)
        self.gmatch = np.array(self.gmatch)
        self.superpoint = np.array(self.superpoint)
        self.lmatch = np.array(self.lmatch)
        self.pnp = np.array(self.pnp)
        
        print('********* Timing analysis ********')
        print('           Frames   Av. Time')
        print('NetVLAD: {:6d} {:10.4f}'.format(len(self.netvlad), np.mean(self.netvlad)))
        print('GMatch: {:6d} {:10.4f}'.format(len(self.gmatch), np.mean(self.gmatch)))
        print('SuperPoint: {:6d} {:10.4f}'.format(len(self.superpoint), np.mean(self.superpoint)))
        print('LMatch: {:6d} {:10.4f}'.format(len(self.lmatch), np.mean(self.lmatch)))
        print('PnP: {:6d} {:10.4f}'.format(len(self.pnp), np.mean(self.pnp)))

