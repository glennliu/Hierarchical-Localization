import numpy as np 

class TimingRecord:
    def __init__(self):
        self.header = ['netvlad', 'gmatch', 'superpoint', 'lmatch']
        self.netvlad = []
        self.gmatch = []
        self.superpoint = []
        self.lmatch = []
        self.pnp = []
        self.src_frames = 0

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

    def write_to_file(self, filename):
        with open(filename,'w') as f:
            f.write('# ')
            f.write(' '.join(self.header)+'\n')
            
            f.write('NetVLAD: {}, {:.6f}\n'.format(self.netvlad.shape[0], np.sum(self.netvlad)))
            f.write('GMatch: {}, {:.6f}\n'.format(self.gmatch.shape[0], np.sum(self.gmatch)))
            f.write('SuperPoint: {}, {:.6f}\n'.format(self.superpoint.shape[0], np.sum(self.superpoint)))
            f.write('LMatch: {}, {:.6f}\n'.format(self.lmatch.shape[0], np.sum(self.lmatch)))
            f.write('PnP: {}, {:.6f}\n'.format(self.pnp.shape[0], np.sum(self.pnp)))
            f.close()
            
            print('Timing records saved to',filename)

    def write_to_file_new(self, filename):
        
        with open(filename,'w') as f:
            f.write('frame number: {}'.format(self.src_frames)+'\n')
            f.write('netvlad: {:.3f}'.format(self.netvlad[0])+'\n')
            f.write('superpoint: {:.3f}'.format(self.superpoint[0])+'\n')
        
        f.close()
        

class SequenceTimingRecord:
    def __init__(self):
        self.db_frames = []
        self.gmatch_times = []
        self.lightglue_times = []
        self.pnp_times = []
        self.netvlad = 0.0
        self.superpoint = 0.0
        
    def analysis(self):
        
        self.db_frames = np.hstack(self.db_frames)
        self.gmatch_times = np.hstack(self.gmatch_times)
        self.lightglue_times = np.hstack(self.lightglue_times)
        self.pnp_times = np.hstack(self.pnp_times)
        self.netvlad = 1000.0 * self.netvlad
        self.superpoint = 1000.0 * self.superpoint
        
        print('********* Timing analysis ********')
        print('           Frames   Av. Time')
        print('NetVLAD: {:6d} {:10.4f}'.format(self.gmatch_times.shape[0], self.netvlad / self.gmatch_times.shape[0]))
        print('SuperPoint: {:6d} {:10.4f}'.format(self.lightglue_times.shape[0], self.superpoint / self.lightglue_times.shape[0]))
        print('GMatch: {:6d} {:10.4f}'.format(self.gmatch_times.shape[0], np.mean(self.gmatch_times)))
        print('LightGlue: {:6d} {:10.4f}'.format(self.lightglue_times.shape[0], np.mean(self.lightglue_times)))
        print('PnP: {:6d} {:10.4f}'.format(self.pnp_times.shape[0], np.mean(self.pnp_times)))
        
        average_sum = np.mean(self.gmatch_times) + np.mean(self.lightglue_times) + np.mean(self.pnp_times) \
                    + self.netvlad / self.gmatch_times.shape[0] + self.superpoint / self.lightglue_times.shape[0]
        print('Average sum: {:10.4f}'.format(average_sum))
        
    def export(self, filename):
        sumary_data = np.vstack([self.db_frames, self.gmatch_times, self.lightglue_times])
        np.save(filename, sumary_data)
        print('Timing records saved to',filename)
        
