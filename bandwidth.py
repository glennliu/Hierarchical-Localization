import numpy as np 

class BandwidthSummary():
    def __init__(self) -> None:
        self.dim_netvlad = 4096
        self.dim_sp_feature = 256 + 1 # desciptor + score
        self.bytes_per_uv = 2
        
        self.image_number = 0
        self.sp_number = []

    def analysis(self):
        self.sp_number = np.array(self.sp_number)
        
        
        netvlad_bw = self.image_number * self.dim_netvlad * 4
        sp_bw = self.sp_number.sum() * (self.dim_sp_feature * 4 + 2 * self.bytes_per_uv)
        rgb_bw = self.image_number * 640 * 480 * 3 * 1
        
        netvlad_bw = netvlad_bw / 1024 # KB
        sp_bw = sp_bw / 1024 # KB
        rgb_bw = rgb_bw / 1024
        total_bw = netvlad_bw + sp_bw + rgb_bw
        
        print('********* Bandwidth analysis ********')
        print('Images: {}, features: {}'.format(self.image_number, self.sp_number.sum()))
        print('           Frames   Bandwidth(KB) AverageBW(KB)')
        print('NetVLAD: {:6d} {:10.1f} {:10.1f}'.format(self.image_number, netvlad_bw, netvlad_bw/self.image_number))
        print('SuperPoint: {:6d} {:10.1f} {:10.1f}'.format(self.sp_number.shape[0], sp_bw, sp_bw/self.sp_number.shape[0]))
        print('RGB Image: {:6d} {:10.1f} {:10.1f}'.format(self.image_number, rgb_bw, rgb_bw/self.image_number))
        print('Total: {:6d} {:10.1f} {:10.1f}'.format(self.image_number, total_bw, total_bw/self.image_number))

        
        