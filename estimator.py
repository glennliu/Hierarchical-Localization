import numpy as np
import teaserpp_python
import open3d as o3d
from scipy.spatial.kdtree import KDTree
from copy import deepcopy
import cv2
# import pygicp

class HybridReg:
    def __init__(
        self,
        src_pcd: o3d.geometry.PointCloud,
        tgt_pcd: o3d.geometry.PointCloud,
        refine=None,  # 'icp' or 'vgicp'
        use_pagor=False,
        only_yaw=False,
        ins_wise=False,
        max_ins=256,
        max_pts=256,
    ):
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd
        self.refine = refine
        self.use_pagor = use_pagor
        self.only_yaw = only_yaw
        self.ins_wise = ins_wise
        self.instance_corres = []
        self.max_inst = max_ins
        self.max_points = max_pts

        voxel_size = 0.05
        self.src_pcd = self.src_pcd.voxel_down_sample(voxel_size)
        self.tgt_pcd = self.tgt_pcd.voxel_down_sample(voxel_size)
        if self.refine == "icp":
            if np.array(self.tgt_pcd.normals).shape[0] == 0:
                self.tgt_pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=100)
                )

        if self.use_pagor or self.ins_wise:
            tgt_points = np.asarray(self.tgt_pcd.points)
            self.tgt_kdtree = KDTree(tgt_points)
        if self.use_pagor:
            self.noise_bounds = [0.1, 0.2, 0.3]
        else:
            self.noise_bounds = [0.2]

    def set_model_pred(self, model_pred: dict):

        corres_scores = model_pred["corres_scores"]  # (C,)
        score_mask = corres_scores > 1e-3
        corres_src = model_pred["corres_src_points"][score_mask]  # (C,3)
        corres_ref = model_pred["corres_ref_points"][score_mask]  # (C,3)
        corres_instances = model_pred["corres_instances"][
            score_mask
        ]  # (C,) values in [0,I-1]

        # pred_nodes = model_pred["pred_nodes"]  # (I,)
        pred_scores = model_pred["pred_scores"]
        corres_src_centroids = model_pred["corres_src_centroids"]  # (I,3)
        corres_ref_centroids = model_pred["corres_ref_centroids"]  # (I,3)
        if pred_scores.shape[0] > self.max_inst:
            inst_pair_indices = np.argsort(pred_scores)[
                ::-1
            ]  # (I,), descending order according to pred_scores
            inst_pair_indices = inst_pair_indices[: self.max_inst]
        else:
            inst_pair_indices = np.arange(pred_scores.shape[0])

        instance_corres = []
        for i in inst_pair_indices:
            pred_score = pred_scores[i]
            corres_src_centroid = corres_src_centroids[i]
            corres_ref_centroid = corres_ref_centroids[i]

            matching = {
                "centorid_corr": np.hstack([corres_src_centroid, corres_ref_centroid]),
                "score": pred_score,
            }
            ins_mask = corres_instances == i
            if len(ins_mask) > self.max_points:
                descending_indices = np.argsort(corres_scores[ins_mask])[::-1]
                points_src = corres_src[ins_mask][descending_indices[: self.max_points]]
                points_ref = corres_ref[ins_mask][descending_indices[: self.max_points]]
                point_corr = np.hstack([points_src, points_ref])
            elif len(ins_mask) > 0:
                point_corr = np.hstack([corres_src[ins_mask], corres_ref[ins_mask]])
            else:
                point_corr = np.empty((0, 6))
            matching["point_corr"] = point_corr
            instance_corres.append(matching)

        self.instance_corres = instance_corres
        return instance_corres

    def solve(self):
        if len(self.instance_corres) == 0:
            return np.eye(4)

        if self.ins_wise:
            # Handle each instance individually
            A_corr_list, B_corr_list = self.front_end()
            tf_candidates = []
            for A_corr, B_corr in zip(A_corr_list, B_corr_list):
                tf = self.back_end(A_corr, B_corr)
                tf_candidates.append(tf)

            scores = [self.chamfer_score(tf) for tf in tf_candidates]
            tf = tf_candidates[np.argmax(scores)]
        else:
            A_corr, B_corr = self.front_end()
            tf = self.back_end(A_corr, B_corr)

        return tf

    def front_end(self):
        # 抽取每个instance的匹配
        A_corr_list, B_corr_list = [], []
        for ins_corr in self.instance_corres:
            # Load point correspondences
            A_corr = ins_corr["point_corr"][:, :3].T # (3, N)
            A_centroid = ins_corr["centorid_corr"][:3].reshape(3, 1)
            A_corr = np.hstack([A_corr, A_centroid]) # (3, N+1)
            A_corr_list.append(A_corr)

            B_corr = ins_corr["point_corr"][:, 3:].T
            B_centroid = ins_corr["centorid_corr"][3:].reshape(3, 1)
            B_corr = np.hstack([B_corr, B_centroid])
            B_corr_list.append(B_corr)
        if self.ins_wise:
            return A_corr_list, B_corr_list
        else:
            # Merge correspondences
            A_corr = np.hstack(A_corr_list) # (3, N)
            B_corr = np.hstack(B_corr_list) # (3, N)
            return A_corr, B_corr

    def back_end(self, A_corr, B_corr):

        tf_candidates = []
        for noise_bound in self.noise_bounds:
            tf = self.solve_by_teaser(A_corr, B_corr, noise_bound)
            if tf_candidates == []:
                tf_candidates.append(tf)
            else:
                similar = [
                    np.linalg.norm(tf_candidate[:3, 3] - tf[:3, 3]) < 0.1
                    for tf_candidate in tf_candidates
                ]
                if not any(similar):
                    tf_candidates.append(tf)

        # Verification
        if len(tf_candidates) > 1:
            scores = [self.chamfer_score(tf) for tf in tf_candidates]
            tf = tf_candidates[np.argmax(scores)]
        else:
            tf = tf_candidates[0]

        return tf

    def chamfer_score(self, tf):
        src_pcd = deepcopy(self.src_pcd)
        src_pcd.transform(tf)
        src_points = np.asarray(src_pcd.points)
        nearest_dists = self.tgt_kdtree.query(src_points, k=1)[0]
        nearest_dists = np.clip(nearest_dists, 0, 0.5)
        return -np.mean(nearest_dists)

    def solve_by_teaser(self, A_corr, B_corr, noise_bound):

        teaser_solver = self.get_teaser_solver(noise_bound=noise_bound)
        teaser_solver.solve(A_corr, B_corr)
        solution = teaser_solver.getSolution()
        tf = np.identity(4)
        tf[:3, :3] = solution.rotation
        tf[:3, 3] = solution.translation

        if self.refine is not None:
            if self.refine == "vgicp":
                assert False
                source = np.asarray(self.src_pcd.points)
                target = np.asarray(self.tgt_pcd.points)
                tf = pygicp.align_points(
                    target,
                    source,
                    initial_guess=tf,
                    max_correspondence_distance=1.0,
                    voxel_resolution=0.5,
                    method="VGICP",
                )
            else:
                loss = o3d.pipelines.registration.GMLoss(k=0.1)
                p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    loss
                )
                reg_p2l = o3d.pipelines.registration.registration_icp(
                    self.src_pcd,
                    self.tgt_pcd,
                    0.5,
                    tf,
                    p2l,
                )
                tf = reg_p2l.transformation

        return tf

    def get_teaser_solver(self, noise_bound):
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1.0
        solver_params.noise_bound = noise_bound
        solver_params.estimate_scaling = False
        solver_params.inlier_selection_mode = (
            teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
        )
        solver_params.rotation_tim_graph = (
            teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
        )
        if self.only_yaw:
            solver_params.rotation_estimation_algorithm = (
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.QUATRO
            )
        else:
            solver_params.rotation_estimation_algorithm = (
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 10000
        solver_params.rotation_cost_threshold = 1e-16
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        return solver

def construct_3d_keypoints(kpts, depth, K, depth_scale=1000.0):
    assert depth is not None
    pts3d = np.zeros((kpts.shape[0],3))
    for i in range(kpts.shape[0]):
        x,y = kpts[i,:2]
        z = depth[int(y),int(x)] / depth_scale
        if(z>0.01):
            pts3d[i] = np.dot(np.linalg.inv(K),np.array([x*z,y*z,z]))
        
    return pts3d

def depth2pcd(depth, K, depth_scale=1000.0):
    h, w = depth.shape
    x = np.arange(0,w)
    y = np.arange(0,h)
    xx, yy = np.meshgrid(x,y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = depth.reshape(-1) / depth_scale
    pts = np.vstack([xx*zz,yy*zz,zz]).T
    pts = np.dot(np.linalg.inv(K),pts.T).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def computeLoopTransformation(src_folder, src_frame, ref_folder, ref_frame, kpts_src, kpts_ref, matches):
    src_depth = cv2.imread('{}/depth/{}.png'.format(src_folder,src_frame),cv2.IMREAD_UNCHANGED) 
    if src_depth is None:
        return False, np.eye(4)
    # ref_depth = cv2.imread('{}/depth/{}.png'.format(ref_folder,ref_frame),cv2.IMREAD_UNCHANGED)
    K = np.loadtxt('{}/intrinsic/intrinsic_depth.txt'.format(src_folder))
    K = K[:3,:3]
    
    pts_m_src = kpts_src[matches[:,0],:2] # (M,2)
    pts_m_ref = kpts_ref[matches[:,1],:2] # (M,2)
    pts_m_ref = pts_m_ref.astype(np.float64)
    pts3d_m_src = construct_3d_keypoints(pts_m_src, src_depth, K)    
    
    mask = pts3d_m_src[:,2]>1e-3
    pts3d_m_src = pts3d_m_src[mask]
    pts_m_ref = pts_m_ref[mask]
    
    centroid3d = np.mean(pts3d_m_src, axis=0)
    centroid2d = np.mean(pts_m_ref, axis=0)
    if pts3d_m_src.shape[0]<5:
        return False, np.eye(4)
    
    rtval, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d_m_src, pts_m_ref, K, None) 
    T_ref_src = np.eye(4)
    if rtval:
        # print('{} matches'.format(matches.shape[0]))
        # print('rvec:{},{},{}, tvec:{},{},{}'.format(rvec[0],rvec[1],rvec[2],tvec[0],tvec[1],tvec[2]))
        inliers = inliers.squeeze()
        R = cv2.Rodrigues(rvec)[0]
        T_ref_src[:3,:3] = R
        T_ref_src[:3,3] = tvec.squeeze()
    else:
        print('Fail PnP. {} input matched features'.format(pts3d_m_src.shape[0]))
        
    
    return rtval, T_ref_src

def computeTeaserTransformation(src_folder,
                                src_frame,
                                ref_folder,
                                ref_frame,
                                kpts_src,
                                kpts_ref,
                                matches
                                ):
    K = np.loadtxt('{}/intrinsic/intrinsic_depth.txt'.format(src_folder))
    K = K[:3,:3]
    src_depth = cv2.imread('{}/depth/{}.png'.format(src_folder,src_frame),
                           cv2.IMREAD_UNCHANGED)
    ref_depth = cv2.imread('{}/depth/{}.png'.format(ref_folder,ref_frame),
                           cv2.IMREAD_UNCHANGED)
    src_pcd = depth2pcd(src_depth, K)
    ref_pcd = depth2pcd(ref_depth, K)
    # tmp_folder = '/data2/sfm/single_session/vins_reverse_loop/tmp'
    # o3d.io.write_point_cloud('{}/{}.ply'.format(tmp_folder,src_frame),src_pcd)
    # o3d.io.write_point_cloud('{}/{}.ply'.format(tmp_folder,ref_frame),ref_pcd)
    
    pts_m_src = kpts_src[matches[:,0],:2] # (M,2)
    pts_m_ref = kpts_ref[matches[:,1],:2] # (M,2)
    pts3d_m_src = construct_3d_keypoints(pts_m_src, src_depth, K)
    pts3d_m_ref = construct_3d_keypoints(pts_m_ref, ref_depth, K)
    
    mask = (pts3d_m_src[:,2]>1e-3) & (pts3d_m_ref[:,2]>1e-3)
    pts3d_m_src = pts3d_m_src[mask]
    pts3d_m_ref = pts3d_m_ref[mask]
    
    if pts3d_m_src.shape[0]<5:
        return None, None, None, None
        return False, np.eye(4)
    else:
        hybrid_reg = HybridReg(src_pcd, ref_pcd, refine=None)
        corr_A = pts3d_m_src.T
        corr_B = pts3d_m_ref.T
        return src_pcd, ref_pcd, corr_A, corr_B
        
        T_ref_src = hybrid_reg.solve_by_teaser(pts3d_m_src.T, 
                                               pts3d_m_ref.T, 0.2)
        return True, T_ref_src
        
    
    