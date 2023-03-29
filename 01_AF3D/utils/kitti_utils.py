'''
# Util for kitti sceneflow and 3D detection
# ZHANG Haodi
# INSA ROUEN NORMANDIE
# 2022.04.05
'''
from __future__ import print_function

import os 
import cv2
import numpy as np 


'''
Input: 
    filepath: Optical flow file path, png, uint16
Output: 
    flow[:,:,0:2]: w*h*2 (du,dv) optical flow
    (1-invalid_idx*1)[:,:,None]: w*h*1 valid mask
'''
def load_png_flow(filepath):
    # Remove png module, use opencv
    flow = cv2.imread(filepath,-1)
    flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
    invalid_idx = (flow[:,:,2] == 0)
    trueflow = (flow[:,:,0:2].astype(np.float32) - 2**15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return trueflow, (1-invalid_idx*1)[:,:,None]

def load_png_disp(filepath):
    disp_np = cv2.imread(filepath,-1).astype(np.float32) / 256.0
    disp_np = np.expand_dims(disp_np, axis=2)
    disp_mask = (disp_np > 0).astype(np.float64)
    return disp_np, disp_mask

def parse_disp_depth(disp, fcalib):
    with open(fcalib, 'r') as f: lines = f.readlines()
    lines = [x.strip() for x in lines]
    P02_rect = np.matrix([float(x) for x in lines[25].split(' ')[1:]])
    B = 0.54
    f = P02_rect[0,0]
    mask = disp > 0
    return f * B / (disp + (1.0 - mask))

def load_label(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


class kittiLidarProjection():
    def __init__(self, dkittiasyn3d, fnum='000000_10'):
        prefix = fnum.split('_')[0]
        self.calib_velo = os.path.join(dkittiasyn3d, f'calib_velo/{prefix}.txt')
        self.calib_cam  = os.path.join(dkittiasyn3d, f'calib_cam/{prefix}.txt')
        self.pcl = os.path.join(dkittiasyn3d, f'velodyne/{fnum}.bin')
    
    def parse_velo_calib(self, calib_velo):
        with open(calib_velo, 'r') as f: lines = f.readlines()
        lines = [x.strip() for x in lines]
        R = np.matrix([float(x) for x in lines[1].split(' ')[1:]]).reshape(3,3)
        T = np.matrix([float(x) for x in lines[2].split(' ')[1:]]).reshape(3,1)
        Tr_velo_to_cam = np.concatenate((R,T), axis=1)
        # To homogeneous
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0,0,0,1], axis=0)
        # print(Tr_velo_to_cam.shape)
        # (4, 4)
        return Tr_velo_to_cam

    def parse_cam_calib(self, calib_cam):
        with open(calib_cam, 'r') as f: lines = f.readlines()
        lines = [x.strip() for x in lines]
        R00_rect = np.matrix([float(x) for x in lines[8].split(' ')[1:]])
        S02_rect = np.matrix([float(x) for x in lines[23].split(' ')[1:]])
        P02_rect = np.matrix([float(x) for x in lines[25].split(' ')[1:]])
        # To homogeneous
        R00_rect = R00_rect.reshape(3, 3)
        R00_rect = np.insert(R00_rect, 3, values=[0,0,0], axis=0)
        R00_rect = np.insert(R00_rect, 3, values=[0,0,0,1], axis=1)
        P02_rect = P02_rect.reshape(3, 4)
        # print(R00_rect.shape, S02_rect.shape, P02_rect.shape)
        # (4, 4) (1, 2) (3, 4)
        return R00_rect, S02_rect, P02_rect
    
    def projection(self):
        Tr_velo_to_cam = self.parse_velo_calib(self.calib_velo)
        R00_rect, S02_rect, P02_rect = self.parse_cam_calib(self.calib_cam)

        # Load Velodyne
        velo = np.fromfile(self.pcl, dtype=np.float32).reshape((-1,4))[:, 0:3]

        # Projection
        velo = np.insert(velo,3,1,axis=1).T
        # velo = np.delete(velo, np.where(velo[0,:]<0), axis=1)

        l2i = P02_rect * R00_rect * Tr_velo_to_cam * velo
        velo = np.delete(velo, np.where(l2i[2,:]<0)[1], axis=1)
        l2i = np.delete(l2i, np.where(l2i[2,:]<0)[1], axis=1)
        l2i[:2] /= l2i[2,:]

        # filter point out of canvas
        u,v,z = l2i
        IMG_W, IMG_H = S02_rect[0,0], S02_rect[0,1]
        u_out = np.logical_or(u<0, u>IMG_W)
        v_out = np.logical_or(v<0, v>IMG_H)
        outlier = np.logical_or(u_out, v_out)
        l2i = np.delete(l2i,np.where(outlier),axis=1)
        velo = np.delete(velo,np.where(outlier), axis=1)

        return l2i, velo 


def pixel_align(l2i, ofl, ofl_mask, depth0, mask0, depth1, mask1, fcalib):
    '''
    image2 coord:
         ----> x-axis (u) W 1242
        |
        |
        v y-axis (v) H 375
    ofl: [H, W, 2] [du,dv]
    depth: [H, W, 1]
    l2i: [3, N] [u, v, z]
    '''
    with open(fcalib, 'r') as f: lines = f.readlines()
    lines = [x.strip() for x in lines]
    S02_rect = np.matrix([float(x) for x in lines[23].split(' ')[1:]])
    IMG_W, IMG_H = S02_rect[0,0], S02_rect[0,1]

    ul,vl,zl = l2i
    num = ul.shape[1]
    pt2d_0 = []
    pt2d_1 = []
    # ofl: [v, u, 2]
    # depth: [v, u, 1]
    for ix in range(num):
        u = int(ul[0,ix])
        v = int(vl[0,ix])
        z = zl[0,ix]

        # if mask0[v,u] == 0: continue        
        # if ofl_mask[v,u] == 0: continue
        u1 = int(u + ofl[v,u,0])
        v1 = int(v + ofl[v,u,1])

        if v1 >= IMG_H or v1 < 0 or u1 >= IMG_W or u1 < 0: 
            # print(ofl[v,u,0],ofl[v,u,1])
            continue
        # if mask1[v1, u1] == 0: continue
        delta_d = depth1[v1,u1][0] - depth0[v,u][0]
        # delta_e = (z - depth0[v,u][0])/z
        pt2d_0.append([u,v,depth0[v,u][0]])
        pt2d_1.append([u1,v1,(delta_d+z)])
    return np.asarray(pt2d_0), np.asarray(pt2d_1)
    # return pt2d_1


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, dkittiasyn3d, prefix='000000'):

        self.calib_velo = os.path.join(dkittiasyn3d, f'calib_velo/{prefix}.txt')
        self.calib_cam  = os.path.join(dkittiasyn3d, f'calib_cam/{prefix}.txt')

        calibs = self.read_calib_file()
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def parse_velo_calib(self):
        with open(self.calib_velo, 'r') as f: lines = f.readlines()
        lines = [x.strip() for x in lines]
        R = np.matrix([float(x) for x in lines[1].split(' ')[1:]]).reshape(3,3)
        T = np.matrix([float(x) for x in lines[2].split(' ')[1:]]).reshape(3,1)
        Tr_velo_to_cam = np.concatenate((R,T), axis=1).reshape(3,4)
        # To homogeneous
        # Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0,0,0,1], axis=0)        
        return Tr_velo_to_cam
    
    def parse_cam_calib(self):
        with open(self.calib_cam, 'r') as f: lines = f.readlines()
        lines = [x.strip() for x in lines]
        R0_rect = np.matrix([float(x) for x in lines[8].split(' ')[1:]])
        P2 = np.matrix([float(x) for x in lines[25].split(' ')[1:]])
        # To homogeneous
        # P2 = P2.reshape(3, 4)
        # R0_rect = R0_rect.reshape(3, 3)
        # R0_rect = np.insert(R0_rect, 3, values=[0,0,0], axis=0)
        # R0_rect = np.insert(R0_rect, 3, values=[0,0,0,1], axis=1)        
        return P2, R0_rect

    def read_calib_file(self):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        
        data['P2'], data['R0_rect'] = self.parse_cam_calib()
        data['Tr_velo_to_cam'] = self.parse_velo_calib()

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        # return pts_2d[:, 0:2]
        return pts_2d

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr



