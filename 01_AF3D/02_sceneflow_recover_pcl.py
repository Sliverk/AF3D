'''
# Generate point cloud for 3D detection
# From sceneflow with PCL_{t-1}
# Include disp0, disp1, flow
# ZHANG Haodi
# INSA ROUEN NORMANDIE
# 2022.04.05
'''

import os
import sys 
import glob
import open3d as o3d

from utils.kitti_utils import *
from utils.visualize import *
import pprint as ppr

from utils.visualizer import *
from utils.structures import *
from utils.data_loader import *

def main():
    ddisp0 = '../data/monosf_selfsup_kitti_3ddet/disp_0'
    ddisp1 = '../data/monosf_selfsup_kitti_3ddet/disp_1'
    dflow = '../data/monosf_selfsup_kitti_3ddet/flow'
    dasyn3d = '../data/kitti_asyn3d'
    dbinsave = '../data/kitti_asyn3d_recbin'
    dtxtlabel = '../data/kitti_3d/training/label_2'
    dtxtcalib = '../data/kitti_3d/training/calib'
    # dsavepcl = 

    flflow = glob.glob(dflow+'/*_10.png')

    for fflow in flflow:
        fflow = '/home/zz/code/06_asyn3DDet/data/monosf_selfsup_kitti_3ddet/flow/000068_10.png'
        fnum = fflow.split('/')[-1].split('.')[0]
        prefix = fnum.split('_')[0]
        print(fnum)
        if fnum == '005242_10': continue

        # S1: Load optical flow
        flow, flow_mask = load_png_flow(fflow)
        # np.savetxt('u.txt', flow[:,:,0],fmt='%d')
        # np.savetxt('v.txt', flow[:,:,1],fmt='%d')
        
        # S2: Load disparity_0 and disparity_1
        fdisp0 = os.path.join(ddisp0, fnum+'.png')
        disp0, mask0 = load_png_disp(fdisp0)

        fdisp1 = os.path.join(ddisp1, fnum+'.png')
        disp1, mask1 = load_png_disp(fdisp1)

        # S3: Recover depth from disparity map
        fcalib = os.path.join(dasyn3d, 'calib_cam', prefix+'.txt')
        depth0 = parse_disp_depth(disp0, fcalib)
        depth1 = parse_disp_depth(disp1, fcalib)

        calib = Calibration(dasyn3d, prefix)

        # S4: Load lidar_0 and project to image plane
        pcl0 = kittiLidarProjection(dasyn3d,fnum)
        pcl0l2i, pcl = pcl0.projection()

        # S5: Apply scene flow to the point cloud
        pt2d_0, pt2d_1 = pixel_align(pcl0l2i, flow, flow_mask, depth0, mask0, depth1, mask1, fcalib)
        
        # S6: Recover lidar_1 point cloud
        pt3d_0= calib.project_image_to_velo(np.asarray(pt2d_0))
        pt3d_1= calib.project_image_to_velo(np.asarray(pt2d_1))
        print(pcl0l2i.shape, pcl.shape)
        print(pt3d_0.shape, pt3d_1.shape, pt2d_1.shape)

        flabel = os.path.join(dtxtlabel, prefix+'.txt')
        fcalib = os.path.join(dtxtcalib, prefix+'.txt')


        gt_bboxes = read_txt_label(flabel, fcalib)['gt_bboxes_3d'].tensor
        # gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
        # gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2

        # annos = load_label(flabel)

        # print(annos['location'])
        # print(annos['dimensions'])
        # print(annos['rotation_y'])

        # vcor_loc = annos['location'][:,[2,0,1]]
        # vcor_loc[:,1] = -vcor_loc[:,1]
        # vcor_loc[:,2] = -vcor_loc[:,2]

        # vcor_dim = annos['dimensions'][:,[2,0,1]]

        # gt_bboxes = np.concatenate((vcor_loc, vcor_dim, (annos['rotation_y'].T)[:,None]), axis=1)
        # print(gt_bboxes)

        fvelo = os.path.join(dasyn3d, f'velodyne/{prefix}_11.bin')
        velo = np.fromfile(fvelo, dtype=np.float32).reshape((-1,4))[:, 0:3]


        # draw_scenes(pt3d_0)
        # draw_scenes(velo,gt_bboxes)
        # draw_scenes(pt3d_1,gt_bboxes)
        # draw_scenes(pt3d_1)
        # draw_scenes(velo)
        
        
        # a = pt3d_1
        # N = a.shape[0]
        # b = np.asarray([1.0]*N).reshape(N,1)
        # c = np.concatenate((a,b), axis=1).astype(np.float32)
        # c.reshape(1,-1).tofile(f'{dbinsave}/{prefix}.bin')

        break

if __name__=='__main__':
    main()