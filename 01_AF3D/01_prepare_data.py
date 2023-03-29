'''
# Prepare data for 3D detection
# Follow KITTI 3D Object Detection dataset devkit/mapping.txt
# Get the previous frame from KITTI RAW 
# Include image_2, image_3, velodyne
# ZHANG Haodi
# INSA ROUEN NORMANDIE
# 2022.04.04
'''

import os

def main():
    dKITTI3D = '../data/kitti_3d'
    dKITTIRAW = '../data/kitti_raw/data_raw'
    dsavepath = '../data/kitti_asyn3d'
    dsaveimage2 = os.path.join(dsavepath,'image_2')
    dsaveimage3 = os.path.join(dsavepath,'image_3')
    dsavevelodyne = os.path.join(dsavepath,'velodyne')
    dsavecalibcam = os.path.join(dsavepath,'calib_cam')
    dsavecalibvelo = os.path.join(dsavepath,'calib_velo')
    if not os.path.exists(dsaveimage2): os.makedirs(dsaveimage2)
    if not os.path.exists(dsaveimage3): os.makedirs(dsaveimage3)
    if not os.path.exists(dsavevelodyne): os.makedirs(dsavevelodyne)
    if not os.path.exists(dsavecalibcam): os.makedirs(dsavecalibcam)
    if not os.path.exists(dsavecalibvelo): os.makedirs(dsavecalibvelo)

    fmaprand = os.path.join(dKITTI3D, 'devkit_object/mapping/train_rand.txt')
    fmapraw = os.path.join(dKITTI3D, 'devkit_object/mapping/train_mapping.txt')

    with open(fmaprand, 'r') as f: maprand = f.readline().strip('\n').split(',')
    with open(fmapraw, 'r') as f: mapraw = f.readlines()
    
    for index, mapindex in enumerate(maprand):
        mapindex = int(mapindex)-1
        rawinfo = mapraw[mapindex].strip('\n').split(' ')
        drawframe = os.path.join(dKITTIRAW, rawinfo[0], rawinfo[1])

        frameindex = '%010d' %(int(rawinfo[2])-1)
        fimage2_10 = os.path.join(drawframe, 'image_02/data', frameindex+'.png')
        fimage3_10 = os.path.join(drawframe, 'image_03/data', frameindex+'.png')
        fvelodyne_10 = os.path.join(drawframe, 'velodyne_points/data', frameindex+'.bin')

        fsaveimage2_10 = os.path.join(dsaveimage2, '%06d_10.png' %index)
        fsaveimage3_10 = os.path.join(dsaveimage3, '%06d_10.png' %index)
        fsavevelodyne_10 = os.path.join(dsavevelodyne, '%06d_10.bin' %index)

        fimage2_11 = os.path.join(drawframe, 'image_02/data', rawinfo[2]+'.png')
        fimage3_11 = os.path.join(drawframe, 'image_03/data', rawinfo[2]+'.png')
        fvelodyne_11 = os.path.join(drawframe, 'velodyne_points/data', rawinfo[2]+'.bin')

        fsaveimage2_11 = os.path.join(dsaveimage2, '%06d_11.png' %index)
        fsaveimage3_11 = os.path.join(dsaveimage3, '%06d_11.png' %index)
        fsavevelodyne_11 = os.path.join(dsavevelodyne, '%06d_11.bin' %index)

        dcalib = os.path.join(dKITTIRAW, rawinfo[0])
        fcalibcam = os.path.join(dcalib, 'calib_cam_to_cam.txt')
        fcalibvelo = os.path.join(dcalib, 'calib_velo_to_cam.txt')
        fsavecalibcam = os.path.join(dsavecalibcam, '%06d.txt' %index)
        fsavecalibvelo = os.path.join(dsavecalibvelo, '%06d.txt' %index)

        
        # save frame t-1
        os.system(f'cp {fimage2_10}  {fsaveimage2_10}')
        os.system(f'cp {fimage3_10}  {fsaveimage3_10}')
        os.system(f'cp {fvelodyne_10}  {fsavevelodyne_10}')

        # save frame t
        os.system(f'cp {fimage2_11}  {fsaveimage2_11}')
        os.system(f'cp {fimage3_11}  {fsaveimage3_11}')
        os.system(f'cp {fvelodyne_11}  {fsavevelodyne_11}')

        # save calib
        os.system(f'cp {fcalibcam}  {fsavecalibcam}')
        os.system(f'cp {fcalibvelo}  {fsavecalibvelo}')


        if index % 100 == 0: 
            print(f'Finish {index}')

        # if index == 199: break



if __name__ == '__main__':
    main()