import numpy as np

def pixel_align2(l2i, pcl, ofl, ofl_mask, depth0, mask0, depth1, mask1):
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
    # with open(fcalib, 'r') as f: lines = f.readlines()
    # lines = [x.strip() for x in lines]
    # S02_rect = np.matrix([float(x) for x in lines[23].split(' ')[1:]])
    # IMG_W, IMG_H = S02_rect[0,0], S02_rect[0,1]
    # print(IMG_W, IMG_H)
    IMG_W, IMG_H = ofl.shape[1], ofl.shape[0]
    # print(IMG_W, IMG_H)

    ul,vl,zl = l2i
    num = ul.shape[1]
    pt2d_0 = []
    pt2d_1 = []
    pt3d_0 = []
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
        pt3d_0.append(pcl[:3,ix])
        pt2d_1.append([u1,v1,(delta_d+z)])
    # print(pt3d_0)
    return np.asarray(pt2d_0), np.asarray(pt2d_1), np.asarray(pt3d_0)

def grid3dmap(pt2d, pt3d, hw):
    grid = np.zeros((hw[0],hw[1],3))
    # print(hw[0],hw[1])
    for ix, p in enumerate(pt2d):
        u, v, z = int(p[0]), int(p[1]), p[2]
        if grid[v,u,0] == 0: grid[v,u] = pt3d[ix]
        elif pt3d[ix,0] < grid[v,u,0]: grid[v,u] = pt3d[ix]
    return grid

####################### 
class PCL_correlation():
    def __init__(self, m0, m1, o, s=10):
        self.map0 = m0
        self.map1 = m1
        self.flow = o
        self.h = self.flow.shape[0]
        self.w = self.flow.shape[1]
        self.step = s
        self.count = 0
    
    def rbc(self):
        for i in range(self.h):
            j = 0
            while j < self.w-1:
                j += 1
                if self.map0[i,j,0] > 0.0:
                    # print('ORI: i:',i,'j:',j)
                    j = self.recu_corr(i,j)
                    # print('AFT: i:',i,'j:',j)
        print(self.count)
        return self.retriev_3dpcl()

    def recu_corr(self, posi, posj):
        next_j = posj
        for j in range(posj+1, self.w):
            next_j = j 
            if self.map0[posi,j,0] > 0.0:                
                if j - posj <= self.step:
                    if self.map0[posi,j,0] <= self.map0[posi,posj,0]:
                        next_j = self.recu_corr(posi,j)
                    ret = self.correlation(posi, posj, j)                    
                    if ret == True: self.count += 1
                break              
        return next_j

    def correlation(self, c0i, c00j, c01j):
        c10i = int(c0i + self.flow[c0i,c00j,1])
        c10j = int(c00j + self.flow[c0i,c00j,0])
        c11i = int(c0i + self.flow[c0i,c01j,1]) 
        c11j = int(c01j + self.flow[c0i,c01j,0])
        p003d = self.map0[c0i,c00j]
        p013d = self.map0[c0i,c01j]
        p103d = self.map1[c10i,c10j]
        p113d = self.map1[c11i,c11j]

        if c11j - c10j > self.step: return False

        l = np.linalg.norm(p013d-p003d)

        flag = None
        LEFT = 101
        RIGHT = 1101
        # P1(x1,y1,z1) is far to corr, ||k*P1-P2|| = l.
        # x1 is depth in velo view.
        if p103d[0] > p113d[0]:
            x1, y1, z1 = p103d
            x2, y2, z2 = p113d
            flag = LEFT
        else:
            x1, y1, z1 = p113d
            x2, y2, z2 = p103d
            flag = RIGHT

        a = x1**2 + y1**2 + z1**2
        b = -2 * (x1*x2 + y1*y2 + z1*z2)
        c = x2**2 + y2**2 + z2**2 - l
        
        k1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        k2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        
        
        # print(k1,k2)

        x1k1 = x1 * k1 - x2
        x1k2 = x1 * k2 - x2

        k = 0 # if sqrt is zero, delete point
        if x1k1 < 0: k = k2
        elif x1k2 < 0: k = k1
        elif x1k1 < x1k2: k = k1
        elif x1k1 > x1k2: k = k2

        if flag == LEFT: self.map1[c10i,c10j] = k * p103d
        elif flag == RIGHT: self.map1[c11i,c11j] = k * p113d
        return True
    
    def retriev_3dpcl(self):
        pt3d = []
        for i in range(self.h):
            for j in range(self.w):
                p = self.map1[i,j]
                if p[0] != 0: pt3d.append(p)
        return np.asarray(pt3d)
    

# def recu():
#     return j

# def rbc(map0, map1, flow, step=10):
#     for i in range(flow.shape[0]):
#         for j in range(flow.shape[1]):
#             if l2igrid[i,j,2] != 0:

#     return 0
