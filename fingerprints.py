from os import path
import utils
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *

STORAGE_PATH = 'storage/'

W = (23, 23)

mcc_sigma_s = 7.0
mcc_tau_psi = 400.0
mcc_mu_psi = 1e-2

def read(path: str):
    fingerprint = cv.imread('samples/sample_1_1.png', cv.IMREAD_GRAYSCALE)
    # img = show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')
    cv.imshow("Fingerprint", fingerprint)
    cv.waitKey(0) 
    cv.destroyAllWindows() 

def pipeline(path: str) -> tuple:
    fingerprint = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # Needs to be resized. Use 256, 256

    # Segment
    gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
    gx2, gy2 = gx**2, gy**2
    gm = np.sqrt(gx2 + gy2)
    sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
    thr = sum_gm.max() * 0.2
    mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)

    gxx = cv.boxFilter(gx2, -1, W, normalize = False)
    gyy = cv.boxFilter(gy2, -1, W, normalize = False)
    gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
    sum_gxx_gyy = gxx + gyy
    strengths = np.divide(cv.sqrt((gxx_gyy**2 + gxy2**2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)
    
    region = fingerprint[10:90,80:130]
    smoothed = cv.blur(region, (5,5), -1)
    xs = np.sum(smoothed, 1)
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
    distances = local_maxima[1:] - local_maxima[:-1]
    ridge_period = np.average(distances)

    # Enhance
    or_count = 8
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]
    nf = 255-fingerprint
    all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)
    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    # Convert to gray scale and apply the mask
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)

    # Minutiae
    _, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
    skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)
    cn_filter = np.array([
        [  1,  2,  4],
        [128,  0,  8],
        [ 64, 32, 16]
    ])

    def compute_crossing_number(values):
        return np.count_nonzero(values < np.roll(values, -1))


    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

    skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
    # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255] 
    neighborhood_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
    # Apply the lookup table to obtain the crossing number of each pixel from the byte value of its neighborhood
    cn = cv.LUT(neighborhood_values, cn_lut)
    # Keep only crossing numbers on the skeleton
    cn[skeleton==0] = 0

    minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]
    mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))

    def compute_next_ridge_following_directions(previous_direction, values):    
        next_positions = np.argwhere(values!=0).ravel().tolist()
        if len(next_positions) > 0 and previous_direction != 8:
            # There is a previous direction: return all the next directions, sorted according to the distance from it,
            #                                except the direction, if any, that corresponds to the previous position
            next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
            if next_positions[-1] == (previous_direction + 4) % 8: # the direction of the previous position is the opposite one
                next_positions = next_positions[:-1] # removes it
        return next_positions
    
    r2 = 2**0.5 # sqrt(2)
    # The eight possible (x, y) offsets with each corresponding Euclidean distance
    xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]
    
    # LUT: for each 8-neighborhood and each previous direction [0,8], 
    #      where 8 means "none", provides the list of possible directions
    nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

    
    def follow_ridge_and_compute_angle(x, y, d = 8):
        px, py = x, y
        length = 0.0
        while length < 20: # max length followed
            next_directions = nd_lut[neighborhood_values[py,px]][d]
            if len(next_directions) == 0:
                break
            # Need to check ALL possible next directions
            if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
                break # another minutia found: we stop here
            # Only the first direction has to be followed
            d = next_directions[0]
            ox, oy, l = xy_steps[d]
            px += ox ; py += oy ; length += l
        # check if the minimum length for a valid direction has been reached
        return math.atan2(-py+y, px-x) if length >= 10 else None
    

    valid_minutiae = []
    for x, y, term in filtered_minutiae:
        d = None
        if term: # termination: simply follow and compute the direction        
            d = follow_ridge_and_compute_angle(x, y)
        else: # bifurcation: follow each of the three branches
            dirs = nd_lut[neighborhood_values[y,x]][8] # 8 means: no previous direction
            if len(dirs)==3: # only if there are exactly three branches
                angles = [follow_ridge_and_compute_angle(x+xy_steps[d][0], y+xy_steps[d][1], d) for d in dirs]
                if all(a is not None for a in angles):
                    a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                    d = angle_mean(a1, a2)                
        if d is not None:
            valid_minutiae.append( (x, y, term, d) )

    mcc_radius = 70
    mcc_size = 16

    g = 2 * mcc_radius / mcc_size
    x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
    y = x[..., np.newaxis]
    iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
    ref_cell_coords = np.column_stack((x[ix], x[iy]))

    mcc_sigma_s = 7.0
    mcc_tau_psi = 400.0
    mcc_mu_psi = 1e-2

    def Gs(t_sqr):
        """"Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
        return np.exp(-0.5 * t_sqr / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)

    def Psi(v):
        """"Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
        return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))
    
    # n: number of minutiae
    # c: number of cells in a local structure

    xyd = np.array([(x,y,d) for x,y,_,d in valid_minutiae]) # matrix with all minutiae coordinates and directions (n x 3)

    # rot: n x 2 x 2 (rotation matrix for each minutia)
    d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
    rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

    # rot@ref_cell_coords.T : n x 2 x c
    # xy : n x 2
    xy = xyd[:,:2]
    # cell_coords: n x c x 2 (cell coordinates for each local structure)
    cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])

    # cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
    # xy                                 : (1 x 1) x n x 2
    # cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
    # dists: n x c x n (for each cell of each local structure, the distance from all minutiae)
    dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)

    # cs : n x c x n (the spatial contribution of each minutia to each cell of each local structure)
    cs = Gs(dists)
    diag_indices = np.arange(cs.shape[0])
    cs[diag_indices,:,diag_indices] = 0 # remove the contribution of each minutia to its own cells

    # local_structures : n x c (cell values for each local structure)
    local_structures = Psi(np.sum(cs, -1))

    ############################################################################################################

    return (fingerprint, valid_minutiae, local_structures)


num_p = 5 # For simplicity: a fixed number of pairs
def similarity(ls1, ls2) -> float:
    dists = np.linalg.norm(ls1[:,np.newaxis,:] - ls2, axis = -1)
    dists /= np.linalg.norm(ls1, axis = 1)[:,np.newaxis] + np.linalg.norm(ls2, axis = 1) # Normalize as in eq. (17) of MCC paper
    pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
    score = 1 - np.mean(dists[pairs[0], pairs[1]]) # See eq. (23) in MCC paper
    return score

def draw_minutiae_and_cylinder(fingerprint, origin_cell_coords, minutiae, values, i, show_cylinder = True):

    def _compute_actual_cylinder_coordinates(x, y, t, d):
        c, s = math.cos(d), math.sin(d)
        rot = np.array([[c, s],[-s, c]])    
        return (rot@origin_cell_coords.T + np.array([x,y])[:,np.newaxis]).T
    
    res = draw_minutiae(fingerprint, minutiae)    
    if show_cylinder:
        for v, (cx, cy) in zip(values[i], _compute_actual_cylinder_coordinates(*minutiae[i])):
            cv.circle(res, (int(round(cx)), int(round(cy))), 3, (0,int(round(v*255)),0), 1, cv.LINE_AA)
    return res

def draw_match_pairs(f1, m1, v1, f2, m2, v2, cells_coords, pairs, i, show_cylinders = True):
    #nd = _current_parameters.ND
    h1, w1 = f1.shape
    h2, w2 = f2.shape
    p1, p2 = pairs
    res = np.full((max(h1,h2), w1+w2, 3), 255, np.uint8)
    res[:h1,:w1] = draw_minutiae_and_cylinder(f1, cells_coords, m1, v1, p1[i], show_cylinders)
    res[:h2,w1:w1+w2] = draw_minutiae_and_cylinder(f2, cells_coords, m2, v2, p2[i], show_cylinders)
    for k, (i1, i2) in enumerate(zip(p1, p2)):
        (x1, y1, *_), (x2, y2, *_) = m1[i1], m2[i2]
        cv.line(res, (int(x1), int(y1)), (w1+int(x2), int(y2)), (0,0,255) if k!=i else (0,255,255), 1, cv.LINE_AA)
    return res