# -*- encoding=utf-8 -*-

import os, shutil, pickle, math
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label, binary_erosion, binary_dilation, find_objects


hello

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Functions

def create_folder(name):
    '''
    Create a new folder. If it already exists, it is erased.
    '''
    if Path(name).exists():
        shutil.rmtree(name)
    os.mkdir(name)
    
# ------------------------------------------------------------------------------------------------------------------------------------------ #

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray =  0.1140 * r + 0.2989 * g + 0.5870 * b

    return gray

#------------------------------------------------------------------------------------------------------------------------------------------ #
# User input

L_file_name = ['input/GR_graincontact_left_manifest_snapshot.png',
               'input/GR_graincontact_right_manifest_snapshot.png']
#file_name = 'GR_graincontact_dissolutionframe.png'

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Plan

create_folder('png')
create_folder('data')

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Load data

L_img = []
# iterate on file
for i_file_name in range(len(L_file_name)):
    file_name = L_file_name[i_file_name]

    # load image
    img = imageio.imread(file_name)
    img = np.asarray(img)

    # convert rgb into gray
    img = rgb2gray(img)

    # save gray picture
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    im = ax1.imshow(img, interpolation = 'nearest', cmap='gray')
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Gray map '+str(i_file_name),fontsize = 30)
    fig.tight_layout()
    fig.savefig('png/gray_map_'+str(i_file_name)+'.png')
    plt.close(fig)

    # save img
    L_img.append(img)

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Extract data

# define extraction
threshold_min = 0
threshold_max = 150

L_img_extracted = []
L_img_extracted_conv = []
# iterate on image 
for i_img in range(len(L_img)):
    img = L_img[i_img]

    # extract grain
    img_extracted = (threshold_min < img) & (img < threshold_max)

    # save extracted picture
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    im = ax1.imshow(img_extracted, interpolation = 'nearest', cmap='gray')
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Extracted map '+str(i_img),fontsize = 30)
    fig.tight_layout()
    fig.savefig('png/extracted_map_'+str(i_img)+'.png')
    plt.close(fig)

    # save img_extracted
    # as img_extracted contains boolean, it is convert to integers for img_extracted_conv
    L_img_extracted.append(img_extracted)
    L_img_extracted_conv.append(img_extracted*1)

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Identify data

L_img_labelled = []
# iterate on extracted image
for i_img_extracted in range(len(L_img_extracted)):
    img_extracted = L_img_extracted[i_img_extracted]

    # image segmentation
    img_labelled, num_features = label(img_extracted)
    print(f'Found {num_features} features for {i_img_extracted}')

    # save extracted picture
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    im = ax1.imshow(img_labelled, interpolation = 'nearest', cmap='gray')
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Labelled map '+str(i_img_extracted),fontsize = 30)
    fig.tight_layout()
    fig.savefig('png/labelled_map_'+str(i_img_extracted)+'.png')
    plt.close(fig)

    # give size of the objects
    objects = find_objects(img_labelled)
    for i in range(len(objects)):
        print('Feature', i+1, 'from', i_img_extracted, 'size (sum of array):', img_labelled[objects[i]].sum())

    # save img_labelled
    L_img_labelled.append(img_labelled)
    print()

#------------------------------------------------------------------------------------------------------------------------------------------ #
# CRN to DEM functions

def compute_vertices(dict_user, dict_sample):
    '''
    From a CRN map, compute vertices coordinates for polyhedral.
    '''
    L_L_vertices = []
    # iterate on the grains
    for i_grain in range(len(dict_sample['L_center'])):
        # compute vertices
        L_vertices = interpolate_vertices(dict_sample['L_crni_map'][i_grain], dict_sample['L_center'][i_grain], dict_user, dict_sample)
        L_L_vertices.append(L_vertices)

    # save data
    dict_save = {
    'L_L_vertices': L_L_vertices
    }
    with open('data/vertices.data', 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def interpolate_vertices(crn_i_map, center, dict_user, dict_sample):
    '''
    Interpolate vertices for polyhedral.
    '''
    # prepare the phi map
    map_phi = []
    L_phi = []
    for i_phi in range(dict_user['n_phi']):
        phi = 2*math.pi*i_phi/dict_user['n_phi']
        L_phi.append(phi)
        map_phi.append([])
    L_phi.append(2*math.pi)
    # iteration on x
    for i_x in range(0, len(dict_sample['x_L'])-1):
        # iteration on y
        for i_y in range(0, len(dict_sample['y_L'])-1):
            L_in = [] # list the nodes inside the grain
            if crn_i_map[-1-i_y    , i_x] > 0.5 :
                L_in.append(0)
            if crn_i_map[-1-(i_y+1), i_x] > 0.5 :
                L_in.append(1)
            if crn_i_map[-1-(i_y+1), i_x+1] > 0.5 :
                L_in.append(2)
            if crn_i_map[-1-i_y    , i_x+1] > 0.5 :
                L_in.append(3)
            if L_in != [] and L_in != [0,1,2,3]:
                center_mesh = (np.array([dict_sample['x_L'][i_x], dict_sample['y_L'][i_y]])+np.array([dict_sample['x_L'][i_x+1], dict_sample['y_L'][i_y+1]]))/2
                u = (center_mesh-np.array(center))/np.linalg.norm(center_mesh-np.array(center))
                # compute phi
                if u[1]>=0:
                    phi = math.acos(u[0])
                else :
                    phi = 2*math.pi-math.acos(u[0])
                # iterate on the lines of the mesh to find the plane intersection
                L_p = []
                if (0 in L_in and 1 not in L_in) or (0 not in L_in and 1 in L_in):# line 01
                    x_p = dict_sample['x_L'][i_x]
                    y_p = (0.5-crn_i_map[-1-i_y, i_x])/(crn_i_map[-1-(i_y+1), i_x]-crn_i_map[-1-i_y, i_x])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
                    L_p.append(np.array([x_p, y_p]))
                if (1 in L_in and 2 not in L_in) or (1 not in L_in and 2 in L_in):# line 12
                    x_p = (0.5-crn_i_map[-1-(i_y+1), i_x])/(crn_i_map[-1-(i_y+1), i_x+1]-crn_i_map[-1-(i_y+1), i_x])*(dict_sample['x_L'][i_x+1]-dict_sample['x_L'][i_x])+dict_sample['x_L'][i_x]
                    y_p = dict_sample['y_L'][i_y+1]
                    L_p.append(np.array([x_p, y_p]))
                if (2 in L_in and 3 not in L_in) or (2 not in L_in and 3 in L_in):# line 23
                    x_p = dict_sample['x_L'][i_x+1]
                    y_p = (0.5-crn_i_map[-1-i_y, i_x+1])/(crn_i_map[-1-(i_y+1), i_x+1]-crn_i_map[-1-i_y, i_x+1])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
                    L_p.append(np.array([x_p, y_p]))
                if (3 in L_in and 0 not in L_in) or (3 not in L_in and 0 in L_in):# line 30
                    x_p = (0.5-crn_i_map[-1-i_y, i_x])/(crn_i_map[-1-i_y, i_x+1]-crn_i_map[-1-i_y, i_x])*(dict_sample['x_L'][i_x+1]-dict_sample['x_L'][i_x])+dict_sample['x_L'][i_x]
                    y_p = dict_sample['y_L'][i_y]
                    L_p.append(np.array([x_p, y_p]))
                # compute the mean point
                p_mean = np.array([0,0])
                for p in L_p :
                    p_mean = p_mean + p
                p_mean = p_mean/len(L_p)
                # look phi in L_phi
                i_phi = 0
                while not (L_phi[i_phi] <= phi and phi <= L_phi[i_phi+1]) :
                    i_phi = i_phi + 1
                # save p_mean in the map
                map_phi[i_phi].append(p_mean)

    L_vertices = ()
    # interpolate vertices
    for i_phi in range(len(map_phi)):
        # mean vertices
        mean_v = np.array([0, 0])
        for v_i in map_phi[i_phi]:
            mean_v = mean_v + v_i/len(map_phi[i_phi])
        # save
        L_vertices = L_vertices + ((mean_v[0], mean_v[1], 0,),)
        L_vertices = L_vertices + ((mean_v[0], mean_v[1], 1),)

    return L_vertices

#------------------------------------------------------------------------------------------------------------------------------------------ #
# CRN to DEM step
# transform a map into vertices

# function definition 
def tuplet_to_list(tuplet):
    '''
    Convert a tuplet into lists.
    '''
    L_x = []
    L_y = []
    for v in tuplet:
        L_x.append(v[0])
        L_y.append(v[1])
    L_x.append(L_x[0])
    L_y.append(L_y[0])
    return L_x, L_y

# create dicts
# dicts are used to transmit information into the different functions
# every parameters should be in one dict
# dict user is related to parameter given by the user
# dict sample is related to computed parameter describing the sample

dict_user = {
    'n_phi': 60 # angular discretization of the grains
}
dict_sample = {
    'x_L': np.arange(0, L_img_extracted[0].shape[1]),  # x coordinates of the mesh (assumed regular and gridded)
    'y_L':np.arange(0, L_img_extracted[0].shape[0]), # y coordinates of the mesh (assumed regular and gridded)
    'L_crni_map': L_img_extracted_conv, # maps extacted from CRN simulations
    'L_center': [[200, 155], [400, 155]] # approximation of the centers
}

# compute vertices
compute_vertices(dict_user, dict_sample)

# compare the masses (in CRN and in DEM)
# load data 
with open('data/vertices.data', 'rb') as handle:
    dict_save = pickle.load(handle)

# iterate on the grains
for i_g in range(len(dict_sample['L_center'])):

    # convert tuplet into list
    L_x, L_y = tuplet_to_list(dict_save['L_L_vertices'][i_g])
    # modify the list of vertices
    L_x_adapted = []
    L_y_adapted = []
    for i in range(len(L_x)):
        if i%2 == 0:
            L_x_adapted.append(L_x[i])
            L_y_adapted.append(L_y[i])

    # compute center
    center = [np.mean(L_x_adapted), np.mean(L_y_adapted)]

    # compute surface (Heron formula)
    surf = 0
    for i_xy in range(len(L_x_adapted)-1):
        # coordinates of vertices of the triangle
        x_i = L_x_adapted[i_xy]
        x_j = L_x_adapted[i_xy+1]
        y_i = L_y_adapted[i_xy]
        y_j = L_y_adapted[i_xy+1]
        # compute lenght of sizes
        a = np.linalg.norm(np.array([x_i-x_j, y_i-y_j]))
        b = np.linalg.norm(np.array([x_j-center[0], y_j-center[1]]))
        c = np.linalg.norm(np.array([center[0]-x_i, center[1]-y_i]))
        # compute surface 
        surf_i = math.sqrt((a**2+b**2+c**2)**2-2*(a**4+b**4+c**4))/4
        surf = surf + surf_i
    # output
    print('Grain', i_g, 'area:', surf)

#------------------------------------------------------------------------------------------------------------------------------------------ #
# DEM visualization

# load data 
with open('data/vertices.data', 'rb') as handle:
    dict_save = pickle.load(handle)

# iterate on the grains
for i_g in range(len(dict_sample['L_center'])):
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))

    im = ax1.imshow(dict_sample['L_crni_map'][i_g], interpolation = 'nearest', cmap='gray', extent=[min(dict_sample['x_L']),max(dict_sample['x_L']),min(dict_sample['y_L']),max(dict_sample['y_L'])])
    L_x, L_y = tuplet_to_list(dict_save['L_L_vertices'][i_g])
    ax1.plot(L_x, L_y, color='r', linewidth=6)

    ax1.axis('equal')
    plt.suptitle('Grain '+str(i_g), fontsize=20)
    fig.tight_layout()
    fig.savefig('png/shape_'+str(i_g)+'.png')
    plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #
# DEM step
# the best is to call Yade. A contact law based on the volume is available. It is much better for irregular shapes
# a faster way is to use a contact law based on the distance. 

# add parameter in dict_user
dict_user['overlap_target'] = 100

# obtain the vertices 
# load data 
with open('data/vertices.data', 'rb') as handle:
    dict_save = pickle.load(handle)

L_L_x = []
L_L_y = []
# iterate on the grains
for i_g in range(len(dict_sample['L_center'])):

    # convert tuplet into list
    L_x, L_y = tuplet_to_list(dict_save['L_L_vertices'][i_g])

    # modify the list of vertices
    L_x_adapted = []
    L_y_adapted = []
    for i in range(len(L_x)):
        if i%2 == 0:
            L_x_adapted.append(L_x[i])
            L_y_adapted.append(L_y[i])
    
    # save vertices
    L_L_x.append(L_x_adapted)
    L_L_y.append(L_y_adapted)

# compute the current overlap
overlap = max(L_L_x[0])-min(L_L_x[1])

# compare the current overlap with the target
# > 0: grains need to be closer
# < 0: grains need to be further
disp = dict_user['overlap_target']-overlap

# translate into rigid body motion
L_rbm = [[disp/2, 0], [-disp/2, 0]]

#------------------------------------------------------------------------------------------------------------------------------------------ #
# DEM to CRN step
# Modify the CRN map with the rigid body motion computed in DEM

L_img_rbm = []
for i_img_extracted_conv in range(len(L_img_extracted_conv)):
    img_extracted_conv = L_img_extracted_conv[i_img_extracted_conv]
    img_rbm_tempo = img_extracted_conv.copy()
    img_rbm = img_extracted_conv.copy()

    # rigid body motion to apply 
    rbm = L_rbm[i_img_extracted_conv]
    # convert into mesh
    rbm_x_mesh = int(round(rbm[0],0))
    rbm_y_mesh = int(round(rbm[1],0))

    # apply the rigid body motion on x axis
    for i_x in range(len(dict_sample['x_L'])):
        # translation on the right
        if rbm_x_mesh >= 0:
            # in the previous domain
            if i_x-rbm_x_mesh >= 0:
                img_rbm_tempo[:, i_x] = img_extracted_conv[:, i_x-rbm_x_mesh]
            # out of the previous domain
            else :
                for i_y in range(len(dict_sample['y_L'])):
                    img_rbm_tempo[i_y, i_x] = 0
        # tanslation on the left
        if rbm_x_mesh < 0:
            # in the previous domain
            if i_x-rbm_x_mesh <= len(dict_sample['x_L'])-1:
                img_rbm_tempo[:, i_x] = img_extracted_conv[:, i_x-rbm_x_mesh]
            # out of the previous domain
            else :
                for i_y in range(len(dict_sample['y_L'])):
                    img_rbm_tempo[i_y, i_x] = 0

    # apply the rigid body motion on y axis
    # not used for the moment
    img_rbm = img_rbm_tempo.copy()

    # print
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9))
    im = ax1.imshow(img_extracted_conv, interpolation = 'nearest', cmap='gray')
    ax1.set_title('before')
    im = ax2.imshow(img_rbm, interpolation = 'nearest', cmap='gray')
    ax2.set_title('after')
    fig.suptitle('Apply RBM '+str(i_img_extracted_conv), fontsize=20)
    fig.tight_layout()
    fig.savefig('png/rbm_'+str(i_img_extracted_conv)+'.png')
    plt.close(fig)

    # save img_rbm
    L_img_rbm.append(img_rbm)
