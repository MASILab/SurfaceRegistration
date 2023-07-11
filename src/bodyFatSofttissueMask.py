"""
Body/Fat/Soft tissue mask 

Note: CT needs to be in orginial space. Normalization and windowing is not available

Yucheng Tang

Jan 2020

"""

from __future__ import print_function

# import torch.utils.data as data
import os
import random
import glob
from PIL import Image
import numpy as np
import nibabel as nb


from scipy import ndimage as ndi
import skimage.morphology
import skimage.measure


# --------------body mask------------------

##-----parameters-paths
image_dir = '/home-local/kimm58/SPIE2023/mask_in'
output_dir = '/home-local/kimm58/SPIE2023/mask_out'
##-----

def mask_CT(image_file, out_file, threshold_value=200):

    rBody=2
    count = 0

#for image in os.listdir(image_dir):
    #image_file = os.path.join(image_dir, image)
    image_nb = nb.load(image_file)
    image_np = np.array(image_nb.dataobj)
    # image_np[image_np==0] == -1024

    BODY = (image_np>=threshold_value)# & (I<=win_max) ### set this to be the intensity value that you wish to threshold at (kind of)
    print(' {} of {} voxels masked.'.format(np.sum(BODY),np.size(BODY)))
    if np.sum(BODY)==0:
        raise ValueError('BODY could not be extracted!')
    # Find largest connected component in 3D
    struct = np.ones((3,3,3),dtype=np.bool)
    BODY = ndi.binary_erosion(BODY,structure=struct,iterations=rBody)

    # if np.sum(BODY)==0:
    #     raise ValueError('BODY mask disappeared after erosion!')        
    BODY_labels = skimage.measure.label(np.asarray(BODY, dtype=np.int))

    props = skimage.measure.regionprops(BODY_labels)
    areas = []
    for prop in props: 
        areas.append(prop.area)
    print('  -> {} areas found.'.format(len(areas)))
    # only keep largest, dilate again and fill holes                
    BODY = ndi.binary_dilation(BODY_labels==(np.argmax(areas)+1),structure=struct,iterations=rBody)
    # Fill holes slice-wise
    for z in range(0,BODY.shape[2]):    
        BODY[:,:,z] = ndi.binary_fill_holes(BODY[:,:,z])  

    #new_image = nb.Nifti1Image(BODY.astype(np.int8), image_nb.affine)
    new_image = nb.Nifti1Image(BODY.astype(np.int16), image_nb.affine)
    #out_file = os.path.join(output_dir, image)
    #out_file = os.path.join(output_dir, "seg.nii.gz")
    nb.save(new_image,out_file)
    count += 1
    print('[{}] Generated body_mask segs in Abwall {}'.format(count, image_file))


# #------------------------------------------------------fat
# ##-----parameters-paths
# image_dir = ''
# body_dir = ''
# output_dir = ''
# ##-----

# import skfuzzy as fuzz
# import math

# def weighted_avg_and_std(values, weights):
#     """
#     Return the weighted average and standard deviation.

#     values, weights -- Numpy ndarrays with the same shape.
#     """
#     average = np.average(values, weights=weights)
#     # Fast and numerically precise:
#     variance = np.average((values-average)**2, weights=weights)
#     return (average, math.sqrt(variance))

# output_st_dir = os.path.join(output_dir, 'softTissue')
# if not os.path.isdir(output_st_dir):
#     os.makedirs(output_st_dir)

# output_fat_dir = os.path.join(output_dir, 'fat')
# if not os.path.isdir(output_fat_dir):
#     os.makedirs(output_fat_dir)

# count = 0
# for image in os.listdir(image_dir):
#     image_file = os.path.join(image_dir, image)
#     imgnb = nb.load(image_file)
#     imgnp = np.array(imgnb.dataobj)

#     body_file = os.path.join(body_dir, image)
#     bodynb = nb.load(body_file)
#     bodynp = np.array(bodynb.dataobj)

#     # modify body mask
#     idx = np.where(imgnp <= -500)
#     bodynp[idx] = 0

#     # fat mask the same as body mask
#     fat = np.ones((imgnp.shape[0], imgnp.shape[1], imgnp.shape[2]))
#     idx = np.where(bodynp == 0)
#     fat[idx] = 0

#     # set ROI for new image
#     new_img = imgnp[bodynp > 0]
#     new_img = np.reshape(new_img, (-1, 1))
#     new_img = np.transpose(new_img)
    
#     # first fuzzy c means
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(new_img, 2, 2, error=0.005, maxiter=1000, init=None)
#     # flip to make sure 1 -> muscle, 2 -> fat
#     if cntr[0] < cntr[1]:
#         cntr[0][0], cntr[1][0] = cntr[1][0], cntr[0][0]
#         tmp = np.copy(u[1])
#         u[1] = u[0]
#         u[0] = tmp

#     # weighted standard divation along 1 and 2 axis
#     avg1, sgm1 = weighted_avg_and_std(new_img, np.reshape(u[0], (1,-1)))
#     avg2, sgm2 = weighted_avg_and_std(new_img, np.reshape(u[1], (1,-1)))

#     # bone indices
#     new_img = imgnp[bodynp > 0]

#     idx_bone = np.where(imgnp > cntr[0][0] + 2 * sgm1)
#     fat[idx_bone] = 0
#     # air indices
#     idx_air = np.where(imgnp < cntr[1][0] - 2 * sgm2)
#     fat[idx_air] = 0

#     # second FCM
#     new_img = imgnp[fat > 0]
#     new_img = np.reshape(new_img, (-1, 1))
#     new_img = np.transpose(new_img)
    
#     # first fuzzy c means
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(new_img, 2, 2, error=0.005, maxiter=1000, init=None)
#     # flip to make sure 1 -> muscle, 2 -> fat
#     if cntr[0] < cntr[1]:
#         cntr[0][0], cntr[1][0] = cntr[1][0], cntr[0][0]
#         tmp = np.copy(u[1])
#         u[1] = u[0]
#         u[0] = tmp
#     # asign fat with cmeans 2 component
#     idx = np.where(fat > 0)
#     fat[idx] = u[1]
#     # split soft tissue and fat
#     softTissue = fat < 0.5
#     softTissue = softTissue.astype(np.int8)
#     idx = np.where(bodynp == 0)
#     softTissue[idx] = 0
#     softTissue[idx_bone] = 0
#     softTissue[idx_air] = 0

#     fatTissue = fat >= 0.5
#     fatTissue = fatTissue.astype(np.int8)

#     ST_nb = nb.Nifti1Image(softTissue, imgnb.affine)
#     ST_file = os.path.join(output_st_dir, image)
#     nb.save(ST_nb, ST_file)

#     F_nb = nb.Nifti1Image(fatTissue, imgnb.affine)
#     F_file = os.path.join(output_fat_dir, image)
#     nb.save(F_nb, F_file)
#     count += 1
#     print('[{}] soft tissue and fat generated {}'.format(count, image))





