from xml.dom.minidom import DOMImplementation
#import numpy as np
import math
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import skimage
image_array = []
image_array.append( 'cones/im2.png')
image_array.append( 'cones/im6.png')
img_keypoints = []

img_decriptors = []
raw_image = []
raw_keypoints = []

for image_id in range(0,len(image_array)):
    img = cv.imread(image_array[image_id],0)
    raw_image.append(img)
    dimensions = img.shape 
    # initialize keypoints
    keypoints = [[set() for c in range(0,dimensions[0],5)] for r in range(0,dimensions[1],5)]
    #img = cv.bilateralFilter(img,100,75,75)
    #rows, cols, _channels = map(int, img.shape)
    #img= cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
    # Initiate ORB dekector
    orb = cv.ORB_create(100)
    # find the keypoints with ORB
    kps = orb.detect(img,None)
    # compute the descriptors with ORB
    kps, des = orb.compute(img, kps)
    raw_keypoints.append(kps)
    img_decriptors.append(des)

    for kp in kps:
        x = kp.pt[0]
        y = kp.pt[1]
        keypoints[math.ceil(x/5)][math.ceil(y/5)].add(kp)
    
    img_keypoints.append(keypoints)

    #for i in range(0,len(kp)):
        #print(kp[i].pt)
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
    #d_image = Image.open('orb_extractor_test.jpeg')
    #d_image.load()
    #d_image2= np.asarray(img, dtype='int32')
    #downsample = 20
    ## first, change to 0-1
    #ds_array = d_image2/255
    #r = skimage.measure.block_reduce(ds_array[:, :, 0],
                                     #(downsample, downsample),
                                     #np.mean)
    #g = skimage.measure.block_reduce(ds_array[:, :, 1],
                                     #(downsample, downsample),
                                     #np.mean)
    #b = skimage.measure.block_reduce(ds_array[:, :, 2],
                                     #(downsample, downsample),
                                     #np.mean)
    #ds_array = np.stack((r, g, b), axis=-1)
    #plt.imshow(img2), plt.show()

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck= True)

matches = bf.match(img_decriptors[0],img_decriptors[1])

matches = sorted(matches, key= lambda x:x.distance)

img_result = cv.drawMatches(raw_image[0],raw_keypoints[0],raw_image[1],raw_keypoints[1],matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_result),plt.show()


