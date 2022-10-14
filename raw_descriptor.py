import sys
import math
from tkinter import W
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv 
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

##1.浮点算法：Gray=R*0.3+G*0.59+B*0.11

##2.整数方法：Gray=(R*30+G*59+B*11)/100

##3.移位方法：Gray =(R*28+G*151+B*77)>>8

#implement circle 
#(0,0,0) ---> BGR representation
img = cv.imread('cones/im2.png',cv.IMREAD_COLOR)
print(sum((img[0][0][2]*0.297,img[0][0][1]*0.584,img[0][0][0]*0.144)))
img = cv.imread('cones/im2.png',cv.IMREAD_GRAYSCALE)
print((img[0][0]))


#best practices : bresenham algorithm to draw cricle but we use a simple bounding box way
radius = 5

indices = [[0 for c in range(0,radius*2+1)] for r in range(0,radius*2+1)]

#range: colse for left, open for right  math ------> [-radius,radius)
for i in range(-radius,radius+1):
    for j in range(-radius,radius+1):
        #print(i+radius,j+radius)
        indices[i+radius][j+radius] = (-i,-j)
print(indices)

print(img.shape)
rows,cols = img.shape

# check whether the point is inside the circle 
def is_inside_circle(delta_x, delta_y):
    if radius * radius >= delta_x * delta_x + delta_y * delta_y :
        return True 

minimum_boundary_point = 5

# find circle algorithm 1: midpoint
def find_circle_boundary(radius):
    boundary_points = set()
    for delta_y in range(0,radius):
        delta_x = math.sqrt(radius*radius - delta_y * delta_y)
        delta_x = math.floor(delta_x)
        # Todo: need to add index to mantain the consistency 
        # up 
        boundary_points.add((delta_x, delta_y)) 
        boundary_points.add((-delta_x, delta_y)) 
        # down 
        boundary_points.add((delta_x, -delta_y)) 
        boundary_points.add((-delta_x, -delta_y)) 
    print("boundary points side {0}".format(len(boundary_points)))
    # Todo : must bigger than minimun boundary points
    return boundary_points


1.#Set initial values of (xc, yc) and (x, y)
2.#Set decision parameter d to d = 3 – (2 * r). 
# 
3.#call drawCircle(int xc, int yc, int x, int y) function.
4.#Repeat steps 5 to 8 until x < = y
5.#Increment value of x.
6.#If d < 0, set d = d + (4*x) + 6
7.#Else, set d = d + 4 * (x – y) + 10 and decrement y by 1.
8.#call drawCircle(int xc, int yc, int x, int y) function

#if (d < 0) {
#d += 2 * x + 1; /* go East */
#} else {
#y--;
#d += 2 * (x – y) + 1; /* go SouthEast */

def init_decision(radius):
    d = 3 - 2*radius
    return d

# may use callback to evalute labels id 
def adding_boundary_point(center_point,x,y,boundary_points):
    xc,yc = center_point
    boundary_points.append([xc+x,yc+y,0])
    boundary_points.append([xc+y,yc+x,1])
    boundary_points.append([xc+y,yc-x,2])
    boundary_points.append([xc+x,yc-y,3])
    boundary_points.append([xc-x,yc-y,4])
    boundary_points.append([xc-y,yc-x,5])
    boundary_points.append([xc-y,yc+x,6])
    boundary_points.append([xc-x,yc+y,7])
    return boundary_points


# find circle_algorithm 2: bresenham' algorithm 
def find_circle_boundary2(center_point, radius):
    d = init_decision(radius)
    x = 0
    y = radius
    boundary_points = []
    adding_boundary_point(center_point,x,y,boundary_points)
    while x <= y:
        x = x + 1
        if (d<0):
            d = d + 4*x + 6  
        else:
            y = y -1
            d = d + (x-y)*4 + 10
        adding_boundary_point(center_point,x,y,boundary_points)
    return boundary_points

# return points list within point[x,y,group_number, labels to make sure consecutive]
def add_labels(boundary_points):
    count = 0
    for boundary_point in boundary_points:
        #print(boundary_point[2])
        if (boundary_point[2] == 0):
           count += 1  
    boundary_points_with_labels = []
    boundary_points.sort(key=lambda x:x[2])
    for i,boundary_point in enumerate(boundary_points):
        boundary_points_with_labels.append([boundary_point[0],boundary_point[1],boundary_point[2],i+(boundary_point[2]*count)])
    boundary_points_with_labels.sort(key=lambda x:x[3])
    return boundary_points_with_labels
     
def clamp(img,point):
    rows,cols,depth = img.shape
    # pixel based coordinates: y for rows and x for cols 
    x = point[0]
    x = max(0,x)
    x = min(cols-1,x)
    y = point[1]
    y = max(0,y)
    y = min(rows-1,y)
    return x,y 

# unit test:
def test_function_extraction(boundarypoints1, boundarypoints2):
    if boundarypoints1 == boundarypoints2:
        return True 
    return False

def dump_cricle(radius):
    #boundary_points = find_circle_boundary(10)
    point = [50,50]
    boundary_points = find_circle_boundary2(point,radius)
    print(boundary_points)
    img = np.zeros([100,100,3],dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    boundary_points_with_labels = add_labels(boundary_points)
    for boundary_point in boundary_points_with_labels:
        draw_point = [boundary_point[0],boundary_point[1]]
        col,row = clamp(img,draw_point)
        #img[row,col] = boundary_point[3]%255 
        # this is the proof: make sure labels are consecutive and we can observe this with human eyes 
        img[row,col] = 0 
        if boundary_point[3] ^ 1 == boundary_point[3] +1:
            img[row,col] = 255 
    plt.imshow(img), plt.show()


# using circle to slide image to search keypoints
#def find_keypoints(boundary_points):

# iterate the image array 
#boundary_points = set() 
#boundary_points = find_circle_boundary(radius)
#descriptors = []
#for i in range(0,rows):
    #for j in range(0,cols):
        #current_pixel_value = img[i][j]
        #pixels = []
        #for point in boundary_points:
            #point = [i+point[0], j+point[1]]
            #col,row = clamp(img,point)
            #pixels.append(img[row][col])
        #if (len(pixels) < 10):
            #continue
        #count = 0
        ## Todo: contiguous pixel point
        #for pixel in pixels:
            #if pixel > current_pixel_value:
                #count += 1
        #if count > len(pixels)/2 :
            #descriptors.append((i,j)) 

#print(img.shape[0]*img.shape[1])
#print(len(descriptors))

dump_cricle(30)
            
        
        
            
        


        #print(i,j)

#plt.imshow(img) , plt.show()



