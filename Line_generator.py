import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw_random_lines(img, w, n,pt_on_line_noise_prob=60,bacground_noise_prob = 75):
    line_masks = []
    for i in range(n):
        point1 = (np.random.randint(low = 0, high = w), np.random.randint(low = 0, high = w))
        point2 = (np.random.randint(low = 0, high = w), np.random.randint(low = 0, high = w))
        mask = np.zeros((w,w), np.uint8)
        cv2.line(img, point1, point2, (255, 0, 0), 5)
        line_masks.append(cv2.line(mask, point1, point2, 255))

    ######################################
    #  Generate binary mask
    ######################################
    mask_gt = np.zeros((w,w), np.uint8)
    for mask in line_masks:
        mask_gt = np.maximum(mask_gt,mask)

    kernel = np.ones((5, 5), 'uint8')
    mask_gt = cv2.dilate(mask_gt, kernel, iterations=1)
    ######################################
    #  Randomly noise the background
    ######################################
    x = y = 0
    while(y<w):
        while(x<w):
            if(np.any(img[x, y] != 0)):
                if(np.random.randint(low=0, high=100) < pt_on_line_noise_prob):
                    img[x, y] = [255, 255, 255]
                else:
                    img[x, y] = [0, 0, 0]
            else:
                if(np.random.randint(low=0, high=100) < bacground_noise_prob):
                    img[x, y] = [255, 255, 255]
                else:
                    img[x, y] = [0, 0, 0]
            x+=1
        x=0
        y+=1
    return img,mask_gt

num_of_images = 1000

for i in range(0,num_of_images):
    print("generating image {}::".format(i))
    w = 512
    img = np.zeros((w,w,3), np.uint8)
    img,mask_gt = draw_random_lines(img, w,np.random.randint(low = 1, high = 5), np.random.randint(low = 70, high = 80),np.random.randint(low = 80, high = 95))
    img = 255-img
    #cv2.imshow("Original", img)
    cv2.imwrite("Data/input/{}.png".format(i), img)
    cv2.imwrite("Data/GT/{}_gt.png".format(i), mask_gt)
    #img = cv2.imread("alo.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for line in lines:
#     for rho,theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
#
# cv2.imshow("Detectada", img)

cv2.waitKey(0)