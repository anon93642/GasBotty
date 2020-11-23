# -*- coding: utf-8 -*-
"""
Author: Anonymous Authors
"""

import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
import glob
from PIL import Image, ImageDraw
import itertools

# import keras
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def load_models():
    
    sign_model = torch.load('./weights/weights_101.pt', map_location=torch.device('cpu'))
    sign_model.eval()
    sign_model.cpu()
     
    #----------------------------------------------------------------------------------------price
    price_model_path = './weights/priceresnet50_csv_1all.h5'
    price_model = models.load_model(price_model_path, backbone_name='resnet50')
    price_model = models.convert_model(price_model)
    # load label to names mapping for visualization purposes
    price_labels_to_names = pd.read_csv('./weights/price_mapping.csv',header=None).T.loc[0].to_dict()
       
    #-----------------------------------------------------------------------------------------digits
    digit_model_path = './weights/resnet101_DIGITSFINAL.h5'
    digit_model = models.load_model(digit_model_path, backbone_name='resnet101')
    digit_model = models.convert_model(digit_model)
    digit_labels_to_names = pd.read_csv('./weights/digits_mapping.csv',header=None).T.loc[0].to_dict()
      
    #-----------------------------------------------------------------------------------------labels
    label_model_path = './weights/resnet101_LABELSG1.h5'
    label_model = models.load_model(label_model_path, backbone_name='resnet101')
    label_model = models.convert_model(label_model)
    label_labels_to_names = pd.read_csv('./weights/labels_mapping.csv',header=None).T.loc[0].to_dict()
     
    return sign_model, price_model, digit_model, label_model 

def get_mask(img, model):
    
    img = img.transpose(2,0,1).reshape(1,3,640,640)
    
    with torch.no_grad():
        res = model(torch.from_numpy(img).type(torch.FloatTensor)/255)        
    result = res['out'].cpu().detach().numpy()[0][0]>0.15
    mask = result.astype(np.uint8)*255
            
    mask_copy = cv2.cvtColor(mask.copy(),cv2.COLOR_GRAY2RGB)    
    mask_copy[np.where((mask_copy == [255,255,255]).all(axis=2))] = ((5,188,251))
    mask = cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=[0])
    return mask

def get_border(mask): 
    
    blurred = cv2.GaussianBlur(mask, (3, 3), 1)
    auto = auto_canny(blurred)
      
    _, contours, hierarchy = cv2.findContours(auto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    config = 0
    counter = 0
    while counter < len(contours):
        if len(contours[counter]) > config:
            config = counter
        
        counter = counter + 1
    contour = contours[config]
    
    auto2 = np.zeros(mask.shape)
    auto2 = cv2.drawContours(auto2, contour, -1, 255, cv2.FILLED).astype(np.uint8)
    
    return auto2



def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def get_lines(auto2, img1):
     all_lines = []
     lines = cv2.HoughLines(auto2,1,np.pi/180,37)
     if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    
                x1 = int(pt1[0])
                y1 = int(pt1[1])
                x2 = int(pt2[0])
                y2 = int(pt2[1])
                    
                if x1 - x2 + y1 - y2 > 1000:
                        
                    value = (x1, y1, x2, y2)
                else:
                    value = (x2, y2, x1, y1)
            
                    
                all_lines.append(value)
            
                cv2.line(img1, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
                    #cv2.line(auto, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            column_name = ['x1', 'y1', 'x2', 'y2']
            df = pd.DataFrame(all_lines, columns=column_name)
            
            return df, img1

def get_intersections(df, img1):
    pnts = []
    cnt = 0
    while cnt  < len(df):
                    
        #print(cnt)
        dx1 = df.iloc[cnt][2] - df.iloc[cnt][0]
        dy1 = df.iloc[cnt][3] - df.iloc[cnt][1]
                    
        mag1 = np.sqrt(dx1**2 + dy1**2)
                    
        m1 = dy1/(dx1+0.01)
                    
        b1 = df.iloc[cnt][3] - df.iloc[cnt][2]*m1
                    
        cnt1 = cnt + 1
                    
        while cnt1 < len(df):
            dx2 = df.iloc[cnt1][2] - df.iloc[cnt1][0]
            dy2 = df.iloc[cnt1][3] - df.iloc[cnt1][1]
                        
            mag2 = np.sqrt(dx2**2 + dy2**2)
                        
            dot_product = np.abs(dx1*dx2 + dy1*dy2)
                        
            if dot_product/(mag1*mag2) > 1:
                pheta = np.degrees(np.arccos(1))
            else:
                pheta = np.degrees(np.arccos(dot_product/(mag1*mag2)))
                      
            if pheta > 30:
                    #print('Do stuff')
                            
                m2 = dy2/(dx2+0.01)
                    
                b2 = df.iloc[cnt1][3] - df.iloc[cnt1][2]*m2
                            
                x = (b2 - b1)/(m1-m2)
                y = m1*x + b1
                            
                if x >= 0 and x <= 642 and y >= 0 and y <= 642:
                    val = (int(np.around(x)), np.around(int(y)))
                                
                    pnts.append(val)
                                
                    cv2.circle(img1, (int(np.around(x)),np.around(int(y))), radius=2, color=(0, 0, 0), thickness=-1)
                        
            cnt1 = cnt1 + 1
                    
                    
        cnt = cnt + 1
    col_name = ['x', 'y']
    df_pnts = pd.DataFrame(pnts, columns=col_name)
    
    
    return df_pnts, img1

def get_corners(df_pnts, img1):
    kmeans = KMeans(n_clusters=4, random_state=3).fit(df_pnts)
    kmeans.labels_
                
    df_pnts['Labels'] = kmeans.labels_
                
    x_zero = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 0]['x'])))
    y_zero = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 0]['y'])))
                
    x_one = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 1]['x'])))
    y_one = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 1]['y'])))
                
    x_two = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 2]['x'])))
    y_two = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 2]['y'])))
                
    x_three = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 3]['x'])))
    y_three = int(np.around(np.average(df_pnts.loc[df_pnts['Labels'] == 3]['y'])))
            
    points = np.array([[x_zero, y_zero], [x_one, y_one], [x_two, y_two], [x_three, y_three]])
    #ordered_pnts = order_points(points)
    ordered_pnts = order_points_old(points)
    ordered_pnts = ordered_pnts.astype(int)

    #cv2.imwrite('./goodlooking/' + str(total_cnt) + '/d.png' , img1)
                
    cv2.circle(img1, (ordered_pnts[0][0], ordered_pnts[0][1]), radius=7, color=(83, 168, 52), thickness=-1) # (83, 168, 52)
                
    cv2.circle(img1, (ordered_pnts[1][0], ordered_pnts[1][1]), radius=7, color=(83, 168, 52), thickness=-1)
                
    cv2.circle(img1, (ordered_pnts[2][0], ordered_pnts[2][1]), radius=7, color=(83, 168, 52), thickness=-1)
                
    cv2.circle(img1, (ordered_pnts[3][0], ordered_pnts[3][1]), radius=7, color=(83, 168, 52), thickness=-1)
                
    src_pts = np.array([[ordered_pnts[0][0], ordered_pnts[0][1]], [ordered_pnts[1][0], ordered_pnts[1][1]], [ordered_pnts[2][0], ordered_pnts[2][1]], [ordered_pnts[3][0], ordered_pnts[3][1]]], dtype=np.float32)
               
    return src_pts, img1


def get_euler_distance(pt1, pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5



def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def get_KS(src_pts, img2):
                
    width = int(get_euler_distance(src_pts[0], src_pts[1]))
    height = int(get_euler_distance(src_pts[0], src_pts[3]))
                
    dst_pts = np.array([[0, 0],   [width, 0],  [width, height], [0, height]], dtype=np.float32)
                
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img2, M, (width, height))
    
    return warp

def price_level(sign, price_model):
    def calculateIntersection(a0, a1, b0, b1):
        if a0 >= b0 and a1 <= b1: # Contained
            intersection = a1 - a0
        elif a0 < b0 and a1 > b1: # Contains
            intersection = b1 - b0
        elif a0 < b0 and a1 > b0: # Intersects right
            intersection = a1 - b0
        elif a1 > b1 and a0 < b1: # Intersects left
            intersection = b1 - a0
        else: # No intersection (either side)
            intersection = 0

        return intersection
    
    
    
    THRES_SCORE = 0.8
    model = price_model
    draw = cv2.cvtColor(sign.copy(), cv2.COLOR_BGR2RGB)
    image = preprocess_image(sign.copy())
    image, scale = resize_image(image)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    boxes /= scale
    data = []
    
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break
    
        data.append((box, score, label))
    


  # Intersecting rectangles:
    intersecting = []

    cnt2 = 0

    while cnt2 < len(data):

        x1 = int(np.rint(data[cnt2][0][0]))
        y1 = int(np.rint(data[cnt2][0][1]))
        x2 = int(np.rint(data[cnt2][0][2]))
        y2 = int(np.rint(data[cnt2][0][3]))

        a = (int(np.rint(data[cnt2][0][0])),
          int(np.rint(data[cnt2][0][1])),
          int(np.rint(data[cnt2][0][2])),
          int(np.rint(data[cnt2][0][3])))
      
      
      # The rectangle against which you are going to test the rest and its area:
        X0, Y0, X1, Y1 = a
        AREA = float((X1 - X0) * (Y1 - Y0))
      
      # Rectangles to check
        rectangles = []
      
        cnt3 = 0
        while cnt3 < len(data):
            rectangles.append((int(np.rint(data[cnt3][0][0])),
            int(np.rint(data[cnt3][0][1])),
            int(np.rint(data[cnt3][0][2])),
            int(np.rint(data[cnt3][0][3]))))
            cnt3 = cnt3 + 1
             
        cnt4 = 0
      
        while cnt4 < len(rectangles):
          
            x0, y0, x1, y1 = rectangles[cnt4]       
            width = calculateIntersection(x0, x1, X0, X1)        
            height = calculateIntersection(y0, y1, Y0, Y1)        
            area = width * height
            percent = area / AREA
         
            if percent > 0.25 and cnt2 != cnt4:
              #print(percent)
              intersecting.append([cnt2, cnt4])
              
            cnt4 = cnt4 + 1
        cnt2 = cnt2 + 1
    cnt5 = 0
    while cnt5 < len(intersecting):
        if intersecting[cnt5][0] > intersecting[cnt5][1]:
            w = intersecting[cnt5][0]
            intersecting[cnt5][0] = intersecting[cnt5][1]
            intersecting[cnt5][1] = w
        cnt5 = cnt5 + 1
      
    delete = []
    cnt6 = 0
    while cnt6 < len(intersecting):
        in1 = intersecting[cnt6][0]
        in2 = intersecting[cnt6][1]
      

        if data[in1][1] < data[in2][1]:
            delete.append(in1)
        else:
            delete.append(in2)
      
        cnt6 = cnt6 + 1

    result = []
    cnt7 = 0
    while cnt7 < len(data):
        if cnt7 not in delete:
          #print(cnt7)
         result.append([int(np.rint(data[cnt7][0][0])),
                        int(np.rint(data[cnt7][0][1])),
                        int(np.rint(data[cnt7][0][2])),
                        int(np.rint(data[cnt7][0][3])),
                        int(data[cnt7][2])  
              
              ])
      
        cnt7 = cnt7 + 1
    cnt8 = 0
    while cnt8 < len(result):
      
        x1 = result[cnt8][0]
        y1 = result[cnt8][1]
        x2 = result[cnt8][2]
        y2 = result[cnt8][3]

        if x1 >= image.shape[1]:
            x1 = image.shape[1] - 1
        if y1 >= image.shape[0]:
            y1 = image.shape[0] - 1
        if x2 >= image.shape[1]:
            x2 = image.shape[1] - 1
        if y2 >= image.shape[0]:
            y2 = image.shape[0] - 1
      

        label = result[cnt8][4]
      
 
      
        cv2.rectangle(sign, (x1, y1), (x2, y2), (5,188,251), thickness = 2)
        cnt8 = cnt8 + 1
    
    #cv2.imwrite('./goodlooking/' + str(total_cnt) + '/price.png' , sign.copy())
    #resized_sign = cv2.resize(sign.copy(),(0,0), fx=2, fy=2) 
    
    # cv2.imshow("Edges", resized_sign)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return result, sign

def label_level(sign, price_model, sign1):
    def calculateIntersection(a0, a1, b0, b1):
        if a0 >= b0 and a1 <= b1: # Contained
            intersection = a1 - a0
        elif a0 < b0 and a1 > b1: # Contains
            intersection = b1 - b0
        elif a0 < b0 and a1 > b0: # Intersects right
            intersection = a1 - b0
        elif a1 > b1 and a0 < b1: # Intersects left
            intersection = b1 - a0
        else: # No intersection (either side)
            intersection = 0

        return intersection
    
    
    
    THRES_SCORE = 0.5
    model = price_model
    draw = cv2.cvtColor(sign.copy(), cv2.COLOR_BGR2RGB)
    image = preprocess_image(sign.copy())
    image, scale = resize_image(image)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    boxes /= scale
    data = []
    
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break
    
        data.append((box, score, label))
    


  # Intersecting rectangles:
    intersecting = []

    cnt2 = 0

    while cnt2 < len(data):

        x1 = int(np.rint(data[cnt2][0][0]))
        y1 = int(np.rint(data[cnt2][0][1]))
        x2 = int(np.rint(data[cnt2][0][2]))
        y2 = int(np.rint(data[cnt2][0][3]))

        a = (int(np.rint(data[cnt2][0][0])),
          int(np.rint(data[cnt2][0][1])),
          int(np.rint(data[cnt2][0][2])),
          int(np.rint(data[cnt2][0][3])))
      
      
      # The rectangle against which you are going to test the rest and its area:
        X0, Y0, X1, Y1 = a
        AREA = float((X1 - X0) * (Y1 - Y0))
      
      # Rectangles to check
        rectangles = []
      
        cnt3 = 0
        while cnt3 < len(data):
            rectangles.append((int(np.rint(data[cnt3][0][0])),
            int(np.rint(data[cnt3][0][1])),
            int(np.rint(data[cnt3][0][2])),
            int(np.rint(data[cnt3][0][3]))))
            cnt3 = cnt3 + 1
             
        cnt4 = 0
      
        while cnt4 < len(rectangles):
          
            x0, y0, x1, y1 = rectangles[cnt4]       
            width = calculateIntersection(x0, x1, X0, X1)        
            height = calculateIntersection(y0, y1, Y0, Y1)        
            area = width * height
            percent = area / AREA
         
            if percent > 0.25 and cnt2 != cnt4:
              #print(percent)
              intersecting.append([cnt2, cnt4])
              
            cnt4 = cnt4 + 1
        cnt2 = cnt2 + 1
    cnt5 = 0
    while cnt5 < len(intersecting):
        if intersecting[cnt5][0] > intersecting[cnt5][1]:
            w = intersecting[cnt5][0]
            intersecting[cnt5][0] = intersecting[cnt5][1]
            intersecting[cnt5][1] = w
        cnt5 = cnt5 + 1
      
    delete = []
    cnt6 = 0
    while cnt6 < len(intersecting):
        in1 = intersecting[cnt6][0]
        in2 = intersecting[cnt6][1]
      

        if data[in1][1] < data[in2][1]:
            delete.append(in1)
        else:
            delete.append(in2)
      
        cnt6 = cnt6 + 1

    result = []
    cnt7 = 0
    while cnt7 < len(data):
        if cnt7 not in delete:
          #print(cnt7)
         result.append([int(np.rint(data[cnt7][0][0])),
                        int(np.rint(data[cnt7][0][1])),
                        int(np.rint(data[cnt7][0][2])),
                        int(np.rint(data[cnt7][0][3])),
                        int(data[cnt7][2])  
              
              ])
      
        cnt7 = cnt7 + 1
    cnt8 = 0
    while cnt8 < len(result):
      
        x1 = result[cnt8][0]
        y1 = result[cnt8][1]
        x2 = result[cnt8][2]
        y2 = result[cnt8][3]

        if x1 >= image.shape[1]:
            x1 = image.shape[1] - 1
        if y1 >= image.shape[0]:
            y1 = image.shape[0] - 1
        if x2 >= image.shape[1]:
            x2 = image.shape[1] - 1
        if y2 >= image.shape[0]:
            y2 = image.shape[0] - 1
      

        label = result[cnt8][4]
      
 
      
        cv2.rectangle(sign1, (x1, y1), (x2, y2), (244, 133, 66), thickness = 2)
        cnt8 = cnt8 + 1
    
    #cv2.imwrite('./goodlooking/' + str(total_cnt) + '/label.png' , sign.copy())
    # resized_sign = cv2.resize(sign1.copy(),(0,0), fx=2, fy=2) 
    
    # cv2.imshow("Edges", resized_sign)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return result, sign1

def read_digits(image, model):
  THRES_SCORE = 0.47
  image_copy = image.copy()

  image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

  # preprocess image for network
  image_copy = preprocess_image(image_copy)
  image_copy, scale = resize_image(image_copy)

  # process image
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image_copy, axis=0))
  #print("processing time: ", time.time() - start)

  # correct for image scale
  boxes /= scale
  data = []

  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < THRES_SCORE:
          break
      #print(box)
      data.append((box, score, label))
      #b = box.astype(int)
      #draw_box(image_copy, b, color=(255,100,0))

      #caption = "{} {:.3f}".format(labels_to_names[label], score)
      #draw_caption(draw, b, caption)
      #print(labels_to_names[label])


  
  def calculateIntersection(a0, a1, b0, b1):
      if a0 >= b0 and a1 <= b1: # Contained
          intersection = a1 - a0
      elif a0 < b0 and a1 > b1: # Contains
          intersection = b1 - b0
      elif a0 < b0 and a1 > b0: # Intersects right
          intersection = a1 - b0
      elif a1 > b1 and a0 < b1: # Intersects left
          intersection = b1 - a0
      else: # No intersection (either side)
          intersection = 0

      return intersection

  # Intersecting rectangles:
  intersecting = []

  cnt2 = 0

  while cnt2 < len(data):

      x1 = int(np.rint(data[cnt2][0][0]))
      y1 = int(np.rint(data[cnt2][0][1]))
      x2 = int(np.rint(data[cnt2][0][2]))
      y2 = int(np.rint(data[cnt2][0][3]))

      a = (int(np.rint(data[cnt2][0][0])),
          int(np.rint(data[cnt2][0][1])),
          int(np.rint(data[cnt2][0][2])),
          int(np.rint(data[cnt2][0][3])))
      
      
      # The rectangle against which you are going to test the rest and its area:
      X0, Y0, X1, Y1 = a
      AREA = float((X1 - X0) * (Y1 - Y0))
      
      # Rectangles to check
      rectangles = []
      
      cnt3 = 0
      while cnt3 < len(data):
          rectangles.append((int(np.rint(data[cnt3][0][0])),
          int(np.rint(data[cnt3][0][1])),
          int(np.rint(data[cnt3][0][2])),
          int(np.rint(data[cnt3][0][3]))))
          cnt3 = cnt3 + 1
             
      cnt4 = 0
      
      while cnt4 < len(rectangles):
          
          x0, y0, x1, y1 = rectangles[cnt4]       
          width = calculateIntersection(x0, x1, X0, X1)        
          height = calculateIntersection(y0, y1, Y0, Y1)        
          area = width * height
          percent = area / AREA
         
          if percent > 0.25 and cnt2 != cnt4:
              #print(percent)
              intersecting.append([cnt2, cnt4])
              
          cnt4 = cnt4 + 1
      cnt2 = cnt2 + 1
      
  cnt5 = 0
  while cnt5 < len(intersecting):
      if intersecting[cnt5][0] > intersecting[cnt5][1]:
          w = intersecting[cnt5][0]
          intersecting[cnt5][0] = intersecting[cnt5][1]
          intersecting[cnt5][1] = w
      cnt5 = cnt5 + 1

  delete = []
  cnt6 = 0
  while cnt6 < len(intersecting):
      in1 = intersecting[cnt6][0]
      in2 = intersecting[cnt6][1]
      
      if data[in1][2] == 10:
          delete.append(in2)
      elif data[in2][2] == 10:
          delete.append(in1)
      elif data[in1][1] < data[in2][1]:
          delete.append(in1)
      else:
          delete.append(in2)
      
      cnt6 = cnt6 + 1

  result = []
  cnt7 = 0
  while cnt7 < len(data):
      if cnt7 not in delete:
          #print(cnt7)
          result.append([int(np.rint(data[cnt7][0][0])),
                        int(np.rint(data[cnt7][0][1])),
                        int(np.rint(data[cnt7][0][2])),
                        int(np.rint(data[cnt7][0][3])),
                        int(data[cnt7][2])  
              
              ])
      
      cnt7 = cnt7 + 1
    

  result.sort()

  if result[-1][4] == 9:

    if np.abs(result[-1][3] - result[-1][1]) < 0.7*np.abs(result[-2][3] - result[-2][1]):
      result[-1][4] = 10

  image2 = image.copy()
  cnt8 = 0

  price = 0.0

  while cnt8 < len(result):
      
      x1 = result[cnt8][0]
      y1 = result[cnt8][1]
      x2 = result[cnt8][2]
      y2 = result[cnt8][3]

      if x1 >= image.shape[1]:
        x1 = image.shape[1] - 1
      if y1 >= image.shape[0]:
        y1 = image.shape[0] - 1
      if x2 >= image.shape[1]:
        x2 = image.shape[1] - 1
      if y2 >= image.shape[0]:
        y2 = image.shape[0] - 1
      

      label = result[cnt8][4]

      #print(label)

      if result[-1][4] == 10:
        if cnt8 == len(result) - 1:
          price = price + 0.009
          price = np.around(price, 3)

        else:
          price = price + 10**(-cnt8)*int(label)
          price = np.around(price, 3)
        
      if result[-1][4] != 10:
        price = price + 10**(-cnt8 )*int(label)
        price = np.around(price, 3)
      #print(cnt8)
      cv2.rectangle(image2, (x1, y1), (x2, y2), (52,67,234), thickness = 2)
      
      # if label != 10:
      #     cv2.putText(image2, str(int(label)), (x1,y2-1),1,  cv2.FONT_HERSHEY_PLAIN, (0, 255, 0) ,  1,  cv2.LINE_AA)
      # else:
      #     cv2.putText(image2, 'F', (x1,y2 -1),1,  cv2.FONT_HERSHEY_PLAIN, (0, 255, 0) ,  1,  cv2.LINE_AA)
      cnt8 = cnt8 + 1
  
  price = np.around(price, 3)

  #print(price)
  #cv2.imwrite('./goodlooking/' + str(total_cnt) + '/' + str(np.random.randint(10, size = 3)) +'price.png' ,image2.copy())
  # resized_sign = cv2.resize(image2.copy(),(0,0), fx=3, fy=3)
  # cv2.imshow("Edges", resized_sign)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()


  return price , image2



def associate(prices, where_labels, sign):
    
    label_labels_to_names = pd.read_csv('./weights/labels_mapping.csv',header=None).T.loc[0].to_dict()
    
    pay_type = []
    p = []
    
    for price in prices:
        p.append((price[4], (price[0] + price[2])/2, (price[1] + price[3])/2))
        
    l = []
    
    for label in where_labels:
        if label[4] < 4:
            l.append((label[4], (label[0] + label[2])/2, (label[1] + label[3])/2))
        if label[4] >= 4:
            pay_type.append((label[4], (label[0] + label[2])/2, (label[1] + label[3])/2))
    
    if len(p) > len(l):
        l.extend(l)
    
    
    while len(p) < len(l):
        p.append((1000, 1000, 1000))
        
    while len(l) < len(p):
        l.append((1000, 1000, 1000))
        
    
    all_poss_label = (list(itertools.permutations(l)))
        
    counter = 0
    cost = 1000000
    
    while counter < len(all_poss_label):
        
        new_cost = 0
        sec_counter = 0
        while sec_counter < len(p):
            if p[sec_counter][0] != 1000 and all_poss_label[counter][sec_counter][0] != 1000:
                curr_cost = np.sqrt((p[sec_counter][1] - all_poss_label[counter][sec_counter][1])**2 + 3*(p[sec_counter][2] - all_poss_label[counter][sec_counter][2])**2 + ((p[sec_counter][2] - all_poss_label[counter][sec_counter][2])**2)*np.heaviside( -1*(p[sec_counter][2] - all_poss_label[counter][sec_counter][2]), 1))
                new_cost = new_cost + curr_cost
            sec_counter = sec_counter + 1
        #print(new_cost)    
        if new_cost < cost:
            cost = new_cost
            
            form = counter
        
        counter = counter + 1
        
        
        
    #print(form, cost)    
        
    cnt2 = 0
    pay_value = []
    while cnt2 < len(p):
        if len(pay_type) == 0:
            pay_value.append('Both')
        else:
            cost = 1000000
            cnt3 = 0
            while cnt3 < len(pay_type):
                curr_cost = np.sqrt((p[cnt2][1] - pay_type[cnt3][1])**2 + (p[cnt2][2] - pay_type[cnt3][2])**2 + ((p[cnt2][2] - pay_type[cnt3][2])**2)*np.heaviside( (p[cnt2][2] - pay_type[cnt3][2]), 1))
                
                if curr_cost < cost:
                    cost = curr_cost
                    form_pay = cnt3
                cnt3 = cnt3 + 1
            pay_value.append(label_labels_to_names[pay_type[form_pay][0]])
        
        cnt2 = cnt2 + 1
    
    
    
    
    
    value = []
    cnt = 0
    while cnt < len(p):
        if p[cnt][0] != 1000 and all_poss_label[form][cnt][0] != 1000:
            value.append((p[cnt][0], label_labels_to_names[all_poss_label[form][cnt][0]], pay_value[cnt]))
            cv2.line(sign, (int(p[cnt][1]), int(p[cnt][2])), (int(all_poss_label[form][cnt][1]), int(all_poss_label[form][cnt][2])), (83, 168, 52), thickness=2)
        cnt = cnt + 1
      
    value.sort()          
    #print(value)   
    
    column_name = ['Price', 'Grade', 'Cash/Credit']
    df = pd.DataFrame(value, columns=column_name)
    
    if len(df.where(df['Grade']=='Regular').dropna()) > 1 and df.where(df['Grade']=='Regular').dropna()['Price'].max() != df.where(df['Grade']=='Regular').dropna()['Price'].min():
        #if df.loc[df.where(df['Grade']=='Regular').dropna()['Price'].idxmax(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Regular').dropna()['Price'].idxmax(), 'Cash/Credit'] = 'Credit'
        
        #if df.loc[df.where(df['Grade']=='Regular').dropna()['Price'].idxmin(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Regular').dropna()['Price'].idxmin(), 'Cash/Credit'] = 'Cash'
    
    if len(df.where(df['Grade']=='Mid-Grade').dropna()) > 1 and df.where(df['Grade']=='Mid-Grade').dropna()['Price'].max() != df.where(df['Grade']=='Mid-Grade').dropna()['Price'].min():
        #if df.loc[df.where(df['Grade']=='Mid-Grade').dropna()['Price'].idxmax(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Mid-Grade').dropna()['Price'].idxmax(), 'Cash/Credit'] = 'Credit'
        
        #if df.loc[df.where(df['Grade']=='Mid-Grade').dropna()['Price'].idxmin(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Mid-Grade').dropna()['Price'].idxmin(), 'Cash/Credit'] = 'Cash'
    
    if len(df.where(df['Grade']=='Premium').dropna()) > 1 and df.where(df['Grade']=='Premium').dropna()['Price'].max() != df.where(df['Grade']=='Premium').dropna()['Price'].min():   
        #if df.loc[df.where(df['Grade']=='Premium').dropna()['Price'].idxmax(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Premium').dropna()['Price'].idxmax(), 'Cash/Credit'] = 'Credit'
        
        #if df.loc[df.where(df['Grade']=='Premium').dropna()['Price'].idxmin(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Premium').dropna()['Price'].idxmin(), 'Cash/Credit'] = 'Cash'
     
    if len(df.where(df['Grade']=='Diesel').dropna()) > 1 and df.where(df['Grade']=='Diesel').dropna()['Price'].max() != df.where(df['Grade']=='Diesel').dropna()['Price'].min():  
        #if df.loc[df.where(df['Grade']=='Diesel').dropna()['Price'].idxmax(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Diesel').dropna()['Price'].idxmax(), 'Cash/Credit'] = 'Credit'
        
        #if df.loc[df.where(df['Grade']=='Diesel').dropna()['Price'].idxmin(), 'Cash/Credit'] == 'Both':
        df.loc[df.where(df['Grade']=='Diesel').dropna()['Price'].idxmin(), 'Cash/Credit'] = 'Cash'
    
    if len(pay_type) > 0:
        pays = np.array(pay_type)[:,0]
        if 4 in pays and 5 in pays and np.average(df['Grade'].value_counts()) == 1:
            cnt = 0
            while cnt < len(df):
                df.loc[cnt, 'Cash/Credit'] = 'Both'
                cnt = cnt + 1
        
    
    #print(df) 
    return df, sign



def ANA(df_predicted, df_actual):
    return 1. if df_predicted.equals(df_actual) else 0.

























