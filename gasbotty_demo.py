# -*- coding: utf-8 -*-
"""
GasBotty: Multi-Metric Extraction in the Wild
Authors: Anonymous 
---
End-to-end demonstration of the GasBotty pipeline on 1 of 5 sample images. 
"""

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import glob
import os
os.chdir('./GasBotty/')

from GasBotty.gasbotty import *

#Step 1: Import each of pre-trained models
sign_model, price_model, digit_model, label_model  = load_models()

#Step 2: Load example image, select example image by index (choose one: [1, 2, 3, 4, 5])
img_idx  = 1
img_file = f'../example-images/example_{img_idx}.png'
gt_file  = f'../example-images/example_{img_idx}.csv' 
image    = cv2.imread(img_file)
plt.imshow(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
plt.show()

#Step 3: From image, predict sign-level mask
mask = get_mask(image, sign_model)
plt.imshow(mask)
plt.show()

#Step 4: From predicted sign-level mask, extract border
border = get_border(mask)
plt.imshow(border)
plt.show()

#Step 5: Using extracted border, detect the Hough lines
line_df, line_image = get_lines(border, image.copy())
plt.imshow(cv2.cvtColor(line_image.copy(), cv2.COLOR_BGR2RGB))
plt.show()   
    
#Step 6: From detected Hough lines, obtain points of intersection
intersection_df, intersection_image = get_intersections(line_df, line_image)
plt.imshow(cv2.cvtColor(intersection_image.copy(), cv2.COLOR_BGR2RGB))
plt.show()  

#Step 7: From points of intersection, obtain sign-level corners
src_pts, corners_image = get_corners(intersection_df, intersection_image)
plt.imshow(cv2.cvtColor(corners_image.copy(), cv2.COLOR_BGR2RGB))
plt.show() 

#Step 8: Keystone correct the sign using four corners; 
#        generate sign-level, perspective-corrected image
keystone_image = get_KS(src_pts, image.copy())
plt.imshow(cv2.cvtColor(keystone_image.copy(), cv2.COLOR_BGR2RGB))
plt.show() 

#Step 9: Using sign-level image, extract all prices
where_prices, price_image = price_level(keystone_image.copy(), price_model)
plt.imshow(cv2.cvtColor(price_image.copy(), cv2.COLOR_BGR2RGB))
plt.show() 

#Step 10: Using sign-level image, extract all labels
where_labels, label_image = label_level(keystone_image.copy(), label_model, price_image)
plt.imshow(cv2.cvtColor(label_image.copy(), cv2.COLOR_BGR2RGB))
plt.show() 

#Step 11: Using all extracted price-level images, detect all digits
prices = []
for price in where_prices:
    current_price, price_image = read_digits(keystone_image.copy()[price[1]:price[3], price[0]:price[2]] , digit_model )
    prices.append(( price[0], price[1], price[2], price[3], current_price))        
    plt.imshow(cv2.cvtColor(price_image.copy(), cv2.COLOR_BGR2RGB))
    plt.title(str(current_price))
    plt.show() 

#Step 12: Using all extracted prices and labels, associate and generate final prediction
df , associate_image = associate(prices, where_labels, label_image)
plt.imshow(cv2.cvtColor(associate_image.copy(), cv2.COLOR_BGR2RGB))
plt.show() 
print(df)

#Final step: evaluate performance with groung truth annotation
df_groundtruth = pd.read_csv(gt_file)
df_groundtruth['Price'] = df_groundtruth['Price'].round(3)
print(f'All-or-Nothing Accuracy (ANA): {ANA(df, df_groundtruth)}\n\nPredicted:\n{df}\n\nGround Truth:\n{df_groundtruth}')

#Thats all Folks!

