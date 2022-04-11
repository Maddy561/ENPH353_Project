#!/usr/bin/env python

import roslib
import numpy as np
from numpy.lib.twodim_base import mask_indices
import sys
import time
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess1 = tf.Session()    
graph1 = tf.compat.v1.get_default_graph()
set_session(sess1)


class license_tracker:
  first = 0
  sec = 0
  third = 0
  fourth = 0

  def __init__(self):
    self.license_pub = rospy.Publisher('/license_plate', String , queue_size=1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image, self.callback, queue_size = 1)
    #self.rate = rospy.Rate(20)
    while(rospy.get_time() < 0.5):
      pass
    time.sleep(1)
    self.license_pub.publish(str('Maddy,enph353,0,0000'))
    self.plate_NN = models.load_model("/home/fizzer/Downloads/conv_model.h5")

  def callback(self,data):
    try:
      start = rospy.get_time()
      if(rospy.get_time()< start+239):
        space = 2
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        (rows,cols,channels) = cv_image.shape
        #cv2.imshow('',)
        #cv2.cv2.waitKey(3)

        #crop image to bottom left corner
        cropped_frame = cv_image[300:rows, 0:int(cols/2)]
        #frames are 720 by 1280

        # Convert BGR to HSV
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        uh = 5
        us = 255
        uv = 255
        lh = 0
        ls = 70
        lv = 70
        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])
        #cv2.imshow('hsv', hsv)
        #cv2.waitKey(2)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        #cv2.imshow('mask', mask)
        #cv2.waitKey(2)

        #find corners
        corners = cv2.goodFeaturesToTrack(mask,10,0.01,85)
        if corners is not None:
          print("Corners detected")
          corners = np.int0(corners)

          #find min and max values of corners and crop
          min_y = cv_image[0].shape[0] - 300
          min_x = 640
          max_y = 0
          max_x = 0
          for r in corners:
            x,y = r.ravel()
            if y < min_y:
              min_y = y 
            if x < min_x:
              min_x = x
            if x > max_x:
              max_x = x
            if y > max_y:
              max_y = y

          if max_x > min_x and max_y > min_y:
            print("max_x:{}, min_x:{}, max_y:{}, min_y:{}".format(max_x, min_x, max_y, min_y))

            recropped_img = cropped_frame[min_y:max_y,min_x:max_x]
            print("Image cropped.")


            #cv2.imshow('recropped', recropped_img)
            #cv2.waitKey(2)

            #new!!!
            # Convert BGR to HSV
            hsv = cv2.cvtColor(recropped_img, cv2.COLOR_BGR2HSV)

            uh = 5
            us = 255
            uv = 255
            lh = 0
            ls = 70
            lv = 70
            lower_hsv = np.array([lh,ls,lv])
            upper_hsv = np.array([uh,us,uv])

            # Threshold the HSV image to get only blue colors
            mask3 = cv2.inRange(hsv, lower_hsv, upper_hsv)
            flipped = cv2.bitwise_not(mask3)


            #cv2.imshow('flipped', flipped)
            #cv2.waitKey(2)

            #find contours in the image
            _,contours,hierarchy= cv2.findContours(flipped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                  
            #find largest contoured area
            array2 = np.zeros(len(contours))
            l = 0
            for cnt in contours:
              approx = cv2.contourArea(cnt) 
              array2[l] = approx
              l = l+1

            max_location = np.where(array2 == np. amax(array2))
            max_area = contours[max_location[0][0]]

            #double-check that the shape is a rectangle
            approx_area = cv2.approxPolyDP(max_area, 0.01* cv2.arcLength(max_area, True), True)
                
            x, y , w, h = cv2.boundingRect(approx_area)

            #crop using contours
            recropped_img = recropped_img[y:y+h,x:x+w-10]

            #cv2.imshow('recropped', recropped_img)
            #cv2.waitKey(2)

            hsv = cv2.cvtColor(recropped_img, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            brightness = np.average(v)

            close_crop_gray = cv2.cvtColor(recropped_img,cv2.COLOR_BGR2GRAY)

            #create binary image of the vehicle
            threshold = brightness
            _, img_bin = cv2.threshold(close_crop_gray, threshold, 255, 0)
            flipped = cv2.bitwise_not(img_bin)  


            #cv2.imshow('flipped2', flipped)
            #cv2.waitKey(2)

            #find contours in the image
            _,contours,hierarchy = cv2.findContours(flipped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                  
            #find largest contoured area
            array2 = np.zeros(len(contours))
            l = 0
            for cnt in contours:
              approx = cv2.contourArea(cnt) 
              array2[l] = approx
              l = l+1

            max_location = np.where(array2 == np. amax(array2))
            max_area = contours[max_location[0][0]]

            #double-check that the shape is a rectangle
            approx_area = cv2.approxPolyDP(max_area, 0.01* cv2.arcLength(max_area, True), True)
                
            x, y , w, h = cv2.boundingRect(approx_area)
           
            #crop using contours
            license = recropped_img[y:y+h,0:recropped_img.shape[1]]

            cv2.imshow('license', license)
            cv2.waitKey(2)


            if (license.shape[1]/license.shape[0]>2):
              resized = cv2.resize(license, (600,298), interpolation = cv2.INTER_AREA)

              def find_char(x):
                if x == 0:
                  return "0"
                if x == 1:
                  return "1"
                if x == 2:
                  return "2"
                if x == 3:
                  return "3"
                if x == 4:
                  return "4"
                if x == 5:
                  return "5"
                if x == 6:
                  return "6"
                if x == 7:
                  return "7"
                if x ==8:
                  return "8"
                if x ==9:
                  return "9"
                if x ==10:
                  return "A"
                if x ==11:
                  return "B"
                if x ==12:
                  return "C"
                if x ==13:
                  return "D"
                if x ==14:
                  return "E"
                if x ==15:
                  return "F"
                if x ==16:
                  return "G"
                if x ==17:
                  return "H"
                if x == 18:
                  return "I"
                if x ==19:
                  return "J"
                if x ==20:
                  return "K"
                if x ==21:
                  return "L"
                if x == 22:
                  return "M"
                if x ==23:
                  return "N"
                if x ==24:
                  return "O"
                if x ==25:
                  return "P"
                if x ==26:
                  return "Q"
                if x ==27:
                  return "R"
                if x == 28:
                  return "S"
                if x == 29:
                  return "T"
                if x ==30:
                  return "U"
                if x ==31:
                  return "V"
                if x ==32:
                  return "W"
                if x ==33:
                  return "X"
                if x ==34:
                  return "Y"
                if x ==35:
                  return "Z" 

                first_char = resized[0][0:resized[0].shape[0], 50:145]
                global sess1
                global graph1
                with graph1.as_default():
                  set_session(sess1)
                  img_aug = np.expand_dims(first_char, axis=0)
                  first = np.argmax(self.plate_NN.predict(img_aug)[0])


                             
                second_char = resized[0][0:resized[0].shape[0], 150:245]
                global sess1
                global graph1
                with graph1.as_default():
                  set_session(sess1)
                  img_aug = np.expand_dims(second_char, axis=0)
                  sec = np.argmax(self.plate_NN.predict(img_aug)[0])

                third_char = resized[0][0:resized[0].shape[0], 350:445]
                global sess1
                global graph1
                with graph1.as_default():
                  set_session(sess1)
                  img_aug = np.expand_dims(third_char, axis=0)
                  third = np.argmax(self.plate_NN.predict(img_aug)[0])

                fourth_char = resized[0][0:resized[0].shape[0], 450:545]
                global sess1
                global graph1
                with graph1.as_default():
                  set_session(sess1)
                  img_aug = np.expand_dims(fourth_char, axis=0)
                  fourth = np.argmax(self.plate_NN.predict(img_aug)[0])

                c1 = find_char(first)
                c2 = find_char(sec)
                c3 = find_char(third)
                c4 = find_char(fourth)
                license = c1+c2+c3+c4
                self.license_pub.publish(str('Maddy,enph353,{},{}'.format(license,str(space))))
                space = space +1

      else:
        self.license_pub.publish(str('Maddy,enph353,-1,0000'))
        
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('license_tracker', anonymous=True)
  lf = license_tracker()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
