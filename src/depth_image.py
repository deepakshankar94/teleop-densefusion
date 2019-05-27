#! /usr/bin/env python

import rospy
from sensor_msgs.msg import Image as ImageSensor_msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

def __depth_callback(msg):
	#pixel value is distance in mm
	try:
		img = CvBridge().imgmsg_to_cv2(msg,desired_encoding="16UC1")
	except CvBridgeError, e:
 		print e
	#Convert the depth image to a Numpy array
	#depth_array = np.array(depth_image, dtype=np.float32)
	rospy.loginfo(img[480/2,640/2]/25.4)


	cv2.imshow("pane name",img)
	cv2.waitKey(1)
 
if __name__ == "__main__":

	rospy.init_node('densfusion_depth', anonymous=True)
	depth_topic = "/camera/aligned_depth_to_color/image_raw"
	rospy.Subscriber(
		depth_topic,
		ImageSensor_msg,
		__depth_callback

	)
	rospy.spin()