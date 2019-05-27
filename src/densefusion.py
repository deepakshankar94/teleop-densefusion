#! /usr/bin/env python

import rospy
from sensor_msgs.msg import Image as ImageSensor_msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import rospkg

#import segmentation code
# Import DOPE code
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('densefusion')
sys.path.append("{0}/src/vanilla_segmentation".format(g_path2package))
import segmentation_helper

def __image_callback(msg):
	#pixel value is distance in mm
	try:
		img = CvBridge().imgmsg_to_cv2(msg,desired_encoding="rgb8")
	except CvBridgeError, e:
 		print e

 	x = segmentation_helper.fetch_boundingboxes(img)
	#Convert the depth image to a Numpy array
	#depth_array = np.array(depth_image, dtype=np.float32)
	rospy.loginfo(x)


	cv2.imshow("pane name",img)
	cv2.waitKey(1)
 
if __name__ == "__main__":

	rospy.init_node('densfusion_image', anonymous=True)
	image_topic = "/camera/color/image_raw"
	rospy.Subscriber(
		image_topic,
		ImageSensor_msg,
		__image_callback

	)
	rospy.spin()