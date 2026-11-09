#!/usr/bin/env python3

import rospy
import cv2
import os
import time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray

class TreeMission:
    def __init__(self):
        rospy.init_node('palm_mission_node')
        self.bridge = CvBridge()
        
        # === CONFIGURATION ===
        self.model_path = '/home/ubuntu/runs/detect/fake_tree_v1/weights/best.pt'
        self.CONFIDENCE = 0.45 
        self.STOP_WIDTH = 60  
        self.TOLERANCE = 100   
        
        # === VELOCITIES ===
        self.SPEED_FORWARD = 0.25
        self.SPEED_TURN = 0.5
        self.SPEED_BACKUP = -0.25
        self.SPEED_ESCAPE = 0.25
        self.SPEED_BLIND_TURN = -0.7 
        self.SPEED_SEARCH = -0.5 
        
        # === NEW: SCAN CONFIGURATION ===
        self.INITIAL_SCAN_DURATION = 6.0  # Seconds to spin before starting
        
        # === STATES ===
        # -1: Initial Scan, 0: Search, 1: Approach, 2: Harvest, 3: Escape, 4: Done
        self.state = -1 
        self.trees_done = 0         
        self.TARGET_TREES = 2       
        self.state_start_time = time.time()

        self.stop_consistency_counter = 0
        self.REQUIRED_CONSISTENCY = 3 

        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO('yolov8n.pt')

        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.det_pub = rospy.Publisher('/yolo/detections', Detection2DArray, queue_size=10)
        
        self.active = True
        self.frame_count = 0
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("MISSION V2: PRE-MISSION SCAN ACTIVE")

    def change_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.stop_consistency_counter = 0 
        rospy.loginfo(f">>> STATE CHANGED TO: {new_state} <<<")
        if new_state in [2, 4]:
            self.cmd_pub.publish(Twist()) 

    def shutdown(self):
        self.active = False
        stop_msg = Twist()
        for i in range(5):
            self.cmd_pub.publish(stop_msg)
            rospy.sleep(0.05)

    def image_callback(self, data):
        if not self.active: return
        self.frame_count += 1
        if self.frame_count % 3 != 0: return

        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            height, width, _ = frame.shape
            center_x = int(width / 2)

            results = self.model(frame, verbose=False, conf=self.CONFIDENCE, imgsz=640)
            
            det_array = Detection2DArray()
            det_array.header = data.header
            target_box = None
            max_width = 0

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    det = Detection2D()
                    det.header = data.header
                    xywh = box.xywh[0].cpu().numpy()
                    det.bbox.center.x, det.bbox.center.y = float(xywh[0]), float(xywh[1])
                    det.bbox.size_x, det.bbox.size_y = float(xywh[2]), float(xywh[3])
                    det_array.detections.append(det)

                    if w > max_width:
                        max_width = w
                        target_box = (x1, y1, x2, y2)

            # Important: We still publish detections to SQL even while scanning
            self.det_pub.publish(det_array)

            twist = Twist()
            elapsed = time.time() - self.state_start_time 
            status = f"Tree: {self.trees_done}/{self.TARGET_TREES} | State: {self.state}"
            color = (0, 255, 255)

            # === INITIAL SCAN MODE (STATE -1) ===
            if self.state == -1:
                status = f"INITIAL SCAN: {int(self.INITIAL_SCAN_DURATION - elapsed)}s"
                color = (255, 100, 0)
                twist.angular.z = self.SPEED_SEARCH # Spin to "see" trees
                if elapsed > self.INITIAL_SCAN_DURATION:
                    rospy.loginfo("SCAN COMPLETE. STARTING MISSION.")
                    self.change_state(0)

            # === SEARCH MODE (STATE 0) ===
            elif self.state == 0:
                if target_box:
                    self.change_state(1) 
                else:
                    status = "SEARCHING..."
                    twist.angular.z = self.SPEED_SEARCH 

            # === APPROACH TARGET (STATE 1) ===
            elif self.state == 1:
                if target_box:
                    x1, y1, x2, y2 = target_box
                    obj_center = int((x1 + x2) / 2)
                    offset = obj_center - center_x
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    if max_width > self.STOP_WIDTH:
                        self.stop_consistency_counter += 1
                    else:
                        self.stop_consistency_counter = 0

                    if self.stop_consistency_counter > self.REQUIRED_CONSISTENCY:
                        self.change_state(2) 
                    else:
                        if abs(offset) < self.TOLERANCE:
                            twist.linear.x = self.SPEED_FORWARD
                            status = "FORWARD"
                        else:
                            twist.linear.x = 0.15 
                            twist.angular.z = self.SPEED_TURN if offset < 0 else -self.SPEED_TURN
                            status = "CORRECTING"
                else:
                    self.change_state(0)

            # === HARVESTING (STATE 2) ===
            elif self.state == 2:
                status = f"HARVESTING... {3 - int(elapsed)}s"
                color = (0, 0, 255)
                if elapsed > 3.0:
                    self.trees_done += 1
                    self.change_state(4 if self.trees_done >= self.TARGET_TREES else 3)

            # === ESCAPE (STATE 3) ===
            elif self.state == 3:
                if elapsed < 1.5: twist.linear.x = self.SPEED_BACKUP
                elif elapsed < 3.5: twist.angular.z = self.SPEED_BLIND_TURN 
                elif elapsed < 8.0: twist.linear.x = self.SPEED_ESCAPE
                else: self.change_state(0)

            # === FINISHED (STATE 4) ===
            elif self.state == 4:
                status = "MISSION COMPLETE!"
                color = (0, 255, 0)

            self.cmd_pub.publish(twist)

            # UI Helpers
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if target_box:
                 cv2.putText(frame, f"W: {max_width}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.imshow("Navigator + Initial Scan", frame)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Err: {e}")

if __name__ == '__main__':
    try:
        TreeMission()
        rospy.spin()
    except rospy.ROSInterruptException: pass
