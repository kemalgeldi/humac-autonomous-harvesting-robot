#!/usr/bin/env python3
import rospy
import cv2
import time
import threading
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2D, Detection2DArray
from ultralytics import YOLO

class AutonomousHarvester:
    def __init__(self):
        rospy.init_node('palm_mission_node')
        self.bridge = CvBridge()
        
        # --- MODEL SETTINGS ---
        self.model_path = '/home/ubuntu/runs/detect/fake_tree_v1/weights/best.pt'
        self.CONFIDENCE = 0.45  
        
        # --- CAMERA SETTINGS ---
        self.IMG_WIDTH = 640
        self.CENTER_X = self.IMG_WIDTH // 2
        
        # --- MOVEMENT SETTINGS ---
        self.STOP_WIDTH = 75     
        self.TOLERANCE = 80      
        self.SPEED_FORWARD = 0.25
        
        # --- MISSION LOGIC ---
        self.state = 0
        self.trees_done = 0
        self.TARGET_TREES = 3    
        self.search_dir = 1      # 1 = Left, -1 = Right
        
        self.state_start_time = time.time()
        self.last_detection_time = time.time()
        
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_header = None 
        self.target_box = None    
        
        # === MAIN FIX (Consistent with tree_navig.py) ===
        self.active = True
        
        # Register shutdown function immediately
        rospy.on_shutdown(self.shutdown)

        # --- LOADING YOLO ---
        if os.path.exists(self.model_path):
            rospy.loginfo(f"Loading Weights: {self.model_path}")
            self.model = YOLO(self.model_path)
        else:
            rospy.logwarn("Weights not found! Loading standard yolov8n.pt.")
            self.model = YOLO('yolov8n.pt')

        # --- ROS TOPICS ---
        self.img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.det_pub = rospy.Publisher('/yolo/detections', Detection2DArray, queue_size=1) 

        # --- START THREADS ---
        self.ai_thread = threading.Thread(target=self.inference_loop)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        
        # Control Loop Timer
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo(f"MISSION START: Looking LEFT first. Target: {self.TARGET_TREES} trees.")

    # === SHUTDOWN FUNCTION ===
    def shutdown(self):
        rospy.loginfo("SHUTDOWN TRIGGERED! Stopping motors...")
        self.active = False
        
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        
        # Publish stop command 10 times to ensure the driver receives it
        for i in range(10):
            self.cmd_pub.publish(stop_msg)
            rospy.sleep(0.05)
            
        rospy.loginfo("MOTORS STOPPED.")

    def img_cb(self, data):
        try:
            with self.lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.latest_header = data.header
        except Exception as e:
            pass

    def change_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        rospy.loginfo(f">>> STATE CHANGED: {new_state} <<<")
        # Short stop during state transition
        self.cmd_pub.publish(Twist())

    def inference_loop(self):
        while not rospy.is_shutdown() and self.active:
            img = None
            header = None
            
            with self.lock:
                if self.latest_frame is not None:
                    img = self.latest_frame.copy()
                    header = self.latest_header
            
            if img is None:
                time.sleep(0.01)
                continue

            results = self.model(img, verbose=False, conf=self.CONFIDENCE, imgsz=640)
            
            det_msg = Detection2DArray()
            det_msg.header = header 
            
            best_box = None
            max_w = 0

            for r in results:
                for box in r.boxes:
                    xywh = box.xywh[0].cpu().numpy()
                    d = Detection2D()
                    d.bbox.center.x = float(xywh[0])
                    d.bbox.center.y = float(xywh[1])
                    d.bbox.size_x = float(xywh[2])
                    d.bbox.size_y = float(xywh[3])
                    det_msg.detections.append(d)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    
                    if w > max_w:
                        max_w = w
                        best_box = (x1, y1, x2, y2)

            self.det_pub.publish(det_msg)

            with self.lock:
                self.target_box = best_box
                if best_box is not None:
                    self.last_detection_time = time.time()
            
            direction_text = "SEARCH: LEFT" if self.search_dir == 1 else "SEARCH: RIGHT"
            cv2.putText(img, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Robot Eye", img)
            if cv2.waitKey(1) == ord('q'):
                rospy.signal_shutdown("User exit")

    def control_loop(self, event):
        if rospy.is_shutdown() or not self.active:
            return

        twist = Twist()
        elapsed = time.time() - self.state_start_time
        time_since_det = time.time() - self.last_detection_time
        
        target = None
        with self.lock:
            target = self.target_box
        
        # === STATE 0: SEARCHING ===
        if self.state == 0:
            if target and time_since_det < 0.5:
                rospy.loginfo(f"Tree Found! Stopping rotation.")
                self.change_state(1)
            else:
                # Rotation
                twist.angular.z = 0.4 * self.search_dir 
        
        # === STATE 1: APPROACHING ===
        elif self.state == 1:
            if not target or time_since_det > 1.0:
                rospy.logwarn("Target lost! Resuming search.")
                self.change_state(0)
            else:
                x1, y1, x2, y2 = target
                w = x2 - x1
                cx = (x1 + x2) // 2
                err_x = cx - self.CENTER_X
                
                if w > self.STOP_WIDTH:
                    rospy.loginfo(f"Arrived at Tree #{self.trees_done + 1}")
                    self.change_state(2)
                else:
                    if abs(err_x) < self.TOLERANCE:
                        twist.linear.x = self.SPEED_FORWARD
                        twist.angular.z = 0.0
                    else:
                        # FIX: Prevent single-wheel pivot
                        # Limit rotation to ensure the robot moves in an arc
                        twist.linear.x = 0.15 
                        rot_val = -float(err_x) / 400.0 
                        
                        # Clamp rotation speed
                        if rot_val > 0.4: rot_val = 0.4
                        if rot_val < -0.4: rot_val = -0.4
                        
                        twist.angular.z = rot_val

        # === STATE 2: HARVESTING ===
        elif self.state == 2:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if elapsed > 3.0: 
                self.trees_done += 1
                if self.trees_done >= self.TARGET_TREES:
                    self.change_state(4) 
                else:
                    self.search_dir *= -1 # Toggle search direction
                    self.change_state(3) 

        # === STATE 3: MANEUVERING ===
        elif self.state == 3:
            if elapsed < 2.0:   # Backing up
                twist.linear.x = -0.25
                twist.angular.z = 0.0
            elif elapsed < 4.5: # Turning
                twist.linear.x = 0.0
                twist.angular.z = 0.5 * self.search_dir
            elif elapsed < 7.5: # Moving forward to next lane
                twist.linear.x = 0.25
                twist.angular.z = 0.0
            else:
                self.change_state(0) 

        # === STATE 4: FINISHED ===
        elif self.state == 4:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        AutonomousHarvester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()