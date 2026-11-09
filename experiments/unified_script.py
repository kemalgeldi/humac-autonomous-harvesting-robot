#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import sqlite3
import os
import cv2
import datetime
import threading
import time
from ultralytics import YOLO  
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PointStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray 
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray

class HarvestUnifiedMapper:
    def __init__(self):
        rospy.init_node('harvest_unified_mapper')
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- 1. AI CONFIG ---
        model_path = '/home/ubuntu/runs/detect/fake_tree_v1/weights/best.pt'
        self.model = YOLO(model_path)
        self.target_classes = [0] 
        self.CONFIDENCE = 0.45 
        
        # --- 2. MOVEMENT CONFIG ---
        self.STOP_WIDTH = 75    
        self.TOLERANCE = 80     
        self.SPEED_FORWARD = 0.25
        self.CENTER_X = 319.5 
        
        # --- 3. STATE MACHINE ---
        self.state = 0
        self.trees_done = 0
        self.TARGET_TREES = 3   
        self.search_dir = 1
        self.state_start_time = time.time()
        self.last_detection_time = time.time()
        self.active = True

        # --- 4. MAPPER & HUD CONFIG ---
        self.CAMERA_HEIGHT = 0.25
        self.is_calibrating = True      
        self.height_samples = []        
        self.dist_threshold = 0.8  
        self.cur_lat = 0.0
        self.cur_lon = 0.0
        self.fx, self.fy, self.cx, self.cy = 525.0, 525.0, 319.5, 239.5
        
        self.lock = threading.Lock()
        self.canvas = None
        self.latest_header = None
        self.detections_to_draw = []
        self.target_box_movement = None 
        self.tree_database = [] 
        self.last_depth_img = None
        
        # FPS Calculation
        self.prev_time = 0
        self.fps = 0
        
        self.db_path = os.path.expanduser('~/harvest_data.db')
        self.init_sqlite()

        # --- 5. PUB/SUBS ---
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb, queue_size=1)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb, queue_size=1)
        rospy.Subscriber('/fix', NavSatFix, self.gps_cb)
        
        self.marker_pub = rospy.Publisher('/harvest_markers', MarkerArray, queue_size=10, latch=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.det_pub = rospy.Publisher('/yolo/detections', Detection2DArray, queue_size=1)

        rospy.on_shutdown(self.shutdown)
        rospy.Timer(rospy.Duration(4.0), self.finish_calibration, oneshot=True)
        rospy.Timer(rospy.Duration(0.1), self.control_loop) 
        
        rospy.loginfo("MAPPER & CONTROLLER INITIALIZED.")

    def shutdown(self):
        self.active = False
        stop_msg = Twist()
        for i in range(10):
            self.cmd_pub.publish(stop_msg)
            rospy.sleep(0.05)

    def init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS tree_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            latitude REAL, longitude REAL, distance_m REAL,
            map_x REAL, map_y REAL)''')
        cursor.execute("SELECT id, map_x, map_y FROM tree_map")
        for row in cursor.fetchall():
            self.tree_database.append({'id': row[0], 'x': row[1], 'y': row[2]})
        conn.close()

    def gps_cb(self, msg):
        with self.lock:
            self.cur_lat = msg.latitude
            self.cur_lon = msg.longitude

    def finish_calibration(self, event):
        if self.height_samples:
            self.CAMERA_HEIGHT = np.median(self.height_samples)
        self.is_calibrating = False
        rospy.loginfo(f"CALIBRATION DONE: {self.CAMERA_HEIGHT:.3f}m")

    def depth_cb(self, msg): 
        self.last_depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        if self.is_calibrating:
            h, w = self.last_depth_img.shape
            roi = self.last_depth_img[h-100:h-20, w//2-30:w//2+30]
            valid = roi[roi > 0]
            if len(valid) > 0: self.height_samples.append(np.mean(valid) / 1000.0)

    def image_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.canvas = cv_img
                self.latest_header = msg.header
            if not any(t.name == "AI_Worker" for t in threading.enumerate()):
                threading.Thread(target=self.run_inference, args=(cv_img.copy(), msg.header.frame_id), name="AI_Worker").start()
        except: pass

    def change_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.cmd_pub.publish(Twist())
        rospy.loginfo(f">>> STATE CHANGED: {new_state} <<<")

    def run_inference(self, frame, frame_id):
        if self.last_depth_img is None: return
        results = self.model.predict(frame, conf=self.CONFIDENCE, verbose=False)[0]
        
        det_msg = Detection2DArray()
        with self.lock: det_msg.header = self.latest_header

        temp_dets = []
        best_box_move = None
        max_w = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in self.target_classes: continue
            
            xyxy = box.xyxy[0].cpu().numpy()
            u, v = int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)
            w_box = int(xyxy[2] - xyxy[0])
            
            patch = self.last_depth_img[v-2:v+3, u-2:u+3]
            valid_pixels = patch[patch > 0]
            if len(valid_pixels) > 0 and not self.is_calibrating:
                c = np.median(valid_pixels) / 1000.0
                dist = np.sqrt(max(0, c**2 - self.CAMERA_HEIGHT**2)) if c > self.CAMERA_HEIGHT else 0.1
                p_map = self.get_map_point(u, v, c, frame_id)
                
                tid = "..."
                if p_map:
                    match_id = None
                    for tree in self.tree_database:
                        if np.hypot(p_map.x - tree['x'], p_map.y - tree['y']) < self.dist_threshold:
                            match_id = tree['id']; break
                    if match_id: tid = str(match_id)
                    else:
                        new_id = self.save_to_sql(dist, p_map)
                        self.tree_database.append({'id': new_id, 'x': p_map.x, 'y': p_map.y})
                        tid = str(new_id)
                        self.publish_markers()
                
                temp_dets.append({'u':u, 'v':v, 'w':w_box, 'h':int(xyxy[3]-xyxy[1]), 'id':tid, 'd':dist})

                if w_box > max_w:
                    max_w = w_box
                    best_box_move = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])

        with self.lock:
            self.detections_to_draw = temp_dets
            self.target_box_movement = best_box_move
            if best_box_move: self.last_detection_time = time.time()
        
        self.det_pub.publish(det_msg)

    def control_loop(self, event):
        if not self.active or self.is_calibrating: return
        twist = Twist()
        elapsed = time.time() - self.state_start_time
        time_since_det = time.time() - self.last_detection_time
        with self.lock: target = self.target_box_movement

        if self.state == 0:
            if target and time_since_det < 0.5: self.change_state(1)
            else: twist.angular.z = 0.4 * self.search_dir
        elif self.state == 1:
            if not target or time_since_det > 1.0: self.change_state(0)
            else:
                x1, y1, x2, y2 = target
                w = x2 - x1
                cx_obj = (x1 + x2) // 2
                err_x = cx_obj - self.cx
                if w > self.STOP_WIDTH: self.change_state(2)
                else:
                    twist.linear.x = self.SPEED_FORWARD if abs(err_x) < self.TOLERANCE else 0.15
                    twist.angular.z = np.clip(-float(err_x)/400.0, -0.4, 0.4)
        elif self.state == 2:
            if elapsed > 3.0:
                self.trees_done += 1
                if self.trees_done >= self.TARGET_TREES: self.change_state(4)
                else: self.search_dir *= -1; self.change_state(3)
        elif self.state == 3:
            if elapsed < 2.0: twist.linear.x = -0.25
            elif elapsed < 4.5: twist.angular.z = 0.5 * self.search_dir
            elif elapsed < 7.5: twist.linear.x = 0.25
            else: self.change_state(0)

        self.cmd_pub.publish(twist)

    def get_map_point(self, u, v, c, frame):
        try:
            p = PointStamped()
            p.header.frame_id, p.header.stamp = frame, rospy.Time(0)
            p.point.x, p.point.y, p.point.z = (u-self.cx)*c/self.fx, (v-self.cy)*c/self.fy, c
            return self.tf_buffer.transform(p, "map", timeout=rospy.Duration(0.01)).point
        except: return None

    def save_to_sql(self, dist, pt):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('INSERT INTO tree_map (timestamp, latitude, longitude, distance_m, map_x, map_y) VALUES (?, ?, ?, ?, ?, ?)', 
                   (datetime.datetime.now().isoformat(), self.cur_lat, self.cur_lon, dist, pt.x, pt.y))
        new_id = cur.lastrowid
        conn.commit(); conn.close()
        return new_id

    def publish_markers(self):
        ma = MarkerArray()
        for tree in self.tree_database:
            m = Marker()
            m.header.frame_id, m.id = "map", tree['id']
            m.type, m.action = Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.scale.x = tree['x'], tree['y'], 0.4
            m.scale.y, m.scale.z, m.color.a, m.color.g = 0.4, 1.0, 1.0, 1.0
            m.pose.orientation.w = 1.0
            ma.markers.append(m)
        self.marker_pub.publish(ma)

if __name__ == '__main__':
    mapper = HarvestUnifiedMapper()
    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
    while not rospy.is_shutdown():
        new_time = time.time()
        with mapper.lock:
            img = mapper.canvas.copy() if mapper.canvas is not None else None
            dets = list(mapper.detections_to_draw)
            cal, h_val, lat, lon = mapper.is_calibrating, mapper.CAMERA_HEIGHT, mapper.cur_lat, mapper.cur_lon
        
        # Calculate FPS
        if mapper.prev_time != 0:
            mapper.fps = 1 / (new_time - mapper.prev_time)
        mapper.prev_time = new_time

        if img is not None:
            cv2.rectangle(img, (0,0), (img.shape[1], 60), (20,20,20), -1)
            if cal:
                cv2.putText(img, "CALIBRATING HEIGHT...", (20, 35), 0, 0.7, (0,140,255), 2)
            else:
                gps_txt = f"GPS: {lat:.6f}, {lon:.6f} | DB: {len(mapper.tree_database)}"
                cv2.putText(img, gps_txt, (20, 25), 0, 0.6, (0,255,0), 1)
                cv2.putText(img, f"H: {h_val:.2f}m | State: {mapper.state}", (20, 48), 0, 0.6, (255,255,255), 1)
                
                # FPS Counter in TOP RIGHT
                fps_txt = f"FPS: {int(mapper.fps)}"
                cv2.putText(img, fps_txt, (img.shape[1] - 120, 35), 0, 0.7, (255, 255, 0), 2)

            for d in dets:
                cv2.rectangle(img, (d['u']-d['w']//2, d['v']-d['h']//2), (d['u']+d['w']//2, d['v']+d['h']//2), (0,255,0), 2)
                label = f"ID:{d['id']} {d['d']:.2f}m"
                cv2.putText(img, label, (d['u']-d['w']//2, d['v']-d['h']//2-10), 0, 0.6, (0,255,0), 2)
            cv2.imshow("HUD", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
