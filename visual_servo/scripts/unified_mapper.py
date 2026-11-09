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
from std_msgs.msg import String, Int32, Float32 

class HarvestInfiniteMapper: 
    def __init__(self): 
        rospy.init_node('harvest_infinite_mapper') 
        self.bridge = CvBridge() 
        self.tf_buffer = tf2_ros.Buffer() 
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 
         
        # --- 1. AI CONFIG --- 
        model_path = '/model_weights/best.pt'
        self.model = YOLO(model_path) 
        self.target_classes = [0]  
        self.CONFIDENCE = 0.45  
         
        # --- 2. MOVEMENT CONFIG (ADAPTED FROM TREE_NAVIG & SPED UP) --- 
        self.SPEED_FORWARD = 0.35   # Ускорено (в tree_navig было 0.25)
        self.SPEED_TURN = 0.5       # Ускорено (в tree_navig было 0.4)
        self.SPEED_BACKUP = -0.35   # Ускорено (в tree_navig было -0.25)
        self.SPEED_ESCAPE = 0.35    # Ускорено (в tree_navig было 0.25)
        self.SPEED_BLIND_TURN = -0.75
        self.SPEED_SEARCH = -0.5  
        self.STOP_WIDTH = 75        # Взято из tree_navig
        self.TOLERANCE = 80
        self.cx = 319.5 
         
        # --- 3. STATE MACHINE & LIMITS --- 
        self.state = 6 # 6:PERMISSION, 0:Search, 1:Approach, 2:Harvest, 3:Maneuver, 5:Stopped
        self.prev_state = 6
        self.trees_done = 0
        self.search_dir = 1  # 1: Left/Right toggle
        self.SEARCH_TIMEOUT = 15.0 
        self.state_start_time = time.time() 
        self.last_detection_time = time.time() 
        self.active = True 

        # --- 4. MAPPER & HUD CONFIG --- 
        self.CAMERA_HEIGHT = 0.25 
        self.is_calibrating = True
        self.height_samples = []
        self.dist_threshold = 0.7
        self.cur_lat = 0.0 
        self.cur_lon = 0.0 
        self.fx, self.fy, self.cx_cam, self.cy_cam = 525.0, 525.0, 319.5, 239.5 
         
        self.lock = threading.Lock() 
        self.canvas = None 
        self.latest_header = None 
        self.detections_to_draw = [] 
        self.target_box_movement = None  
        self.tree_database = []  
        self.last_depth_img = None 
        self.prev_time = 0 
        self.fps = 0 
         
        self.db_dir = '/home/jetson/tree_database'
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            rospy.loginfo(f"Created directory: {self.db_dir}")

        self.db_path = os.path.join(self.db_dir, 'harvest_data.db') 
        self.init_sqlite() 

        # --- 5. PUB/SUBS --- 
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb, queue_size=1) 
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb, queue_size=1) 
        rospy.Subscriber('/fix', NavSatFix, self.gps_cb)
        rospy.Subscriber('/emergency_stop', String, self.estop_callback)
        rospy.Subscriber('/start_mission', String, self.start_cb) 
         
        self.marker_pub = rospy.Publisher('/harvest_markers', MarkerArray, queue_size=10, latch=True) 
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10) 
        self.det_pub = rospy.Publisher('/yolo/detections', Detection2DArray, queue_size=1)
        
        self.state_pub = rospy.Publisher('/current_state', String, queue_size=10)
        self.trees_pub = rospy.Publisher('/trees_done', Int32, queue_size=10)
        self.fps_pub = rospy.Publisher('/fps', Float32, queue_size=10)
        self.gps_pub = rospy.Publisher('/gps/fps', NavSatFix, queue_size=10)
        self.hud_pub = rospy.Publisher('/camera/processed_hud', Image,  queue_size=1) 

        rospy.on_shutdown(self.shutdown) 
        rospy.Timer(rospy.Duration(4.0), self.finish_calibration, oneshot=True) 
        rospy.Timer(rospy.Duration(0.05), self.control_loop)  
         
        rospy.loginfo("INFINITE MISSION INITIALIZED.") 

    def shutdown(self): 
        self.active = False 
        self.cmd_pub.publish(Twist()) 

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
            self.cur_lat, self.cur_lon = msg.latitude, msg.longitude
            self.gps_pub.publish(msg) 

    def finish_calibration(self, event): 
        if self.height_samples: self.CAMERA_HEIGHT = np.median(self.height_samples) 
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
        rospy.loginfo(f"STATE -> {new_state}") 

    def run_inference(self, frame, frame_id): 
        if self.last_depth_img is None: return 
        results = self.model.predict(frame, conf=self.CONFIDENCE, verbose=False)[0] 
        det_msg = Detection2DArray() 
        with self.lock: det_msg.header = self.latest_header 

        temp_dets = [] 
        best_target = None 
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
                    best_target = (u, w_box) 

        with self.lock: 
            self.detections_to_draw = temp_dets 
            self.target_box_movement = best_target 
            if best_target: self.last_detection_time = time.time() 
        self.det_pub.publish(det_msg) 

    def control_loop(self, event): 
        if not self.active or self.is_calibrating: return 
        twist = Twist()
        labels = {0: "SEARCH", 1: "APPROACH", 2: "HARVEST", 3: "MANEUVER", 4: "MANUAL", 5: "STOPPED", 6: "PERMISSION", 7: "PAUSED"}
        self.state_pub.publish(labels.get(self.state, "IDLE"))
        self.trees_pub.publish(self.trees_done)
        self.fps_pub.publish(float(self.fps)) 
        
        elapsed = time.time() - self.state_start_time 
        time_since_det = time.time() - self.last_detection_time 
        with self.lock: target = self.target_box_movement

        if self.state == 6: # PERMISSION
            twist.linear.x = 0
            twist.angular.z = 0
            self.cmd_pub.publish(twist)
            return

        elif self.state == 7: # PAUSED
            twist.linear.x = 0
            twist.angular.z = 0

        elif self.state == 4: # MANUAL (WASD)
            pass 

        # === STATE 0: SEARCHING (LOGIC FROM TREE_NAVIG) ===
        elif self.state == 0: 
            if target and time_since_det < 0.5: 
                self.change_state(1) 
            elif elapsed > self.SEARCH_TIMEOUT:
                rospy.logwarn("NO TREE DETECTED - STOPPING")
                self.change_state(5)
            else: 
                twist.angular.z = self.SPEED_TURN * self.search_dir 

        # === STATE 1: APPROACHING (LOGIC FROM TREE_NAVIG) ===
        elif self.state == 1: 
            if not target or time_since_det > 1.0: 
                self.change_state(0) 
            else: 
                u, w_box = target
                err_x = u - self.cx 
                
                if w_box > self.STOP_WIDTH: 
                    self.change_state(2)
                else:
                    if abs(err_x) < self.TOLERANCE: 
                        twist.linear.x = self.SPEED_FORWARD 
                        twist.angular.z = 0.0
                    else: 
                        # Предотвращение разворота на месте (из tree_navig)
                        twist.linear.x = 0.20 # Сделал чуть быстрее (было 0.15)
                        rot_val = -float(err_x) / 400.0 
                        
                        if rot_val > 0.5: rot_val = 0.5
                        if rot_val < -0.5: rot_val = -0.5
                        
                        twist.angular.z = rot_val

        # === STATE 2: HARVESTING (LOGIC FROM TREE_NAVIG) ===
        elif self.state == 2: 
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if elapsed > 3.0: 
                self.trees_done += 1 
                self.search_dir *= -1 # Переключаем сторону поиска
                self.change_state(3) 

        # === STATE 3: MANEUVERING (LOGIC FROM TREE_NAVIG) ===
        elif self.state == 3: 
            if elapsed < 2.0:   # Backing up
                twist.linear.x = self.SPEED_BACKUP
                twist.angular.z = 0.0
            elif elapsed < 4.5: # Turning (Right -> Left -> Right...)
                twist.linear.x = 0.0
                twist.angular.z = self.SPEED_TURN * self.search_dir
            elif elapsed < 7.5: # Moving forward to next lane
                twist.linear.x = self.SPEED_ESCAPE
                twist.angular.z = 0.0
            else:
                self.change_state(0) 

        # === STATE 5: STOPPED ===
        elif self.state == 5: 
            twist.linear.x = 0; twist.angular.z = 0 

        self.cmd_pub.publish(twist)
        if self.state not in [4, 5, 7]:
            self.cmd_pub.publish(twist) 

    def get_map_point(self, u, v, c, frame): 
        try: 
            p = PointStamped() 
            p.header.frame_id, p.header.stamp = frame, rospy.Time(0) 
            p.point.x, p.point.y, p.point.z = (u-self.cx_cam)*c/self.fx, (v-self.cy_cam)*c/self.fy, c 
            return self.tf_buffer.transform(p, "map", timeout=rospy.Duration(0.01)).point 
        except: return None 

    def save_to_sql(self, dist, pt): 
        conn = sqlite3.connect(self.db_path); cur = conn.cursor() 
        cur.execute('INSERT INTO tree_map (timestamp, latitude, longitude, distance_m, map_x, map_y) VALUES (?, ?, ?, ?, ?, ?)',  
                   (datetime.datetime.now().isoformat(), self.cur_lat, self.cur_lon, dist, pt.x, pt.y)) 
        new_id = cur.lastrowid; conn.commit(); conn.close() 
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

    def estop_callback(self, msg):
        if msg.data == "STOP":
            self.change_state(5)
            rospy.logwarn("!!! EMERGENCY STOP FROM DASHBOARD !!!")
        elif msg.data == "RELEASE":
            self.change_state(0)
            rospy.loginfo("E-STOP LIFTED. Resuming...")

    def start_cb(self, msg):
        cmd = msg.data.upper()
        if cmd == "START":
            self.state = 0
        elif cmd == "PAUSE":
            if self.state != 7:
                self.prev_state = self.state
                self.state = 7
        elif cmd == "RESUME":
            if self.state == 7:
                self.state = getattr(self, 'prev_state', 0)
        elif cmd == "RESET":
             self.state = 6
             stop_twist = Twist()
             self.cmd_pub.publish(stop_twist)
        rospy.loginfo(f"--- COMMAND RECEIVED FROM DASHBOAD: {cmd} ---") 

if __name__ == '__main__': 

    mapper = HarvestInfiniteMapper() 
    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL) 
    while not rospy.is_shutdown(): 
        new_time = time.time() 
        with mapper.lock: 
            if mapper.canvas is None: continue 
            img, dets = mapper.canvas.copy(), list(mapper.detections_to_draw) 
            cal, h_val, lat, lon = mapper.is_calibrating, mapper.CAMERA_HEIGHT, mapper.cur_lat, mapper.cur_lon 
            st = mapper.state
         
        if mapper.prev_time != 0: mapper.fps = 1 / (new_time - mapper.prev_time) 
        mapper.prev_time = new_time 

        # HUD Drawing
        cv2.rectangle(img, (0,0), (img.shape[1], 65), (20,20,20), -1) 
        if cal: 
            cv2.putText(img, "CALIBRATING HEIGHT...", (20, 40), 0, 0.8, (0,140,255), 2) 
        elif st == 5:
            cv2.putText(img, "!!! NO TREE DETECTED - STOPPED !!!", (20, 40), 0, 0.7, (0,0,255), 2)
        else: 
            cv2.putText(img, f"GPS: {lat:.6f}, {lon:.6f} | Trees: {mapper.trees_done}", (20, 25), 0, 0.6, (0,255,0), 1) 
            cv2.putText(img, f"H: {h_val:.2f}m | State: {st} | Dir: {'L' if mapper.search_dir > 0 else 'R'}", (20, 50), 0, 0.6, (255,255,255), 1) 
            cv2.putText(img, f"FPS: {int(mapper.fps)}", (img.shape[1] - 120, 40), 0, 0.7, (255, 255, 0), 2) 

        for d in dets: 
            cv2.rectangle(img, (d['u']-d['w']//2, d['v']-d['h']//2), (d['u']+d['w']//2, d['v']+d['h']//2), (0,255,0), 2) 
            cv2.putText(img, f"ID:{d['id']} {d['d']:.2f}m", (d['u']-d['w']//2, d['v']-d['h']//2-10), 0, 0.6, (0,255,0), 2) 
        
        cv2.imshow("HUD", img)
        try:
            hud_msg = mapper.bridge.cv2_to_imgmsg(img, "bgr8")
            if mapper.latest_header:
                hud_msg.header = mapper.latest_header
            mapper.hud_pub.publish(hud_msg)
        except Exception as e:
            rospy.logerr(f"HUD Publish Error: {e}") 
        if cv2.waitKey(1) & 0xFF == ord('q'): break