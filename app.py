import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AI ตรวจจับวัตถุเว็บแคม", 
    page_icon="📹", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px #ccc;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #424242;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-active {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .status-inactive {
        background-color: #F5F5F5;
        border-left: 4px solid #757575;
    }
    .status-stopped {
        background-color: #FFEBEE;
        border-left: 4px solid #E53935;
    }
    .info-box {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .big-button {
        font-size: 1.2rem !important;
        height: 3rem;
        margin: 0.5rem 0;
    }
    .stat-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stat-label {
        font-size: 1rem;
        color: #757575;
    }
    .fps-meter {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
        border-radius: 0.5rem;
        background: linear-gradient(90deg, #f5f5f5 0%, #e0f7fa 100%);
        margin-bottom: 1rem;
    }
    .detection-history-header {
        font-size: 1.3rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    footer {
        margin-top: 3rem;
        text-align: center;
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False

if 'running' not in st.session_state:
    st.session_state.running = False

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# โหลดโมเดล YOLOv8
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # ใช้โมเดลขนาดเล็ก

model = load_model()

# COCO class names for detected objects
class_names = {
    0: "Person",
    15: "Cat",    # แมว
    16: "Dog",    # สุนัข
    67: "Phone"
}

# Color mapping for detected objects
color_map = {
    0: (0, 255, 0),    # Person - Green
    15: (255, 0, 0),   # Cat - Blue
    16: (0, 0, 255),   # Dog - Red
    67: (255, 255, 0)  # Phone - Yellow
}

# เพิ่มโค้ดหลังจาก imports
SAVE_DIR = "captured_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ฟังก์ชันสำหรับหยุดการตรวจจับ
def stop_detection():
    st.session_state.stop_detection = True
    st.session_state.running = False
    st.session_state.start_time = None

# ฟังก์ชันสำหรับเริ่มตรวจจับ
def start_detection():
    st.session_state.running = True
    st.session_state.stop_detection = False
    st.session_state.start_time = datetime.now()

# ฟังก์ชันสำหรับล้างประวัติการตรวจจับ
def clear_history():
    st.session_state.detection_history = []

# แก้ไขฟังก์ชัน save_detection ให้บันทึกรูป
def save_detection(frame, detected_counts):
    if any(count > 0 for count in detected_counts.values()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # บันทึกรูปภาพ
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.save(filepath)
        
        # เก็บข้อมูลการตรวจจับ
        detection_info = {
            "timestamp": datetime.now(),
            "filename": filename,
            "detections": {
                "person": detected_counts[0],
                "cat": detected_counts[15],
                "dog": detected_counts[16],
                "phone": detected_counts[67]
            }
        }
        st.session_state.detection_history.append(detection_info)

# ฟังก์ชันสำหรับดึงภาพจากเว็บแคม
def detect_objects():
    try:
        # เปิดเว็บแคม
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("ไม่สามารถเปิดเว็บแคมได้ กรุณาตรวจสอบการเชื่อมต่อหรือการอนุญาตให้ใช้กล้อง")
            st.session_state.running = False
            return
            
        # สร้างพื้นที่สำหรับแสดงผลวิดีโอและข้อมูลสถิติ
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # สถานะการทำงาน
            status_box = st.empty()
            status_box.markdown(
                """<div class="status-box status-active">
                   <h3>📹 กำลังตรวจจับ...</h3>
                   <p>กำลังประมวลผลภาพจากเว็บแคมของคุณ</p>
                   </div>""", 
                unsafe_allow_html=True
            )
            
            # พื้นที่แสดงวิดีโอ
            video_container = st.container()
            with video_container:
                stframe = st.empty()
        
        with col2:
            # สรุปการตรวจจับ
            st.markdown('<h3 class="sub-header">📊 สถิติการตรวจจับ</h3>', unsafe_allow_html=True)
            
            # แสดง FPS
            fps_container = st.empty()
            
            # แสดงเวลาที่ใช้ในการตรวจจับ
            timer_container = st.empty()
            
            # แสดงจำนวนวัตถุที่ตรวจพบ
            stats_container = st.container()
            with stats_container:
                person_stat = st.empty()
                cat_stat = st.empty()
                dog_stat = st.empty()
                phone_stat = st.empty()
        
        # ตัวแปรสำหรับคำนวณ FPS
        frame_count = 0
        fps_list = []
        prev_time = 0
        curr_time = 0
        fps = 0
        
        # บันทึกเวลาเริ่มต้น
        record_interval = 5  # บันทึกทุก 5 วินาที
        last_record_time = time.time()
        
        # วนลูปเพื่อประมวลผลวิดีโอ
        while cap.isOpened() and not st.session_state.stop_detection:
            # อ่านเฟรมจากเว็บแคม
            ret, frame = cap.read()
            if not ret:
                st.warning("ไม่สามารถอ่านเฟรมจากเว็บแคมได้")
                break
            
            # คำนวณ FPS
            curr_time = time.time()
            if prev_time > 0:
                fps = 1 / (curr_time - prev_time)
                fps_list.append(fps)
                if len(fps_list) > 30:  # คำนวณค่าเฉลี่ย FPS จาก 30 เฟรมล่าสุด
                    fps_list.pop(0)
            prev_time = curr_time
            
            # นับจำนวนเฟรม
            frame_count += 1
            
            # ใช้โมเดล YOLO ตรวจจับวัตถุ
            results = model(frame, conf=0.5)  # กำหนดค่า confidence threshold
            
            # สร้างตัวแปรสำหรับนับจำนวนวัตถุ
            detected_counts = {
                0: 0,   # Person
                15: 0,  # Cat
                16: 0,  # Dog
                67: 0   # Phone
            }
            
            # วาดกรอบและนับจำนวนวัตถุ
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id in detected_counts:
                        detected_counts[class_id] += 1
                        
                        # ดึงพิกัดและค่า confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # วาดกรอบรอบวัตถุ
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[class_id], 2)
                        
                        # เพิ่มข้อความแสดงชื่อวัตถุและค่า confidence
                        label = f"{class_names[class_id]} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[class_id], 2)
            
            # แสดงจำนวนวัตถุและ FPS บนภาพ
            overlay = frame.copy()
            # วาดพื้นหลังโปร่งใสสำหรับข้อความ
            cv2.rectangle(overlay, (10, 10), (250, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            y_pos = 40
            for class_id, name in class_names.items():
                count = detected_counts[class_id]
                cv2.putText(frame, f"{name}: {count}", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[class_id], 2)
                y_pos += 30
                
            avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # เพิ่มวันที่และเวลาบนภาพ
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (frame.shape[1] - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # แสดงภาพใน Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)
            
            # คำนวณเวลาที่ใช้ไป
            elapsed_time = datetime.now() - st.session_state.start_time if st.session_state.start_time else datetime.now()
            elapsed_seconds = elapsed_time.total_seconds()
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # อัปเดต FPS
            avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
            fps_color = "green" if avg_fps > 15 else "orange" if avg_fps > 10 else "red"
            fps_container.markdown(
                f'<div class="fps-meter">FPS: <span style="color:{fps_color}">{avg_fps:.1f}</span></div>', 
                unsafe_allow_html=True
            )
            
            # อัปเดตเวลาที่ใช้
            timer_container.markdown(
                f'<div class="stat-card">เวลาตรวจจับ: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}</div>', 
                unsafe_allow_html=True
            )
            
            # อัปเดตจำนวนวัตถุที่ตรวจพบ
            person_stat.markdown(
                f'<div class="stat-card"><div class="stat-value">{detected_counts[0]}</div><div class="stat-label">คน</div></div>', 
                unsafe_allow_html=True
            )
            cat_stat.markdown(
                f'<div class="stat-card"><div class="stat-value">{detected_counts[15]}</div><div class="stat-label">แมว</div></div>', 
                unsafe_allow_html=True
            )
            dog_stat.markdown(
                f'<div class="stat-card"><div class="stat-value">{detected_counts[16]}</div><div class="stat-label">สุนัข</div></div>', 
                unsafe_allow_html=True
            )
            phone_stat.markdown(
                f'<div class="stat-card"><div class="stat-value">{detected_counts[67]}</div><div class="stat-label">โทรศัพท์</div></div>', 
                unsafe_allow_html=True
            )
            
            # บันทึกสถิติการตรวจจับทุก 5 วินาที
            if curr_time - last_record_time > record_interval:
                save_detection(frame, detected_counts)
                last_record_time = curr_time

            # บันทึกภาพเมื่อกดหยุด
            if st.session_state.stop_detection and any(detected_counts.values()):
                save_detection(frame, detected_counts)
        
        # ปิดการเชื่อมต่อกับเว็บแคม
        cap.release()
        
        # แสดงสถานะหยุดการตรวจจับ
        status_box.markdown(
            """<div class="status-box status-stopped">
               <h3>⏹️ หยุดการตรวจจับแล้ว</h3>
               <p>การตรวจจับถูกหยุดโดยผู้ใช้หรือเกิดข้อผิดพลาด</p>
               </div>""", 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
    finally:
        st.session_state.running = False
        st.session_state.stop_detection = False

# UI ใน Streamlit
st.markdown('<h1 class="main-header">🎥 AI ตรวจจับวัตถุเว็บแคม</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ การตั้งค่า</h2>', unsafe_allow_html=True)
    
    confidence = st.slider(
        "ค่าความเชื่อมั่น (Confidence)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="ค่าความเชื่อมั่นขั้นต่ำสำหรับการตรวจจับวัตถุ (ค่ายิ่งสูงยิ่งเข้มงวด)"
    )
    
    st.markdown("---")
    
    st.markdown('<h2 class="sub-header">📝 คำอธิบาย</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
        <p>แอปพลิเคชันนี้ใช้ AI ตรวจจับวัตถุแบบเรียลไทม์ด้วยโมเดล YOLOv8 สามารถตรวจจับวัตถุต่อไปนี้:</p>
        <ul>
            <li><strong>คน</strong> (สีเขียว)</li>
            <li><strong>แมว</strong> (สีน้ำเงิน)</li>
            <li><strong>สุนัข</strong> (สีแดง)</li>
            <li><strong>โทรศัพท์</strong> (สีเหลือง)</li>
        </ul>
        <p>ระบบจะแสดงกรอบรอบวัตถุที่ตรวจพบพร้อมค่าความเชื่อมั่น และนับจำนวนวัตถุแต่ละประเภท</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    st.markdown('<h3 class="sub-header">📸 ภาพที่บันทึก</h3>', unsafe_allow_html=True)
    if st.button("ล้างประวัติ", on_click=clear_history):
        # ลบไฟล์รูปภาพทั้งหมด
        for file in os.listdir(SAVE_DIR):
            if file.endswith('.jpg'):
                os.remove(os.path.join(SAVE_DIR, file))
    
    if st.session_state.detection_history:
        # แสดงรูปภาพล่าสุด 5 รูป
        latest_detections = st.session_state.detection_history[-5:]
        for detection in reversed(latest_detections):
            col1, col2 = st.columns([1, 2])
            
            # แสดงรูปภาพ
            img_path = os.path.join(SAVE_DIR, detection['filename'])
            if os.path.exists(img_path):
                with col1:
                    st.image(img_path, width=150)
            
            # แสดงรายละเอียดการตรวจจับ
            with col2:
                timestamp = detection['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"**เวลา:** {timestamp}")
                detections = detection['detections']
                detected_objects = []
                if detections['person']: detected_objects.append(f"คน ({detections['person']})")
                if detections['cat']: detected_objects.append(f"แมว ({detections['cat']})")
                if detections['dog']: detected_objects.append(f"สุนัข ({detections['dog']})")
                if detections['phone']: detected_objects.append(f"โทรศัพท์ ({detections['phone']})")
                st.markdown(f"**พบ:** {', '.join(detected_objects)}")
            
            st.markdown("---")
    else:
        st.info("ยังไม่มีภาพที่บันทึก")

# Add after UI section before webcam controls
st.markdown("### หรือ อัปโหลดรูปภาพ")
uploaded_file = st.file_uploader("เลือกรูปภาพ", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # อ่านไฟล์รูปภาพ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # แสดงรูปต้นฉบับ
    st.image(frame, caption='รูปภาพที่อัปโหลด', channels="BGR", use_container_width=True)
    
    if st.button("วิเคราะห์รูปภาพ"):
        # ใช้โมเดล YOLO ตรวจจับวัตถุ
        results = model(frame, conf=0.5)
        
        # สร้างตัวแปรสำหรับนับจำนวนวัตถุ
        detected_counts = {
            0: 0,   # Person
            15: 0,  # Cat
            16: 0,  # Dog
            67: 0   # Phone
        }
        
        # วาดกรอบและนับจำนวนวัตถุ
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in detected_counts:
                    detected_counts[class_id] += 1
                    
                    # ดึงพิกัดและค่า confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # วาดกรอบรอบวัตถุ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[class_id], 2)
                    
                    # เพิ่มข้อความแสดงชื่อวัตถุและค่า confidence
                    label = f"{class_names[class_id]} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[class_id], 2)
        
        # แสดงรูปที่มีการตรวจจับ
        st.image(frame, caption='ผลการตรวจจับ', channels="BGR", use_container_width=True)
        
        # แสดงสรุปผลการตรวจจับ
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("คน", detected_counts[0])
        with col2:
            st.metric("แมว", detected_counts[15])
        with col3:
            st.metric("สุนัข", detected_counts[16])
        with col4:
            st.metric("โทรศัพท์", detected_counts[67])
        
        # บันทึกผลการตรวจจับ
        save_detection(frame, detected_counts)

st.markdown("---")

# แสดงสถานะการทำงาน
if not st.session_state.running:
    st.markdown(
        """<div class="status-box status-inactive">
           <h3>🎬 พร้อมตรวจจับ</h3>
           <p>กดปุ่ม "เริ่มตรวจจับ" เพื่อเริ่มต้นการตรวจจับวัตถุจากเว็บแคมของคุณ</p>
           </div>""", 
        unsafe_allow_html=True
    )

# ปุ่มควบคุมการทำงาน
col1, col2 = st.columns(2)

with col1:
    # เพิ่มปุ่มเริ่มตรวจจับ
    start_button = st.button("▶️ เริ่มตรวจจับ", key="start_button", on_click=start_detection, 
                             disabled=st.session_state.running, use_container_width=True)

with col2:
    # เพิ่มปุ่มหยุดการตรวจจับ
    stop_button = st.button("⏹️ หยุดการตรวจจับ", key="stop_button", on_click=stop_detection, 
                           disabled=not st.session_state.running, use_container_width=True)

# เริ่มการตรวจจับถ้าสถานะกำลังทำงาน
if st.session_state.running:
    detect_objects()

# Footer
st.markdown(
    """
    <footer>
        <p>© 2025 AI ตรวจจับวัตถุเว็บแคม - ใช้ YOLOv8 และ Streamlit</p>
    </footer>
    """, 
    unsafe_allow_html=True
)