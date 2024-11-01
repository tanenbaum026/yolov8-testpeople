import cv2
import time
from ultralytics import YOLO
from datetime import datetime, timedelta
import os
import threading
from queue import Queue
import logging
from aligo import Aligo

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
VIDEO_SAVE_DIR = "captured_videos"
PERSON_CONFIDENCE_THRESHOLD = 0.5
UPLOAD_INTERVAL = 3600  # 上传检查间隔（秒）

class VideoProcessor:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.video_writer = None
        self.recording = False
        self.last_person_time = None
        self.current_video_path = None
        self.rtsp_url = "rtsp://admin:RGJNSD@192.168.6.120/mpeg4/ch1/sub/av_stream"
        self.reconnect_interval = 5  # 重连间隔（秒）
        self.max_retries = 3  # 单次重连最大尝试次数
        self.enable_night_mode = True  # 添加夜间模式开关
        self.night_start_hour = 18  # 夜间模式开始时间
        self.night_end_hour = 6  # 夜间模式结束时间

    def connect_to_stream(self):
        """尝试连接到视频流"""
        retries = 0
        while retries < self.max_retries:
            cap = cv2.VideoCapture(self.rtsp_url)
            if cap.isOpened():
                logging.info("成功连接到视频流")
                return cap
            retries += 1
            logging.warning(f"连接失败，第 {retries} 次重试...")
            time.sleep(self.reconnect_interval)
        return None

    def enhance_night_image(self, frame):
        """增强夜间图像"""
        try:
            # 转换到LAB色彩空间
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 对亮度通道进行CLAHE处理
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # 合并通道
            enhanced_lab = cv2.merge((l, a, b))

            # 转换回BGR色彩空间
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # 可选：增加亮度和对比度
            alpha = 1.3  # 对比度
            beta = 30  # 亮度
            enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=alpha, beta=beta)

            return enhanced_frame
        except Exception as e:
            logging.error(f"图像增强处理失败: {str(e)}")
            return frame

    def is_night_time(self):
        """判断当前是否是夜间"""
        current_hour = datetime.now().hour
        return (current_hour >= self.night_start_hour or
                current_hour < self.night_end_hour)

    def process_video_stream(self):
        os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

        while True:
            try:
                cap = self.connect_to_stream()
                if cap is None:
                    logging.error("无法连接到视频流，等待后重试")
                    time.sleep(self.reconnect_interval)
                    continue

                consecutive_failures = 0
                while True:
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            consecutive_failures += 1
                            logging.warning(f"读取帧失败，连续失败次数: {consecutive_failures}")

                            if consecutive_failures >= 3:
                                logging.error("视频流中断，准备重新连接")
                                self.stop_recording()
                                cap.release()
                                break

                            time.sleep(1)
                            continue

                        consecutive_failures = 0

                        # 夜间模式图像增强
                        if self.enable_night_mode and self.is_night_time():
                            frame = self.enhance_night_image(frame)

                        # 使用YOLOv8进行检测
                        results = self.model(frame, conf=0.3)  # 降低夜间检测阈值

                        # 检查是否检测到人
                        person_detected = False
                        for result in results:
                            for box in result.boxes:
                                if box.cls == 0:  # 人类类别
                                    conf = float(box.conf)
                                    # 根据时间调整置信度阈值
                                    threshold = (PERSON_CONFIDENCE_THRESHOLD * 0.6
                                                 if self.is_night_time()
                                                 else PERSON_CONFIDENCE_THRESHOLD)
                                    if conf > threshold:
                                        person_detected = True
                                        # 在图像上标注检测框和置信度
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(frame, f'Person {conf:.2f}',
                                                    (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.9, (0, 255, 0), 2)
                                        break

                        if person_detected:
                            self.handle_person_detection(frame)
                        else:
                            self.handle_no_person()

                    except Exception as e:
                        logging.error(f"处理视频帧时发生错误: {str(e)}")
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            self.stop_recording()
                            cap.release()
                            break
                        time.sleep(1)

            except Exception as e:
                logging.error(f"视频处理主循环发生错误: {str(e)}")
                time.sleep(self.reconnect_interval)
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                self.stop_recording()

    def handle_person_detection(self, frame):
        current_time = datetime.now()

        if not self.recording:
            # 开始新的录制
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            self.current_video_path = os.path.join(VIDEO_SAVE_DIR, f"{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                self.current_video_path,
                fourcc,
                20.0,
                (frame.shape[1], frame.shape[0])
            )
            self.recording = True

        self.last_person_time = current_time
        self.video_writer.write(frame)

    def handle_no_person(self):
        if self.recording and self.last_person_time:
            # 如果超过5秒没有检测到人，停止录制
            if (datetime.now() - self.last_person_time).seconds > 5:
                self.stop_recording()

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
        self.recording = False
        self.video_writer = None
        self.last_person_time = None


class OSSUploader:
    def __init__(self):
        self.ali = Aligo()  # 在初始化时创建 Aligo 实例
        self.cloud_folder_name = '家中视频监控'  # 云盘目标文件夹名称
        self.ensure_cloud_folder()

    def ensure_cloud_folder(self):
        """确保云盘文件夹存在"""
        try:
            remote_folder = self.ali.get_folder_by_path(self.cloud_folder_name)
            if remote_folder is None:
                remote_folder = self.ali.create_folder(self.cloud_folder_name)
                if remote_folder is None:
                    raise RuntimeError('无法创建云盘文件夹')
                logging.info(f"已创建云盘文件夹: {self.cloud_folder_name}")
            self.remote_folder = remote_folder
        except Exception as e:
            logging.error(f"确保云盘文件夹存在时发生错误: {str(e)}")
            raise RuntimeError('确保云盘文件夹存在时发生错误')

    def upload_videos(self):
        while True:
            try:
                current_time = datetime.now()
                seven_days_ago = current_time - timedelta(days=7)

                # 遍历视频目录
                for filename in os.listdir(VIDEO_SAVE_DIR):
                    if not filename.endswith('.avi'):
                        continue

                    file_path = os.path.join(VIDEO_SAVE_DIR, filename)
                    file_time = datetime.strptime(filename.split('.')[0], "%Y%m%d_%H%M%S")

                    # 检查文件是否在7天内
                    if file_time < seven_days_ago:
                        # 可选：删除超过7天的文件
                        os.remove(file_path)
                        logging.info(f"删除过期文件: {filename}")

                    # 上传文件
                if os.path.exists(VIDEO_SAVE_DIR) and os.listdir(VIDEO_SAVE_DIR):
                   logging.info("开始上传文件夹...")
                   self.ali.upload_folder(VIDEO_SAVE_DIR, self.remote_folder.file_id,None,check_name_mode='overwrite')
                   logging.info("文件夹上传完成")

            except Exception as e:
                logging.error(f"上传过程中发生错误: {str(e)}")

            # 等待一段时间后再次检查
            time.sleep(UPLOAD_INTERVAL)


def main():

    # 创建并启动视频处理线程
    video_processor = VideoProcessor()
    video_thread = threading.Thread(target=video_processor.process_video_stream)  # 修改这里的方法名
    video_thread.daemon = True
    video_thread.start()

    # 创建并启动上传线程
    uploader = OSSUploader()
    upload_thread = threading.Thread(target=uploader.upload_videos)
    upload_thread.daemon = True
    upload_thread.start()

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("程序正在退出...")


if __name__ == "__main__":
    main()