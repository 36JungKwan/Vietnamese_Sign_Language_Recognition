import cv2
import asyncio
import websockets
import json
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

async def stream_webcam():
    uri = "ws://localhost:8000/ws/stream"
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở Webcam!")
        return
        
    print("⏳ Đang kết nối tới VSL Server...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Đã kết nối! Bắt đầu truyền hình ảnh...")
            
            async def receive_results():
                while True:
                    try:
                        response = await websocket.recv()
                        data = json.loads(response)
                        if data.get("type") == "final_sentence":
                            print(f"\n🟢 [LLM Dịch]: {data['text']}\n")
                    except websockets.ConnectionClosed:
                        print("\n⚠️ Nhận tín hiệu Server đã đóng kết nối.")
                        break

            recv_task = asyncio.create_task(receive_results())
            
            # Khởi tạo MediaPipe trên Client để vẽ UI
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.05)
                        continue
                    
                    # 1. Tạo một bản sao ảnh nguyên gốc để gửi cho Server AI
                    frame_to_send = frame.copy()
                    
                    # 2. Xử lý và vẽ khung xương lên ảnh hiện tại (Chỉ để hiển thị cho bạn xem)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
                    cv2.imshow("VSL Client (Live Debug)", frame)
                    
                    # 3. Nén và gửi ảnh GỐC (frame_to_send) lên server
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode('.jpg', frame_to_send, encode_param)
                    await websocket.send(buffer.tobytes())
                    
                    await asyncio.sleep(0.03) # Giữ tốc độ ~30 FPS
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            recv_task.cancel()
            
    except Exception as e:
        print(f"❌ Lỗi Client: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(stream_webcam())