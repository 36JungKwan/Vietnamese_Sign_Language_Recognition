import cv2
import asyncio
import websockets
import json
import unicodedata
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để vẽ bằng cv2.putText không bị lỗi font."""
    if not input_str:
        return ""
    nfkd = unicodedata.normalize('NFKD', input_str)
    res = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return res.replace('đ', 'd').replace('Đ', 'D')

async def stream_webcam():
    uri = "ws://localhost:8000/ws/stream"
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở Webcam!")
        return
        
    print("⏳ Đang kết nối tới VSL Server...")
    
    # Bộ lưu trạng thái giao diện Client
    state = {
        "is_recording": True,
        "current_word": "",
        "recent_words": [],
        "final_sentence": ""
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Đã kết nối tới VSL Server!")
            print("💡 [HƯỚNG DẪN]:")
            print("   - Nhấn phím SPACE trên cửa sổ Camera để BẬT / TẮT nhận diện.")
            print("   - Nhấn phím 'q' trên cửa sổ Camera để thoát.")
            print("-------------------------------------------------------")
            
            async def receive_results():
                while True:
                    try:
                        response = await websocket.recv()
                        data = json.loads(response)
                        msg_type = data.get("type")
                        
                        if msg_type == "status":
                            state["is_recording"] = data.get("is_recording", True)
                            status_txt = "🟢 BẬT" if state["is_recording"] else "🔴 TẮT"
                            print(f"[Trạng thái] Nhận diện: {status_txt}")
                            
                        elif msg_type == "word":
                            w = data.get("word", "")
                            state["current_word"] = w
                            state["recent_words"].append(w)
                            if len(state["recent_words"]) > 6:
                                state["recent_words"].pop(0)
                            print(f"⚡ [AI Đoán Từ]: {w}")
                            
                        elif msg_type == "sentence_start":
                            raw_words = data.get("words", [])
                            print(f"⏳ [Ngắt câu - Đang gửi LLM]: {' '.join(raw_words)}...")
                            
                        elif msg_type == "final_sentence":
                            txt = data.get("text", "")
                            state["final_sentence"] = txt
                            state["recent_words"] = []
                            print(f"\n🟢 [LLM Dịch Hoàn Chỉnh]: {txt}\n")
                            
                    except websockets.ConnectionClosed:
                        print("\n⚠️ Nhận tín hiệu Server đã đóng kết nối.")
                        break
                    except Exception as e:
                        print(f"⚠️ [Lỗi nhận gói tin]: {e}")

            recv_task = asyncio.create_task(receive_results())
            
            # Khởi tạo MediaPipe trên Client để vẽ UI khung xương
            TARGET_FPS = 18
            FRAME_INTERVAL = 1.0 / TARGET_FPS

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    loop_start = asyncio.get_event_loop().time()
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.05)
                        continue
                    
                    # 1. Tối ưu kích thước ảnh gửi lên Server AI (chuẩn hóa về max width 640px)
                    if frame.shape[1] > 640:
                        scale_factor = 640.0 / frame.shape[1]
                        new_h = int(frame.shape[0] * scale_factor)
                        frame_to_send = cv2.resize(frame, (640, new_h))
                    else:
                        frame_to_send = frame.copy()
                    
                    # 2. Xử lý và vẽ khung xương lên ảnh hiện tại
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
                    h, w, _ = frame.shape

                    # 3. Vẽ Banner Trạng thái trên đầu màn hình (Top Banner)
                    if state["is_recording"]:
                        cv2.rectangle(frame, (0, 0), (w, 40), (0, 160, 0), -1)
                        banner_text = "[REC] DANG NHAN DIEN - Nhan SPACE de tam dung"
                    else:
                        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
                        banner_text = "[PAUSED] TAM DUNG - Nhan SPACE de bat dau"
                        
                    cv2.putText(frame, banner_text, (15, 26), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                    # 3.1. Cảnh báo góc nhìn & Khoảng cách (Ý 4): Kiểm tra 2 vai có nằm trọn trong khung hình
                    shoulders_ok = True
                    if results.pose_landmarks:
                        l_sh = results.pose_landmarks.landmark[11]
                        r_sh = results.pose_landmarks.landmark[12]
                        if l_sh.visibility < 0.5 or r_sh.visibility < 0.5 or l_sh.x < 0.05 or r_sh.x > 0.95:
                            shoulders_ok = False
                    else:
                        shoulders_ok = False

                    if not shoulders_ok and state["is_recording"]:
                        cv2.rectangle(frame, (0, 42), (w, 72), (0, 140, 255), -1)
                        cv2.putText(frame, "[CANH GOC] Hay lui lai de thay ro 2 vai va nua nguoi tren!", 
                                    (15, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # 4. Vẽ Hộp Thông Tin Kết Quả dưới đáy màn hình (Bottom HUD)
                    cv2.rectangle(frame, (0, h - 85), (w, h), (25, 25, 25), -1)
                    
                    # Dòng 1: Từ vừa nhận diện
                    cur_word_ascii = remove_accents(state["current_word"])
                    word_display = f"Tu vua nhan: {cur_word_ascii}" if cur_word_ascii else "Tu vua nhan: ..."
                    cv2.putText(frame, word_display, (15, h - 52), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Dòng 2: Câu dịch hoàn chỉnh từ LLM (hoặc chuỗi từ gần đây)
                    if state["final_sentence"]:
                        sent_ascii = remove_accents(state["final_sentence"])
                        sent_display = f"Dich: {sent_ascii}"
                    elif state["recent_words"]:
                        recent_ascii = " ".join([remove_accents(x) for x in state["recent_words"]])
                        sent_display = f"Chuoi tu: {recent_ascii}"
                    else:
                        sent_display = "Dich: (Chua co cau moi)"
                        
                    cv2.putText(frame, sent_display, (15, h - 18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2, cv2.LINE_AA)
                    
                    cv2.imshow("VSL Client (Live Debug)", frame)
                    
                    # 5. Nén và gửi ảnh đã tối ưu lên server
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode('.jpg', frame_to_send, encode_param)
                    await websocket.send(buffer.tobytes())
                    
                    # 6. Điều tiết nhịp khung hình ổn định ở mức 18 FPS (Ý 1 & Ý 3)
                    elapsed = asyncio.get_event_loop().time() - loop_start
                    sleep_time = max(0.005, FRAME_INTERVAL - elapsed)
                    await asyncio.sleep(sleep_time)
                    
                    # 7. Bắt phím điều khiển trên cửa sổ OpenCV
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == 32: # Phím Spacebar
                        await websocket.send("toggle_recording")
                        
            recv_task.cancel()
            
    except Exception as e:
        print(f"❌ Lỗi Client: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(stream_webcam())