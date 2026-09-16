import numpy as np
from enum import Enum
from typing import List, Tuple, Optional

class State(Enum):
    IDLE = 0
    SIGNING = 1

class TemporalDecoder:
    def __init__(
        self, 
        idx_to_class: dict, 
        alpha: float = 0.5, 
        conf_thresh: float = 0.8, 
        motion_thresh: float = 0.008,
        end_frames_thresh: int = 40,
        cooldown_frames: int = 20  # THÊM THÔNG SỐ NÀY (Khoảng 0.6 giây)
    ):
        self.idx_to_class = idx_to_class
        self.alpha = alpha
        self.conf_thresh = conf_thresh
        self.motion_thresh = motion_thresh
        self.end_frames_thresh = end_frames_thresh
        self.cooldown_frames = cooldown_frames
        
        # State & Buffers
        self.state = State.IDLE
        self.smoothed_probs = None
        self.sentence_buffer = []
        
        # Tracking variables
        self.idle_counter = 0
        self.last_emitted_word = None
        self.is_new_segment = True 
        self.current_cooldown = 0  # BỘ ĐẾM COOLDOWN

    def reset(self):
        self.state = State.IDLE
        self.smoothed_probs = None
        self.sentence_buffer = []
        self.idle_counter = 0
        self.last_emitted_word = None
        self.is_new_segment = True
        self.current_cooldown = 0

    def process_window(
        self, 
        raw_probs: np.ndarray, 
        motion_energy: float, 
        hand_detected: bool
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        
        new_word = None
        completed_sentence = None

        # 1. Probability Smoothing
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs
        else:
            self.smoothed_probs = self.alpha * raw_probs + (1 - self.alpha) * self.smoothed_probs

        max_idx = np.argmax(self.smoothed_probs)
        max_prob = self.smoothed_probs[max_idx]
        current_word = self.idx_to_class.get(max_idx, "")

        # 2. Physical State
        is_moving = motion_energy > self.motion_thresh
        is_confident = max_prob > self.conf_thresh

        # --- TRỪ DẦN COOLDOWN ---
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        # --- STATE MACHINE ---
        if self.state == State.IDLE:
            if is_moving or (is_confident and hand_detected):
                self.state = State.SIGNING
                self.idle_counter = 0
                self.is_new_segment = True
                
        elif self.state == State.SIGNING:
            # Chỉ cho phép nhận từ nếu ĐÃ HẾT COOLDOWN
            if is_confident and self.current_cooldown == 0:
                if current_word != self.last_emitted_word or self.is_new_segment:
                    new_word = current_word
                    self.sentence_buffer.append(new_word)
                    self.last_emitted_word = new_word
                    self.is_new_segment = False 
                    
                    # BẬT COOLDOWN NGAY KHI VỪA NHẬN TỪ XONG
                    self.current_cooldown = self.cooldown_frames
            
            # Đánh dấu nhịp múa chùng xuống
            if not is_confident and not is_moving:
                self.is_new_segment = True 

            # Điều kiện ngắt câu (END)
            if not hand_detected or (not is_moving and not is_confident):
                self.idle_counter += 1
            else:
                self.idle_counter = 0 

            # Kích hoạt ngắt câu
            if self.idle_counter >= self.end_frames_thresh:
                if len(self.sentence_buffer) > 0:
                    completed_sentence = self.sentence_buffer.copy()
                self.reset() 

        return new_word, completed_sentence