import os
import re
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Định nghĩa cấu trúc JSON ép buộc LLM phải trả về
class SignLanguageTranslation(BaseModel):
    sentence: str
    used_words: list[str]

# 2. Bộ kiểm duyệt từ vựng độc lập (Validator)
class ConstraintValidator:
    @staticmethod
    def normalize_word(word: str) -> str:
        """Chuẩn hóa từ: chuyển chữ thường, loại bỏ dấu câu dính kèm."""
        word = word.lower().strip()
        # Loại bỏ các dấu câu cơ bản ở đầu/cuối từ
        word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        return word

    @staticmethod
    def validate(input_words: list[str], used_words: list[str]) -> bool:
        """
        Kiểm tra xem tập từ vựng LLM sử dụng có nằm hoàn toàn trong tập từ input không.
        Tuyệt đối không cho phép sinh thêm từ mang ngữ nghĩa mới.
        """
        input_set = set(ConstraintValidator.normalize_word(w) for w in input_words)
        used_set = set(ConstraintValidator.normalize_word(w) for w in used_words)
        
        # Kiểm tra used_set có phải là tập con (subset) của input_set không
        is_valid = used_set.issubset(input_set)
        
        if not is_valid:
            illegal_words = used_set - input_set
            print(f"[Validator] TỪ CHỐI: LLM đã ảo giác (thêm từ mới): {illegal_words}")
            
        return is_valid

# 3. LLM Agent Tích hợp Gemini 3.5 Flash
class LLMAgent:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash"):
        # Ưu tiên lấy API key từ tham số, nếu không có thì lấy từ biến môi trường
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GEMINI_API_KEY. Vui lòng cung cấp qua tham số hoặc biến môi trường.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.validator = ConstraintValidator()
        
        self.system_instruction = """
        Bạn là bộ hiệu chỉnh câu tiếng Việt từ các từ được nhận diện bởi hệ thống ngôn ngữ ký hiệu.

        QUY TẮC NGHIÊM NGẶT:
        1. CHỈ sử dụng các từ/cụm từ có trong mảng đầu vào.
        2. Được phép thay đổi thứ tự từ để tạo câu tiếng Việt tự nhiên.
        3. Được phép thêm dấu câu (., ?, !) và viết hoa chữ cái đầu.
        4. KHÔNG được thêm từ mang ý nghĩa mới (đặc biệt cấm thêm trạng từ, tính từ, đại từ không có trong input).
        5. KHÔNG suy đoán ý định của người dùng.
        6. Nếu input đã là câu hợp lệ, giữ nguyên nội dung.
        
        Bạn phải phân tách rõ ràng câu hoàn chỉnh (sentence) và danh sách các từ gốc bạn đã chọn sử dụng (used_words).
        """

    def process_sentence(self, input_words: list[str], max_retries: int = 1) -> str:
        """
        Gửi chuỗi từ thô lên Gemini và xác thực kết quả.
        Nếu thất bại (ảo giác), thử lại. Nếu vẫn thất bại, fallback về raw text.
        """
        if not input_words:
            return ""

        prompt = f"Input: {input_words}"
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        response_mime_type="application/json",
                        response_schema=SignLanguageTranslation,
                        temperature=0.1, # Nhiệt độ cực thấp để model không sáng tạo lung tung
                    ),
                )
                
                # Gemini SDK tự động parse JSON thành object (được định nghĩa bởi response_schema)
                # Tuy nhiên, nếu model trả text thuần (dù hiếm), ta cần extract JSON.
                # Ở SDK mới, cấu trúc trả về phụ thuộc kiểu gọi, giả định ta parse từ text
                import json
                result_data = json.loads(response.text)
                
                llm_sentence = result_data.get("sentence", "")
                llm_used_words = result_data.get("used_words", [])
                
                # Xác thực (Validation)
                if self.validator.validate(input_words, llm_used_words):
                    return llm_sentence
                else:
                    print(f"[LLM Agent] Lần thử {attempt + 1} thất bại do vi phạm ngữ nghĩa.")
                    
            except Exception as e:
                print(f"[LLM Agent] Lỗi gọi API ở lần thử {attempt + 1}: {e}")
                
        # --- FALLBACK ---
        # Nếu LLM liên tục ảo giác hoặc sập API, trả về chuỗi thô (ghép bằng dấu cách) 
        # để đảm bảo pipeline Real-time không bao giờ bị nghẽn (Fail-safe).
        print("[LLM Agent] Kích hoạt Fallback: Trả về chuỗi từ thô.")
        fallback_sentence = " ".join(input_words).capitalize() + "."
        return fallback_sentence