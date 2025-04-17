import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load biến môi trường
load_dotenv()

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# Lưu trữ lịch sử hội thoại (tạm thời trong bộ nhớ)
chat_history: List[Dict] = []

def process_image(image_path: str, query: str) -> Dict:
    try:
        # Kiểm tra file ảnh có tồn tại không
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"error": f"Image file not found: {image_path}"}

        # Đọc và mã hóa ảnh
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            if len(image_content) > 5 * 1024 * 1024:  # Giới hạn 5MB
                logger.error("Image size exceeds 5MB limit")
                return {"error": "Image size exceeds 5MB limit"}

            encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Xác thực định dạng ảnh
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}

        # Thêm câu hỏi của người dùng vào lịch sử
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
        chat_history.append(user_message)

        # Chuẩn bị tin nhắn cho OpenAI API
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced medical doctor with over 20 years of clinical practice. "
                    "Your sole purpose is to assist with medical-related questions and concerns. "
                    "Provide detailed, accurate, and professional responses based on the patient's input and conversation history. "
                    "If the input is vague, ask clarifying questions instead of guessing. "
                    "If the question is unrelated to medicine or health (e.g., math, weather, general knowledge), politely decline to answer and redirect the user to ask a medical question. "
                    "Avoid definitive diagnoses; suggest possible conditions and next steps. "
                    "Use a calm, empathetic, and authoritative tone. "
                    "Check for urgent symptoms (e.g., chest pain, difficulty breathing, severe bleeding) and prioritize emergency advice if detected. "
                    "Always conclude with: 'Please note, this information is for educational purposes only and does not replace a formal medical evaluation. Consult a healthcare professional for an accurate diagnosis and treatment plan.'"
                )
            }
        ] + chat_history

        # Kiểm tra mức độ khẩn cấp
        urgent_keywords = ["chest pain", "difficulty breathing", "severe", "bleeding", "unconscious"]
        is_urgent = any(keyword in query.lower() for keyword in urgent_keywords)
        if is_urgent:
            messages.append({
                "role": "system",
                "content": "This may indicate a medical emergency. Prioritize advising immediate action."
            })

        # Gửi yêu cầu tới OpenAI API
        response = requests.post(
            OPENAI_API_URL,
            json={
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 1500
            },
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response from GPT-4o API: {answer[:100]}...")

            # Thêm phản hồi của AI vào lịch sử
            chat_history.append({"role": "assistant", "content": answer})

            return {"gpt-4o": answer}
        else:
            error_detail = response.json().get("error", {}).get("message", response.text)
            logger.error(f"Error from OpenAI API: {response.status_code} - {error_detail}")
            return {"error": f"Error from OpenAI API: {error_detail}"}

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def process_image_no_image(query: str) -> Dict:
    """Xử lý câu hỏi không có ảnh."""
    try:
        # Thêm câu hỏi của người dùng vào lịch sử
        chat_history.append({"role": "user", "content": [{"type": "text", "text": query}]})

        # Chuẩn bị tin nhắn cho OpenAI API
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced medical doctor with over 20 years of clinical practice. "
                    "Your sole purpose is to assist with medical-related questions and concerns. "
                    "Provide detailed, accurate, and professional responses based on the patient's input and conversation history. "
                    "If the input is vague, ask clarifying questions instead of guessing. "
                    "If the question is unrelated to medicine or health (e.g., math, weather, general knowledge), politely decline to answer and redirect the user to ask a medical question. "
                    "Avoid definitive diagnoses; suggest possible conditions and next steps. "
                    "Use a calm, empathetic, and authoritative tone. "
                    "Check for urgent symptoms (e.g., chest pain, difficulty breathing, severe bleeding) and prioritize emergency advice if detected. "
                    "Always conclude with: 'Please note, this information is for educational purposes only and does not replace a formal medical evaluation. Consult a healthcare professional for an accurate diagnosis and treatment plan.'"
                )
            }
        ] + chat_history

        # Kiểm tra mức độ khẩn cấp
        urgent_keywords = ["chest pain", "difficulty breathing", "severe", "bleeding", "unconscious"]
        is_urgent = any(keyword in query.lower() for keyword in urgent_keywords)
        if is_urgent:
            messages.append({
                "role": "system",
                "content": "This may indicate a medical emergency. Prioritize advising immediate action."
            })

        # Gửi yêu cầu tới OpenAI API
        response = requests.post(
            OPENAI_API_URL,
            json={
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 1500
            },
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response from GPT-4o API: {answer[:100]}...")

            # Thêm phản hồi của AI vào lịch sử
            chat_history.append({"role": "assistant", "content": answer})

            return {"gpt-4o": answer}
        else:
            error_detail = response.json().get("error", {}).get("message", response.text)
            logger.error(f"Error from OpenAI API: {response.status_code} - {error_detail}")
            return {"error": f"Error from OpenAI API: {error_detail}"}

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def interactive_chat():
    """Chạy chatbot trong chế độ tương tác qua terminal."""
    print("Welcome to the AI Medical Diagnostic Chatbot!")
    print("This chatbot is designed for medical questions only. Enter your symptoms or ask a question (type 'exit' to quit).")
    image_path = None

    while True:
        # Yêu cầu người dùng nhập đường dẫn ảnh (nếu muốn)
        if not image_path:
            img_input = input("Enter image path (or press Enter to skip): ").strip()
            if img_input:
                image_path = img_input
                if not os.path.exists(image_path):
                    print(f"Error: Image file '{image_path}' not found.")
                    image_path = None
                    continue

        # Nhập câu hỏi
        query = input("Your question/symptoms: ").strip()
        if query.lower() == "exit":
            break

        # Xử lý và in phản hồi
        result = process_image(image_path, query) if image_path else process_image_no_image(query)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Dr. AI: {result['gpt-4o']}")

if __name__ == "__main__":
    # Chạy chế độ tương tác qua terminal
    interactive_chat()