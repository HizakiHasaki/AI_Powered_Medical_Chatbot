from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
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

app = FastAPI()

# Mount thư mục static nếu tồn tại
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.warning(f"Directory '{static_dir}' does not exist. Static files will not be served.")

# Đường dẫn đến templates
templates = Jinja2Templates(directory="templates")

# Cấu hình OpenAI API
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# Lưu trữ lịch sử hội thoại (tạm thời trong bộ nhớ)
chat_history: Dict[str, List[Dict]] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(None),
    query: str = Form(...),
    session_id: str = Form(default="default")
):
    try:
        # Nếu có ảnh, xử lý ảnh
        encoded_image = None
        if image:
            image_content = await image.read()
            if not image_content:
                raise HTTPException(status_code=400, detail="Empty image file")
            if len(image_content) > 5 * 1024 * 1024:  # Giới hạn 5MB
                raise HTTPException(status_code=400, detail="Image size exceeds 5MB limit")

            encoded_image = base64.b64encode(image_content).decode("utf-8")
            try:
                img = Image.open(io.BytesIO(image_content))
                img.verify()
            except Exception as e:
                logger.error(f"Invalid image format: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Khởi tạo lịch sử hội thoại nếu chưa có
        if session_id not in chat_history:
            chat_history[session_id] = []

        # Thêm câu hỏi của người dùng vào lịch sử
        user_message = {"role": "user", "content": [{"type": "text", "text": query}]}
        if encoded_image:
            user_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
        chat_history[session_id].append(user_message)

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
        ] + chat_history[session_id]

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
            logger.info(f"Processed diagnosis: {answer[:100]}...")

            # Thêm phản hồi của AI vào lịch sử
            chat_history[session_id].append({"role": "assistant", "content": answer})

            # Trả về phản hồi (không kèm ảnh)
            return JSONResponse(status_code=200, content={"diagnosis": answer})
        else:
            error_detail = response.json().get("error", {}).get("message", response.text)
            logger.error(f"Error from OpenAI API: {response.status_code} - {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=f"Error from OpenAI API: {error_detail}")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)