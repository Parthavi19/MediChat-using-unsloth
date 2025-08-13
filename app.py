from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import logging
from typing import Optional
import uvicorn
import re
import asyncio
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Chatbot API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    status: str

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection"""
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]

class MedicalChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.loading = False
        
    def load_model_async(self):
        """Load model in background thread"""
        if self.loading or self.model_loaded:
            return
            
        self.loading = True
        try:
            logger.info("Starting model loading in background...")
            
            # Import here to avoid blocking startup
            import torch
            from unsloth import FastLanguageModel
            from unsloth import is_bfloat16_supported
            
            model_path = "./medical_chatbot_model"
            max_seq_length = 2048
            dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
            
            # Use base model if fine-tuned model not available
            if not os.path.exists(model_path) or not os.path.isdir(model_path):
                logger.warning(f"Fine-tuned model not found. Using base model.")
                model_name = "unsloth/tinyllama-1.1b-bnb-4bit"
            else:
                model_name = model_path
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
        finally:
            self.loading = False
    
    def generate_response(self, message: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the medical chatbot"""
        if not self.model_loaded:
            return self.get_fallback_response(message)
            
        try:
            import torch
            
            # Validate inputs
            max_length = min(max_length, 500)
            temperature = max(0.1, min(temperature, 1.0))
            
            # Create prompt
            prompt = f"""Below is a medical instruction. Write an appropriate response.

### Instruction:
{sanitize_input(message)}

### Response:
"""
            
            # Tokenize input
            inputs = self.tokenizer([prompt], return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in full_response:
                response = full_response.split("### Response:")[-1].strip()
            else:
                response = full_response.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self.get_fallback_response(message)
    
    def get_fallback_response(self, message: str) -> str:
        """Provide rule-based responses when model is not available"""
        message_lower = message.lower()
        
        # Medical keywords and responses
        if any(keyword in message_lower for keyword in ['fever', 'temperature', 'hot']):
            return "For fever management: Stay hydrated, rest, and consider fever reducers like acetaminophen or ibuprofen as directed. Seek medical attention if fever exceeds 103°F or persists."
        
        elif any(keyword in message_lower for keyword in ['headache', 'head pain']):
            return "For headaches: Try rest in a quiet, dark room, stay hydrated, and consider over-the-counter pain relievers. Seek medical care for severe, sudden, or recurring headaches."
        
        elif any(keyword in message_lower for keyword in ['diabetes', 'blood sugar']):
            return "Common diabetes symptoms include frequent urination, excessive thirst, unexplained weight loss, and fatigue. Proper management involves diet, exercise, medication, and regular monitoring."
        
        elif any(keyword in message_lower for keyword in ['blood pressure', 'hypertension']):
            return "High blood pressure often has no symptoms but can lead to serious complications. Management includes a healthy diet, regular exercise, limiting sodium, and taking prescribed medications."
        
        elif any(keyword in message_lower for keyword in ['chest pain', 'heart']):
            return "Chest pain can have various causes. Seek immediate medical attention for severe, crushing chest pain, especially with shortness of breath, sweating, or nausea."
        
        else:
            return "I'm currently loading my advanced medical knowledge base. In the meantime, for any serious symptoms or concerns, please consult with a healthcare professional immediately."

# Initialize the chatbot
chatbot = MedicalChatbot()

@app.on_event("startup")
async def startup_event():
    """Start model loading in background after server starts"""
    def start_model_loading():
        time.sleep(2)  # Give server time to start
        chatbot.load_model_async()
    
    threading.Thread(target=start_model_loading, daemon=True).start()

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Create index.html if it doesn't exist
if not os.path.exists("static/index.html"):
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Medical Chatbot Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 20px; }
        .disclaimer { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        .chat-messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; background: #fafafa; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background: #007bff; color: white; text-align: right; }
        .bot-message { background: #e9ecef; color: black; }
        .input-group { display: flex; gap: 10px; }
        #messageInput { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        #sendButton { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        #sendButton:hover { background: #0056b3; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Chatbot Assistant</h1>
            <p>Your AI-powered health information companion</p>
        </div>
        
        <div class="disclaimer">
            ⚠️ This chatbot provides general medical information only. Always consult healthcare professionals for medical advice, diagnosis, or treatment.
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                👋 Hello! I'm your medical information assistant. I can help answer general health questions. How can I assist you today?
            </div>
        </div>

        <div class="input-group">
            <input type="text" id="messageInput" placeholder="Ask about symptoms, conditions, medications..." />
            <button id="sendButton">Send</button>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = content;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            addMessage(message, true);
            messageInput.value = '';
            
            addMessage('Thinking...', false);

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();
                
                // Remove "Thinking..." message
                chatMessages.removeChild(chatMessages.lastChild);
                
                if (data.status === 'success') {
                    addMessage(data.response, false);
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }
            } catch (error) {
                chatMessages.removeChild(chatMessages.lastChild);
                addMessage('Connection error. Please check your internet connection.', false);
            }
        }

        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>'''
    
    with open("static/index.html", "w") as f:
        f.write(html_content)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Medical Chatbot API</h1>
                <p>API is available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """, status_code=200)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for the medical chatbot"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate response (will use fallback if model not loaded)
        response = chatbot.generate_response(
            request.message, 
            request.max_length, 
            request.temperature
        )
        
        # Add disclaimer
        disclaimer = "\n\n**Disclaimer: This is an AI assistant and should not replace professional medical advice. Always consult healthcare professionals for medical concerns.**"
        response += disclaimer
        
        return ChatResponse(response=response, status="success")
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": chatbot.model_loaded,
        "model_loading": chatbot.loading
    }

@app.get("/model-status")
async def model_status():
    """Get model loading status"""
    return {
        "loaded": chatbot.model_loaded,
        "loading": chatbot.loading
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        workers=1
    )
