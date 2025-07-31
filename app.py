from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import os
import logging
from typing import Optional
import uvicorn
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Chatbot API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    status: str

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection"""
    # Remove control characters and excessive whitespace
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Limit length to prevent abuse
    return text[:1000]

class MedicalChatbot:
    def __init__(self, model_path: str = "./medical_chatbot_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.max_seq_length = 2048
        self.dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned medical chatbot model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            if not os.path.exists(self.model_path) or not os.path.isdir(self.model_path):
                logger.warning(f"Model path {self.model_path} not found or invalid. Using base model.")
                model_name = "unsloth/tinyllama-1.1b-bnb-4bit"
            else:
                model_name = self.model_path
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load model")
    
    def create_medical_prompt(self, instruction: str, input_text: str = ""):
        """Create a standardized prompt format for medical conversations"""
        instruction = sanitize_input(instruction)
        input_text = sanitize_input(input_text)
        if input_text:
            prompt = f"""Below is a medical instruction that describes a task, paired with input. Write an appropriate response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""Below is a medical instruction. Write an appropriate response.

### Instruction:
{instruction}

### Response:
"""
        return prompt
    
    def generate_response(self, message: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the medical chatbot"""
        try:
            # Validate inputs
            max_length = min(max_length, 1000)  # Cap max_length to prevent abuse
            temperature = max(0.1, min(temperature, 1.0))  # Clamp temperature
            
            # Create prompt
            prompt = self.create_medical_prompt(message)
            
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
            return "I apologize, but I encountered an error processing your request. Please try again."

# Initialize the chatbot
chatbot = MedicalChatbot()

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
                <p>Frontend not found. Please ensure index.html is in the static folder.</p>
                <p>API is available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """, status_code=404)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for the medical chatbot"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate response
        response = chatbot.generate_response(
            request.message, 
            request.max_length, 
            request.temperature
        )
        
        # Conditionally add disclaimer
        disclaimer = os.environ.get("ADD_DISCLAIMER", "true").lower() == "true"
        if disclaimer:
            response += "\n\n**Disclaimer: This is an AI assistant and should not replace professional medical advice. Always consult healthcare professionals for medical concerns.**"
        
        return ChatResponse(response=response, status="success")
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": chatbot.model is not None}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_path": chatbot.model_path,
        "max_seq_length": chatbot.max_seq_length,
        "dtype": str(chatbot.dtype),
        "cuda_available": torch.cuda.is_available()
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
