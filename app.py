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
        """Attempt to load model in background thread (Cloud Run fallback)"""
        if self.loading or self.model_loaded:
            return
            
        self.loading = True
        try:
            logger.info("Attempting to load model for Cloud Run...")
            
            # Try to load a lightweight model if available
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Use a small model that works on CPU
                model_name = "microsoft/DialoGPT-small"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                
                self.model_loaded = True
                logger.info("Lightweight model loaded successfully")
                
            except Exception as model_error:
                logger.warning(f"Could not load model: {model_error}")
                self.model_loaded = False
                
        except Exception as e:
            logger.error(f"Error in model loading: {str(e)}")
            self.model_loaded = False
        finally:
            self.loading = False
    
    def generate_response(self, message: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the medical chatbot or use fallback"""
        
        # Always use fallback for now on Cloud Run for reliability
        return self.get_fallback_response(message)
        
        # Commented out model-based generation for Cloud Run stability
        """
        if not self.model_loaded:
            return self.get_fallback_response(message)
            
        try:
            import torch
            
            # Create a medical-focused prompt
            prompt = f"Medical question: {sanitize_input(message)}\nMedical answer:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + min(max_length, 150),
                    temperature=max(0.1, min(temperature, 1.0)),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            if len(response) < 10:  # If response too short, use fallback
                return self.get_fallback_response(message)
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self.get_fallback_response(message)
        """
    
    def get_fallback_response(self, message: str) -> str:
        """Provide comprehensive rule-based medical responses"""
        message_lower = message.lower()
        
        # Fever and temperature
        if any(keyword in message_lower for keyword in ['fever', 'temperature', 'hot', 'chills']):
            return """For fever management:
            
• **Rest** in a comfortable environment
• **Stay hydrated** with water, clear broths, or electrolyte drinks
• **Fever reducers**: Consider acetaminophen or ibuprofen as directed on packaging
• **Cool measures**: Light clothing, cool compresses, tepid baths
• **Monitor**: Keep track of temperature changes

**Seek immediate medical attention if:**
- Fever exceeds 103°F (39.4°C)
- Fever persists more than 3 days
- Accompanied by severe symptoms like difficulty breathing, chest pain, or severe headache
- Signs of dehydration appear"""

        # Headaches
        elif any(keyword in message_lower for keyword in ['headache', 'head pain', 'migraine']):
            return """For headache relief:
            
• **Rest** in a quiet, dark room
• **Hydration**: Drink plenty of water (dehydration is a common cause)
• **Cold/heat therapy**: Apply cold compress to forehead or warm compress to neck
• **Pain relievers**: Over-the-counter medications like acetaminophen or ibuprofen
• **Relaxation**: Try gentle neck stretches, meditation, or deep breathing

**Seek immediate medical care for:**
- Sudden, severe headache unlike any before
- Headache with fever, stiff neck, vision changes
- Headache after head injury
- Progressive worsening over days/weeks"""

        # Diabetes
        elif any(keyword in message_lower for keyword in ['diabetes', 'blood sugar', 'insulin']):
            return """Diabetes overview:
            
**Common symptoms:**
- Frequent urination and excessive thirst
- Unexplained weight loss or gain
- Extreme fatigue and weakness
- Blurred vision
- Slow-healing wounds
- Frequent infections

**Management strategies:**
- **Diet**: Focus on balanced meals, limit processed sugars
- **Exercise**: Regular physical activity helps control blood sugar
- **Monitoring**: Check blood glucose as recommended
- **Medications**: Take prescribed medications consistently
- **Regular check-ups**: Monitor A1C, blood pressure, cholesterol"""

        # Blood pressure
        elif any(keyword in message_lower for keyword in ['blood pressure', 'hypertension', 'high pressure']):
            return """Blood pressure information:
            
**Understanding readings:**
- Normal: Less than 120/80 mmHg
- Elevated: 120-129 systolic, less than 80 diastolic
- High (Stage 1): 130-139/80-89 mmHg
- High (Stage 2): 140/90 mmHg or higher

**Management approaches:**
- **Diet**: Reduce sodium, increase potassium-rich foods
- **Exercise**: At least 150 minutes moderate activity weekly
- **Weight management**: Maintain healthy BMI
- **Stress reduction**: Practice relaxation techniques
- **Limit alcohol and avoid tobacco**
- **Medication**: Take as prescribed by healthcare provider"""

        # Chest pain and heart
        elif any(keyword in message_lower for keyword in ['chest pain', 'heart', 'cardiac']):
            return """⚠️ **IMPORTANT: Chest pain can be serious**

**Seek immediate emergency care (call 911) if experiencing:**
- Severe, crushing chest pain
- Pain radiating to arm, jaw, or back
- Shortness of breath
- Sweating, nausea, dizziness
- Feeling of impending doom

**Other chest pain causes may include:**
- Muscle strain
- Heartburn/GERD
- Anxiety
- Respiratory issues

**Never ignore chest pain** - when in doubt, seek immediate medical evaluation."""

        # Cold and flu
        elif any(keyword in message_lower for keyword in ['cold', 'flu', 'cough', 'sore throat', 'runny nose']):
            return """Cold and flu care:
            
**Symptom relief:**
- **Rest**: Get plenty of sleep to help immune system
- **Fluids**: Water, warm teas, broths help with hydration
- **Humidifier**: Moist air can ease congestion
- **Salt water gargle**: For sore throat relief
- **Over-the-counter medications**: Follow package directions

**When to see a doctor:**
- Symptoms worsen after initial improvement
- High fever (over 101.3°F) lasting more than 3 days
- Difficulty breathing or chest pain
- Severe headache or sinus pain
- Symptoms lasting more than 10 days"""

        # General wellness
        elif any(keyword in message_lower for keyword in ['wellness', 'healthy', 'prevention', 'diet', 'exercise']):
            return """General wellness tips:
            
**Physical health:**
- **Nutrition**: Eat a balanced diet with fruits, vegetables, whole grains
- **Exercise**: Aim for 150 minutes moderate activity weekly
- **Sleep**: 7-9 hours of quality sleep nightly
- **Hydration**: Drink adequate water throughout the day

**Preventive care:**
- **Regular check-ups**: Annual physical exams
- **Screenings**: Age-appropriate health screenings
- **Vaccinations**: Stay current with recommended vaccines
- **Mental health**: Practice stress management and seek support when needed"""

        # Mental health
        elif any(keyword in message_lower for keyword in ['anxiety', 'depression', 'stress', 'mental health']):
            return """Mental health support:
            
**Managing stress and anxiety:**
- **Deep breathing**: Practice breathing exercises
- **Physical activity**: Regular exercise reduces stress
- **Sleep hygiene**: Maintain consistent sleep schedule
- **Social support**: Connect with friends, family, or support groups
- **Mindfulness**: Try meditation or mindfulness practices

**When to seek professional help:**
- Persistent sadness or anxiety
- Changes in sleep or appetite
- Difficulty functioning in daily activities
- Thoughts of self-harm

**Crisis resources:**
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741"""

        # Medication questions
        elif any(keyword in message_lower for keyword in ['medication', 'medicine', 'drug', 'prescription']):
            return """Medication safety:
            
**General guidelines:**
- **Follow prescriptions**: Take exactly as directed by your healthcare provider
- **Don't share**: Never share prescription medications
- **Storage**: Store medications properly (temperature, moisture, light)
- **Expiration**: Don't use expired medications
- **Interactions**: Inform all healthcare providers of all medications you take

**Questions to ask your pharmacist/doctor:**
- How and when to take the medication
- Possible side effects
- Drug interactions
- What to do if you miss a dose
- How long to take the medication"""

        # Allergies
        elif any(keyword in message_lower for keyword in ['allergy', 'allergic', 'rash', 'hives']):
            return """Allergy management:
            
**Common allergy symptoms:**
- Sneezing, runny nose, congestion
- Itchy, watery eyes
- Skin reactions (rash, hives, eczema)
- Digestive issues (for food allergies)

**Management strategies:**
- **Avoidance**: Identify and avoid triggers when possible
- **Antihistamines**: Over-the-counter options for mild symptoms
- **Environment**: Use air purifiers, wash bedding frequently
- **Food allergies**: Read labels carefully, carry emergency medications if prescribed

**Seek emergency care for:**
- Severe allergic reaction (anaphylaxis)
- Difficulty breathing or swallowing
- Rapid pulse, dizziness, widespread rash"""

        # Default response
        else:
            return """I'm here to help with general medical information. While I can provide educational information about various health topics, I want to emphasize that this should never replace professional medical advice.

**For immediate medical concerns:**
- Call 911 for emergencies
- Contact your healthcare provider
- Visit urgent care or emergency room if needed

**Some topics I can help with:**
- General information about common conditions
- Wellness and prevention tips
- When to seek medical care
- Basic symptom management
- Health maintenance advice

**What specific health topic would you like to know more about?**"""

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
    <title>🏥 Medical Assistant Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container { 
            width: 90%;
            max-width: 900px; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 85vh;
        }
        
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px; 
            text-align: center;
        }
        
        .header h1 { font-size: 1.8em; margin-bottom: 5px; }
        .header p { opacity: 0.9; font-size: 0.9em; }
        
        .disclaimer { 
            background: #fff3cd; 
            border-left: 4px solid #ffc107;
            padding: 15px 20px; 
            margin: 0;
            font-size: 0.85em;
            color: #856404;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-messages { 
            flex: 1;
            overflow-y: auto; 
            padding: 20px; 
            background: #f8f9fa;
            scrollbar-width: thin;
            scrollbar-color: #ccc transparent;
        }
        
        .chat-messages::-webkit-scrollbar { width: 6px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
        
        .message { 
            margin: 15px 0; 
            padding: 15px 20px; 
            border-radius: 18px; 
            max-width: 80%;
            word-wrap: break-word;
            white-space: pre-line;
            line-height: 1.4;
        }
        
        .user-message { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        
        .bot-message { 
            background: white;
            color: #333; 
            border: 1px solid #e9ecef;
            margin-right: auto;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .input-group { 
            padding: 20px; 
            background: white;
            border-top: 1px solid #e9ecef;
            display: flex; 
            gap: 15px;
            align-items: end;
        }
        
        #messageInput { 
            flex: 1; 
            padding: 15px; 
            border: 2px solid #e9ecef; 
            border-radius: 25px; 
            outline: none;
            resize: none;
            min-height: 50px;
            max-height: 120px;
            font-family: inherit;
        }
        
        #messageInput:focus { border-color: #667eea; }
        
        #sendButton { 
            padding: 15px 25px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            height: 50px;
        }
        
        #sendButton:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .typing-indicator {
            display: none;
            padding: 15px 20px;
            color: #666;
            font-style: italic;
        }
        
        .typing-indicator.show { display: block; }
        
        @media (max-width: 768px) {
            .container { width: 95%; height: 90vh; }
            .message { max-width: 90%; }
            .header h1 { font-size: 1.5em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Assistant</h1>
            <p>Your AI-powered health information companion</p>
        </div>
        
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This chatbot provides general medical information for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment. In emergencies, call 911 immediately.
        </div>

        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    👋 <strong>Welcome!</strong> I'm your medical information assistant. I can help answer general health questions, provide wellness tips, and guide you on when to seek medical care.
                    
                    Some topics I can help with:
                    • Common symptoms and conditions
                    • Wellness and prevention advice  
                    • When to seek medical attention
                    • General health information
                    
                    <strong>How can I assist you today?</strong>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                🤔 Thinking about your question...
            </div>
            
            <div class="input-group">
                <textarea id="messageInput" placeholder="Ask about symptoms, conditions, wellness tips..." rows="1"></textarea>
                <button id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = content;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function showTyping() {
            typingIndicator.classList.add('show');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.classList.remove('show');
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Disable input while processing
            sendButton.disabled = true;
            messageInput.disabled = true;

            addMessage(message, true);
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            showTyping();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();
                
                hideTyping();
                
                if (data.status === 'success') {
                    addMessage(data.response, false);
                } else {
                    addMessage('I apologize, but I encountered an error processing your request. Please try again or rephrase your question.', false);
                }
            } catch (error) {
                hideTyping();
                addMessage('I\'m having trouble connecting right now. Please check your internet connection and try again.', false);
            } finally {
                // Re-enable input
                sendButton.disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
            }
        }

        sendButton.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Focus on input when page loads
        messageInput.focus();
    </script>
</body>
</html>'''
    
    with open("static/index.html", "w", encoding='utf-8') as f:
        f.write(html_content)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding='utf-8') as f:
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
        
        # Generate response
        response = chatbot.generate_response(
            request.message, 
            request.max_length, 
            request.temperature
        )
        
        # Add disclaimer
        disclaimer = "\n\n**⚠️ Medical Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical concerns."
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
        "model_loading": chatbot.loading,
        "service": "medical_chatbot_cloud_run"
    }

@app.get("/model-status")
async def model_status():
    """Get model loading status"""
    return {
        "loaded": chatbot.model_loaded,
        "loading": chatbot.loading,
        "deployment": "cloud_run"
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
