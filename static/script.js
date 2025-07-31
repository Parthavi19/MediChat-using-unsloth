// Initialize welcome time
document.getElementById('welcomeTime').textContent = new Date().toLocaleTimeString();

// Auto-resize textarea
const messageInput = document.getElementById('messageInput');
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});

// Send message on Enter (but allow Shift+Enter for new lines)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Toggle settings panel
document.getElementById('settingsToggle').addEventListener('click', toggleSettings);
function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    const toggle = document.getElementById('settingsToggle');
    const isOpen = panel.style.display === 'block';
    panel.style.display = isOpen ? 'none' : 'block';
    toggle.setAttribute('aria-expanded', !isOpen);
}

document.getElementById('sendButton').addEventListener('click', sendMessage);

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;

    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');

    // Disable input and button
    messageInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show typing indicator
    typingIndicator.style.display = 'block';
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        // Get settings
        const temperature = parseFloat(document.getElementById('temperature').value);
        const maxLength = parseInt(document.getElementById('maxLength').value);

        // Send request to API
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                temperature: temperature,
                max_length: maxLength
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Hide typing indicator
        typingIndicator.style.display = 'none';
        
        // Add bot response
        addMessage(data.response, 'bot');

    } catch (error) {
        console.error('Error:', error);
        typingIndicator.style.display = 'none';
        addMessage('I apologize, but I encountered an error processing your request. Please try again later.', 'bot');
    } finally {
        // Re-enable input and button
        messageInput.disabled = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

function addMessage(content, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const currentTime = new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-content">${formatMessage(content)}</div>
        <div class="message-time">${currentTime}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(content) {
    // Basic formatting for better readability
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

// Sample medical questions for quick testing
const sampleQuestions = [
    "What are the symptoms of the common cold?",
    "How can I prevent heart disease?",
    "What should I know about diabetes?",
    "What are the signs of dehydration?",
    "How much sleep do adults need?"
];

// Add sample questions button (optional)
function addSampleQuestions() {
    const container = document.querySelector('.chat-input-container');
    const samplesDiv = document.createElement('div');
    samplesDiv.innerHTML = `
        <div style="margin-bottom: 10px;">
            <small>Quick questions:</small>
            ${sampleQuestions.map(q => 
                `<button onclick="document.getElementById('messageInput').value='${q}'" 
                        style="margin: 2px; padding: 4px 8px; font-size: 11px; border: 1px solid #ddd; border-radius: 10px; background: white; cursor: pointer;">
                    ${q.substring(0, 30)}...
                </button>`
            ).join('')}
        </div>
    `;
    container.insertBefore(samplesDiv, container.firstChild);
}

// Uncomment to add sample questions
// addSampleQuestions();
