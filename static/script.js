document.addEventListener('DOMContentLoaded', () => {
    // Initialize welcome time
    const welcomeTimeEl = document.getElementById('welcomeTime');
    if (welcomeTimeEl) {
        welcomeTimeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    // Auto-resize textarea
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send message on Enter (allow Shift+Enter for new lines)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Bind send button
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }

    // Bind settings toggle
    const settingsToggle = document.getElementById('settingsToggle');
    if (settingsToggle) {
        settingsToggle.addEventListener('click', toggleSettings);
    }

    // Sample medical questions
    const sampleQuestions = [
        "What are the symptoms of the common cold?",
        "How can I prevent heart disease?",
        "What should I know about diabetes?",
        "What are the signs of dehydration?",
        "How much sleep do adults need?"
    ];

    // Add sample questions if container exists
    const sampleQuestionsDiv = document.getElementById('sampleQuestions');
    if (sampleQuestionsDiv) {
        sampleQuestions.forEach(question => {
            const button = document.createElement('button');
            button.textContent = question.substring(0, 30) + (question.length > 30 ? '...' : '');
            button.className = 'sample-question-btn';
            button.addEventListener('click', () => {
                if (messageInput) {
                    messageInput.value = question;
                    messageInput.focus();
                }
            });
            sampleQuestionsDiv.appendChild(button);
        });
    }

    // Test connection on page load
    console.log('Page loaded, testing connection...');
    fetch('/health')
        .then(response => {
            console.log('Health check response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('Health check successful:', data);
        })
        .catch(error => {
            console.error('Health check failed:', error);
            addErrorMessage('Connection test failed. The server may be starting up. Please wait a moment and try again.');
        });

    function toggleSettings() {
        const panel = document.getElementById('settingsPanel');
        const toggle = document.getElementById('settingsToggle');
        
        if (!panel || !toggle) return;
        
        const isOpen = panel.style.display === 'block';
        panel.style.display = isOpen ? 'none' : 'block';
        toggle.setAttribute('aria-expanded', !isOpen);
        
        if (!isOpen) {
            const temperatureInput = document.getElementById('temperature');
            if (temperatureInput) {
                temperatureInput.focus();
            }
        }
    }

    function saveSettings() {
        const temperatureInput = document.getElementById('temperature');
        const maxLengthInput = document.getElementById('maxLength');
        
        if (!temperatureInput || !maxLengthInput) return;
        
        let temperature = parseFloat(temperatureInput.value);
        let maxLength = parseInt(maxLengthInput.value);

        // Validate inputs
        if (isNaN(temperature) || temperature < 0.1 || temperature > 2.0) {
            temperature = 0.7;
            temperatureInput.value = 0.7;
            alert('Temperature must be between 0.1 and 2.0. Reset to default (0.7).');
        }
        if (isNaN(maxLength) || maxLength < 50 || maxLength > 500) {
            maxLength = 200;
            maxLengthInput.value = 200;
            alert('Max length must be between 50 and 500. Reset to default (200).');
        }

        toggleSettings();
    }

    async function sendMessage() {
        const messageInput = document.getElementById('messageInput');
        if (!messageInput) {
            console.error('Message input element not found');
            return;
        }

        let message = messageInput.value.trim();
        console.log('Attempting to send message:', message);

        // Sanitize input on frontend
        message = message.replace(/[<>&"]/g, (c) => ({
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;'
        }[c])).substring(0, 1000);

        if (!message) {
            console.log('Message is empty, not sending');
            return;
        }

        const sendButton = document.getElementById('sendButton');
        const chatMessages = document.getElementById('chatMessages');
        const typingIndicator = document.getElementById('typingIndicator');

        if (!sendButton || !chatMessages) {
            console.error('Required elements not found');
            return;
        }

        // Disable input and button
        messageInput.disabled = true;
        sendButton.disabled = true;

        // Add user message
        addMessage(message, 'user');
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Show typing indicator
        if (typingIndicator) {
            typingIndicator.style.display = 'block';
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // Get settings
            const temperatureInput = document.getElementById('temperature');
            const maxLengthInput = document.getElementById('maxLength');
            
            const temperature = temperatureInput ? parseFloat(temperatureInput.value) || 0.7 : 0.7;
            const maxLength = maxLengthInput ? parseInt(maxLengthInput.value) || 200 : 200;

            console.log('Making fetch request to /chat with settings:', { temperature, maxLength });

            // Send request to API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    temperature: temperature,
                    max_length: maxLength
                })
            });

            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error:', errorText);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);
            
            // Hide typing indicator
            if (typingIndicator) {
                typingIndicator.style.display = 'none';
            }
            
            // Add bot response
            if (data.status === 'success') {
                addMessage(data.response, 'bot');
            } else {
                addErrorMessage('I encountered an error processing your request. Please try again.');
            }

        } catch (error) {
            console.error('Error sending message:', error);
            
            // Hide typing indicator
            if (typingIndicator) {
                typingIndicator.style.display = 'none';
            }
            
            // Show user-friendly error message
            let errorMessage = 'I apologize, but I encountered an error processing your request.';
            
            if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Unable to connect to the server. Please check your internet connection and try again.';
            } else if (error.message.includes('500')) {
                errorMessage = 'Server error occurred. Please try again in a moment.';
            } else if (error.message.includes('400')) {
                errorMessage = 'Invalid request. Please rephrase your question and try again.';
            }
            
            addErrorMessage(errorMessage + ` (Error: ${error.message})`);
            
        } finally {
            // Re-enable input and button
            messageInput.disabled = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    function addMessage(content, type) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-content">${formatMessage(content)}</div>
            <div class="message-time">${currentTime}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addErrorMessage(content) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const errorDiv = document.createElement('div');
        errorDiv.className = 'message bot error-message';
        errorDiv.style.background = '#f8d7da';
        errorDiv.style.color = '#721c24';
        errorDiv.style.border = '1px solid #f5c6cb';
        
        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        errorDiv.innerHTML = `
            <div class="message-content">⚠️ ${content}</div>
            <div class="message-time">${currentTime}</div>
        `;
        
        chatMessages.appendChild(errorDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatMessage(content) {
        // Enhanced formatting for better readability
        if (!content) return '';
        
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/(\d+\.\s)/g, '<br>$1') // Add line breaks for numbered lists
            .replace(/(Disclaimer:.*)/g, '<div class="disclaimer-text">$1</div>')
            .replace(/• /g, '• '); // Ensure bullet points display correctly
    }

    // Make functions available globally for event handlers
    window.sendMessage = sendMessage;
    window.toggleSettings = toggleSettings;
    window.saveSettings = saveSettings;
});
