document.addEventListener('DOMContentLoaded', () => {
    // Initialize welcome time
    document.getElementById('welcomeTime').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Auto-resize textarea
    const messageInput = document.getElementById('messageInput');
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

    // Bind send button
    document.getElementById('sendButton').addEventListener('click', sendMessage);

    // Bind settings toggle
    document.getElementById('settingsToggle').addEventListener('click', toggleSettings);

    // Sample medical questions
    const sampleQuestions = [
        "What are the symptoms of the common cold?",
        "How can I prevent heart disease?",
        "What should I know about diabetes?",
        "What are the signs of dehydration?",
        "How much sleep do adults need?"
    ];

    // Add sample questions
    const sampleQuestionsDiv = document.getElementById('sampleQuestions');
    sampleQuestions.forEach(question => {
        const button = document.createElement('button');
        button.textContent = question.substring(0, 30) + (question.length > 30 ? '...' : '');
        button.addEventListener('click', () => {
            messageInput.value = question;
            messageInput.focus();
        });
        sampleQuestionsDiv.appendChild(button);
    });

    function toggleSettings() {
        const panel = document.getElementById('settingsPanel');
        const toggle = document.getElementById('settingsToggle');
        const isOpen = panel.style.display === 'block';
        panel.style.display = isOpen ? 'none' : 'block';
        toggle.setAttribute('aria-expanded', !isOpen);
        if (!isOpen) {
            document.getElementById('temperature').focus();
        }
    }

    function saveSettings() {
        const temperatureInput = document.getElementById('temperature');
        const maxLengthInput = document.getElementById('maxLength');
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
        let message = messageInput.value.trim();

        // Sanitize input
        message = message.replace(/[<>&"]/g, (c) => ({
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;'
        }[c])).substring(0, 1000);

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
        
        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-content">${formatMessage(content)}</div>
            <div class="message-time">${currentTime}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatMessage(content) {
        // Enhanced formatting for better readability
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/(\d+\.\s)/g, '<br>$1') // Add line breaks for numbered lists
            .replace(/(Disclaimer:.*)/g, '<div class="disclaimer-text">$1</div>');
    }
});
