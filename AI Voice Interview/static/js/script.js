document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const socket = io({
        reconnectionAttempts: 5,
        reconnectionDelay: 2000,
    });

    // --- Document object model Elements ---
    const uploadArea = document.getElementById('upload-area');
    const resumeFile = document.getElementById('resume-file');
    const interviewControls = document.getElementById('interview-controls');
    const messagesDiv = document.getElementById('message-log'); // Optional log area
    const statusDiv = document.getElementById('status');
    const speakButton = document.getElementById('speakButton');
    const agentImage = document.getElementById('agent-image');
    const dialogueText = document.getElementById('dialogue-text');

    // --- State ---
    let isAgentSpeaking = false;
    let recognition = null;
    let synthesis = window.speechSynthesis;
    let currentUtterance = null;

    // --- NEW: Speech Queue ---
    let speechQueue = []; // Holds message data {text: "...", is_question: bool}
    let currentMessageData = null; // Track data of the currently speaking message


    // --- Utility Functions ---
    function addLogMessage(text, sender = 'status') {
       if (!messagesDiv) return;
       const MAX_LOG_MESSAGES = 50;
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        const sanitizedText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        messageElement.innerHTML = `[${sender.toUpperCase()}] ${sanitizedText}`;
        messagesDiv.appendChild(messageElement);
        while (messagesDiv.childNodes.length > MAX_LOG_MESSAGES) {
            messagesDiv.removeChild(messagesDiv.firstChild);
        }
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function updateStatus(message) {
        statusDiv.textContent = message;
        console.log("Status:", message);
    }

    // --- NEW: Function to process the next item in the queue ---
    function processSpeechQueue() {
        // Don't start next message if already speaking or queue is empty
        if (isAgentSpeaking || speechQueue.length === 0) {
            return;
        }

        // Get the next message data from the queue
        currentMessageData = speechQueue.shift(); // Removes the first item

        // Start speaking this message
        _speakInternal(currentMessageData.text);
    }

    // --- RENAMED: Internal function to handle actual SpeechSynthesis ---
    function _speakInternal(text) {
        // This function assumes it's safe to speak (called by processSpeechQueue)
        if (!synthesis) {
             updateStatus("Speech Synthesis not supported.");
             console.error("Speech Synthesis not supported");
             dialogueText.textContent = text + " (Speech not supported)";
             // If speech fails, maybe try processing next queue item?
             setTimeout(processSpeechQueue, 100); // Try next after short delay
             return;
        }

        // Update dialogue box for the message being spoken
        dialogueText.textContent = text;

        isAgentSpeaking = true;
        speakButton.disabled = true; // Always disable button while agent is preparing/speaking
        updateStatus("Agent is speaking...");
        console.log("Agent speaking:", text);

        currentUtterance = new SpeechSynthesisUtterance(text);
        currentUtterance.rate = 0.95;
        currentUtterance.pitch = 1;
        // Optional: Voice selection logic here

        currentUtterance.onend = () => {
            console.log("Agent finished speaking utterance.");
            isAgentSpeaking = false;
            const wasQuestion = currentMessageData?.is_question; // Check the data of the message that JUST finished
            currentUtterance = null;
            currentMessageData = null;

            // Decide button state based on whether the finished utterance was a question
            if (wasQuestion) {
                speakButton.disabled = false;
                updateStatus("Your turn. Press 'Start Speaking'.");
            } else {
                speakButton.disabled = true; // Keep disabled after feedback/non-questions
                updateStatus("Waiting for next step...");
            }

            // IMPORTANT: Check queue for next message AFTER finishing
            processSpeechQueue();
        };

        currentUtterance.onerror = (event) => {
            console.error("SpeechSynthesis Error:", event.error, event);
            dialogueText.textContent = text + " (Speech error)";
            updateStatus("Error playing agent voice.");
            isAgentSpeaking = false;
            const wasQuestion = currentMessageData?.is_question; // Check data of errored message
            currentUtterance = null;
            currentMessageData = null;

            // If the utterance that failed was a question, maybe still enable button?
             if (wasQuestion) {
                  speakButton.disabled = false;
                  updateStatus("Agent speech error. Press 'Start Speaking'.");
             } else {
                 speakButton.disabled = true;
                 updateStatus("Agent speech error. Waiting...");
             }

            // IMPORTANT: Try processing next item even if current one failed
            processSpeechQueue();
        };

        // Start speaking
        try {
            // Cancel any lingering speech just before speaking - belt and braces
            if (synthesis.speaking || synthesis.pending) {
                 synthesis.cancel();
                 console.warn("Cancelled lingering speech before new utterance.");
                 // Add a tiny delay if cancelling right before speaking
                 setTimeout(() => synthesis.speak(currentUtterance), 50);
             } else {
                 synthesis.speak(currentUtterance);
             }
        } catch (e) {
            console.error("Error calling synthesis.speak:", e);
            dialogueText.textContent = text + " (Speech error)";
            updateStatus("Could not initiate agent voice.");
            isAgentSpeaking = false;
            // Try processing next queue item
            processSpeechQueue();
        }
    }

    // --- NEW: Public function to add messages to the queue ---
    function speak(data) {
        if (!data || !data.text) {
            console.warn("Speak function called with invalid data");
            return;
        }
        addLogMessage(data.text, 'agent'); // Log when message is received/queued
        speechQueue.push(data); // Add message data to the end of the queue
        processSpeechQueue();   // Attempt to process the queue immediately
    }


    // --- Speech Recognition (User Speaking) - No changes needed here ---
    function initializeSpeechRecognition() {
        window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!window.SpeechRecognition) {
            updateStatus("Speech Recognition API not supported. Use Chrome/Edge.");
            speakButton.disabled = true;
            console.error("Speech Recognition not supported.");
            return null;
        }
        // ... (rest of initializeSpeechRecognition remains the same as previous version) ...
         try {
            const recognizer = new SpeechRecognition();
            recognizer.continuous = false;
            recognizer.lang = 'en-US';
            recognizer.interimResults = false;
            recognizer.maxAlternatives = 1;

            recognizer.onstart = () => {
                updateStatus("Listening...");
                speakButton.textContent = "Listening...";
                speakButton.classList.add('listening');
                speakButton.disabled = true;
                console.log("Speech recognition started.");
            };

            recognizer.onresult = (event) => {
                // Ensure agent isn't trying to speak over recognition results
                 if (synthesis.speaking || synthesis.pending) {
                     synthesis.cancel(); // Force agent to stop if user response comes in
                     console.warn("Agent speech cancelled due to user response processing.");
                     isAgentSpeaking = false; // Update state if we cancelled manually
                 }
                const transcript = event.results[0][0].transcript;
                console.log("Speech recognized:", transcript);
                updateStatus("Processing your answer...");
                addLogMessage(transcript, 'user');
                socket.emit('user_response', { text: transcript });
            };

            recognizer.onspeechend = () => {
                console.log("Speech ended.");
            };

            recognizer.onend = () => {
                updateStatus("Processing complete. Waiting for agent...");
                speakButton.textContent = "Start Speaking";
                speakButton.classList.remove('listening');
                console.log("Speech recognition cycle ended.");
            };

            recognizer.onerror = (event) => {
                console.error("SpeechRecognition Error:", event.error, event.message);
                let errorMessage = "Speech recognition error.";
                 if (event.error === 'no-speech') {
                     errorMessage = "Didn't hear anything. Please try speaking again.";
                 } else if (event.error === 'audio-capture') {
                     errorMessage = "Microphone error. Check permissions and connection.";
                 } else if (event.error === 'not-allowed') {
                     errorMessage = "Microphone access denied. Please allow access.";
                 } else {
                      errorMessage = `Recognition error: ${event.error}. Please try again.`;
                  }
                updateStatus(errorMessage);
                speakButton.textContent = "Start Speaking";
                speakButton.classList.remove('listening');
                speakButton.disabled = false; // Re-enable button on error
            };
            return recognizer;

        } catch (e) {
            console.error("Failed to initialize SpeechRecognition:", e);
            updateStatus("Speech Recognition could not start. Check browser compatibility/permissions.");
            speakButton.disabled = true;
            return null;
        }
    }

    recognition = initializeSpeechRecognition();

    // --- WebSocket Event Handlers ---
    socket.on('connect', () => {
        console.log('Socket connected:', socket.id);
        updateStatus('Connected. Please upload your resume.');
        // Reset UI and queue on connect/reconnect
        speechQueue = [];
        if (synthesis.speaking) synthesis.cancel();
        isAgentSpeaking = false;
        uploadArea.style.display = 'block';
        interviewControls.style.display = 'none';
        dialogueText.textContent = "Upload your resume to begin.";
        if (messagesDiv) messagesDiv.innerHTML = '';
        speakButton.disabled = true;
        speakButton.classList.remove('listening');
        speakButton.textContent = "Start Speaking";
    });

    socket.on('disconnect', (reason) => {
        console.log('Socket disconnected:', reason);
        updateStatus('Disconnected. Attempting to reconnect...');
        speakButton.disabled = true;
         speechQueue = []; // Clear queue on disconnect
        if (synthesis.speaking) synthesis.cancel();
        isAgentSpeaking = false;
    });

     socket.on('connect_error', (error) => {
         console.error('Connection Error:', error);
         updateStatus('Connection failed. Please check server and refresh.');
         speakButton.disabled = true;
         speechQueue = [];
     });

    socket.on('status_update', (data) => {
        updateStatus(data.message);
        addLogMessage(data.message, 'status');
    });

    socket.on('error_message', (data) => {
        console.error('Server Error:', data.message);
        const errorText = `Error: ${data.message}`;
        updateStatus(errorText);
        dialogueText.textContent = errorText; // Show error in dialogue box
        addLogMessage(errorText, 'error');
        speakButton.disabled = true;
        speechQueue = []; // Clear queue on critical error
        if (synthesis.speaking) synthesis.cancel();
        isAgentSpeaking = false;
    });

    socket.on('agent_message', (data) => {
         // Instead of calling _speakInternal directly, add to queue using public speak function
         speak(data);
     });


     socket.on('interview_finished', (data) => {
         const finishMessage = data.message || "Interview Finished.";
         updateStatus(finishMessage);
         // The final feedback should have been the last message queued and spoken
         speakButton.disabled = true; // Interview over
         console.log("Interview finished signal received.");
         // Optional: Add 'Start Over' button
     });


    // --- Drag and Drop & File Input --- (No changes needed here) ---
     function handleFileSelect(file) {
        if (!file) return;
        if (file.type === 'application/pdf') {
            updateStatus(`Selected: ${file.name}. Uploading...`);
            uploadArea.style.display = 'none';
            interviewControls.style.display = 'block';
            dialogueText.textContent = "Analyzing resume...";
            if (messagesDiv) messagesDiv.innerHTML = '';
            speakButton.disabled = true;
            speechQueue = []; // Clear queue on new upload
            if (synthesis.speaking) synthesis.cancel(); // Cancel speech on new upload

            const reader = new FileReader();
            reader.onload = function(e) {
                socket.emit('upload_resume', {
                    file_name: file.name,
                    file_data: e.target.result
                });
                updateStatus("Resume uploaded. Waiting for analysis...");
            };
            reader.onerror = function(e) {
                 console.error("FileReader error:", e);
                 updateStatus("Error reading file.");
                 uploadArea.style.display = 'block';
                 interviewControls.style.display = 'none';
                 dialogueText.textContent = "Error reading file.";
            };
            reader.readAsDataURL(file);
        } else {
            updateStatus('Please select a PDF file only.');
            resumeFile.value = '';
        }
    }
    // Drag and Drop listeners... (remain the same)
     ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
         uploadArea.addEventListener(eventName, (e) => {
             e.preventDefault();
             e.stopPropagation();
         });
     });
     ['dragenter', 'dragover'].forEach(eventName => {
         uploadArea.addEventListener(eventName, () => uploadArea.classList.add('highlight'));
     });
     ['dragleave', 'drop'].forEach(eventName => {
         uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('highlight'));
     });
     uploadArea.addEventListener('drop', (e) => {
         const files = e.dataTransfer.files;
         if (files.length > 0) handleFileSelect(files[0]);
     });
     // Click to upload listeners... (remain the same)
     uploadArea.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON' && e.target.id !== 'resume-file') {
              resumeFile.click();
         }
     });
     resumeFile.addEventListener('change', (e) => {
         if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
     });


    // --- Speak Button Listener --- (No changes needed here) ---
    speakButton.addEventListener('click', () => {
        if (!recognition) {
            updateStatus("Speech recognition not available.");
            return;
        }
        // Prevent user speaking if agent is speaking OR if there are messages waiting in the queue
        if (isAgentSpeaking || synthesis.speaking || synthesis.pending || speechQueue.length > 0) {
            console.warn("Tried to speak while agent was speaking or has pending messages.");
            updateStatus("Please wait for the agent to finish.");
            return;
        }
        try {
            recognition.start();
        } catch (e) {
             console.error("Error starting recognition:", e);
             if (e.name === 'InvalidStateError') {
                 updateStatus("Listening already started?");
             } else {
                  updateStatus("Could not start listening. Check mic permissions.");
                  speakButton.disabled = false; // Allow retry maybe?
             }
        }
    });

     // --- Initial State ---
     updateStatus("Waiting for resume upload...");
     interviewControls.style.display = 'none'; // Hide controls initially

     // Stop processes on page unload
     window.addEventListener('beforeunload', () => {
        socket.disconnect(); // Disconnect socket
        if (recognition) {
             try { recognition.abort(); } catch(e) {}
         }
        if (synthesis) synthesis.cancel(); // Cancel any ongoing speech
    });
});