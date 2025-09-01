/**
 * Enhanced Voice Call Manager with SeamlessM4T Real-time Translation
 * Supports Arabic, English, and French real-time voice-to-voice translation
 */

class EnhancedVoiceCall {
    constructor() {
        this.callId = null;
        this.websocket = null;
        this.webrtc = new WebRTCManager();
        this.audioVisualization = new AudioVisualization('voice-visualization');
        this.mediaRecorder = null;
        this.isRecording = false;
        this.isTranslationActive = false;
        
        // Language settings
        this.userLanguage = 'en'; // User's preferred language
        this.supportedLanguages = {
            'ar': 'Arabic',
            'en': 'English', 
            'fr': 'French'
        };
        
        // Translation settings
        this.translationSettings = {
            enabled: true,
            chunkDuration: 2000, // 2 seconds
            overlapDuration: 500, // 0.5 second overlap
            minChunkSize: 16000   // Minimum audio size for processing
        };
        
        // Audio processing
        this.audioChunks = [];
        this.isProcessingAudio = false;
        
        // UI elements
        this.initializeUI();
    }
    
    initializeUI() {
        // Get UI elements
        this.elements = {
            callStatus: document.getElementById('call-status'),
            participantsList: document.getElementById('participants-list'),
            translationToggle: document.getElementById('translation-toggle'),
            languageSelect: document.getElementById('language-select'),
            transcriptionContainer: document.getElementById('transcription-container'),
            audioVisualization: document.getElementById('audio-visualization'),
            hangupButton: document.getElementById('hangup-button'),
            muteButton: document.getElementById('mute-button')
        };
        
        // Setup language selector
        this.setupLanguageSelector();
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    setupLanguageSelector() {
        if (this.elements.languageSelect) {
            // Clear existing options
            this.elements.languageSelect.innerHTML = '';
            
            // Add language options
            Object.entries(this.supportedLanguages).forEach(([code, name]) => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = name;
                if (code === this.userLanguage) {
                    option.selected = true;
                }
                this.elements.languageSelect.appendChild(option);
            });
        }
    }
    
    setupEventListeners() {
        // Language change
        if (this.elements.languageSelect) {
            this.elements.languageSelect.addEventListener('change', (e) => {
                this.changeLanguage(e.target.value);
            });
        }
        
        // Translation toggle
        if (this.elements.translationToggle) {
            this.elements.translationToggle.addEventListener('change', (e) => {
                this.toggleTranslation(e.target.checked);
            });
        }
        
        // Hangup button
        if (this.elements.hangupButton) {
            this.elements.hangupButton.addEventListener('click', () => {
                this.endCall();
            });
        }
        
        // Mute button
        if (this.elements.muteButton) {
            this.elements.muteButton.addEventListener('click', () => {
                this.toggleMute();
            });
        }
    }
    
    async init(config) {
        try {
            this.callId = config.callId;
            this.userLanguage = config.userLanguage || 'en';
            
            console.log(`🎙️ Initializing enhanced voice call: ${this.callId}`);
            console.log(`🌍 User language: ${this.supportedLanguages[this.userLanguage]}`);
            
            // Update UI
            this.updateCallStatus('Connecting...');
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Initialize WebRTC
            await this.webrtc.initialize(this.callId, this.websocket, config.isInitiator || false);
            
            // Get user media
            await this.webrtc.getUserMedia();
            
            // Setup audio visualization
            this.audioVisualization.setupAudioAnalysis(this.webrtc.localStream, null);
            
            // Start real-time translation if enabled
            if (this.translationSettings.enabled) {
                await this.startRealtimeTranslation();
            }
            
            this.updateCallStatus('Connected');
            console.log('✅ Enhanced voice call initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize enhanced voice call:', error);
            this.updateCallStatus('Connection failed');
            throw error;
        }
    }
    
    async connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/voice-call/ws/${this.callId}`;
        
        // Get token from session storage or wherever it's stored
        const token = localStorage.getItem('access_token') || sessionStorage.getItem('token');
        
        const fullUrl = `${wsUrl}?token=${encodeURIComponent(token)}&language=${this.userLanguage}`;
        
        this.websocket = new WebSocket(fullUrl);
        
        return new Promise((resolve, reject) => {
            this.websocket.onopen = () => {
                console.log('🔌 WebSocket connected for enhanced voice call');
                resolve();
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };
            
            this.websocket.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.updateCallStatus('Disconnected');
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                reject(error);
            };
        });
    }
    
    handleWebSocketMessage(message) {
        console.log('📩 Received message:', message.type);
        
        switch (message.type) {
            case 'connected':
                this.handleConnected(message);
                break;
            case 'voice_translation':
                this.handleVoiceTranslation(message);
                break;
            case 'webrtc_offer':
                this.webrtc.handleOffer(message);
                break;
            case 'webrtc_answer':
                this.webrtc.handleAnswer(message);
                break;
            case 'ice_candidate':
                this.webrtc.handleIceCandidate(message);
                break;
            case 'participant_joined':
                this.handleParticipantJoined(message);
                break;
            case 'participant_left':
                this.handleParticipantLeft(message);
                break;
            case 'call_ended':
                this.handleCallEnded(message);
                break;
            case 'error':
                this.handleError(message);
                break;
            case 'pong':
                // Heartbeat response
                break;
            default:
                console.warn('Unknown message type:', message.type);
        }
    }
    
    handleConnected(message) {
        console.log('✅ Connected to call:', message.call_id);
        
        if (message.translation_available) {
            console.log('🔄 Real-time translation available');
            console.log('🌍 Supported languages:', message.supported_languages);
        } else {
            console.warn('⚠️ Real-time translation not available');
            this.showNotification('Translation service not available', 'warning');
        }
        
        this.updateCallStatus('In call');
    }
    
    handleVoiceTranslation(message) {
        try {
            console.log(`🔄 Received translation from user ${message.from_user}`);
            console.log(`🌍 ${message.source_language} → ${message.target_language}`);
            
            // Play translated audio
            if (message.audio_data) {
                this.playTranslatedAudio(message.audio_data);
            }
            
            // Display transcription if available
            if (message.transcription) {
                this.addTranscription(message.transcription, message.target_language, message.from_user);
            }
            
        } catch (error) {
            console.error('❌ Failed to handle voice translation:', error);
        }
    }
    
    async playTranslatedAudio(audioData) {
        try {
            // Decode base64 audio
            const arrayBuffer = Uint8Array.from(atob(audioData), c => c.charCodeAt(0)).buffer;
            
            // Create audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Play audio
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            
            // Connect to visualization
            const analyser = audioContext.createAnalyser();
            source.connect(analyser);
            analyser.connect(audioContext.destination);
            
            // Update visualization for remote audio
            this.audioVisualization.visualizeRemoteAudio(analyser);
            
            source.start();
            
        } catch (error) {
            console.error('❌ Failed to play translated audio:', error);
        }
    }
    
    async startRealtimeTranslation() {
        if (!this.webrtc.localStream || this.isTranslationActive) {
            return;
        }
        
        console.log('🎙️ Starting real-time voice translation');
        this.isTranslationActive = true;
        
        // Configure MediaRecorder for optimal processing
        const options = {
            mimeType: 'audio/webm;codecs=opus',
            audioBitsPerSecond: 16000 // 16kHz for SeamlessM4T
        };
        
        try {
            this.mediaRecorder = new MediaRecorder(this.webrtc.localStream, options);
        } catch (error) {
            console.warn('WebM not supported, trying alternative format');
            this.mediaRecorder = new MediaRecorder(this.webrtc.localStream);
        }
        
        // Handle data available
        this.mediaRecorder.ondataavailable = async (event) => {
            if (event.data.size > this.translationSettings.minChunkSize && this.isTranslationActive) {
                await this.processVoiceChunk(event.data);
            }
        };
        
        // Handle recording stop
        this.mediaRecorder.onstop = () => {
            if (this.isTranslationActive) {
                // Restart recording for continuous translation
                setTimeout(() => {
                    if (this.mediaRecorder && this.isTranslationActive) {
                        this.mediaRecorder.start(this.translationSettings.chunkDuration);
                    }
                }, this.translationSettings.overlapDuration);
            }
        };
        
        // Start recording
        this.mediaRecorder.start(this.translationSettings.chunkDuration);
        
        // Update UI
        if (this.elements.translationToggle) {
            this.elements.translationToggle.checked = true;
        }
    }
    
    async processVoiceChunk(audioBlob) {
        if (this.isProcessingAudio) {
            return; // Skip if already processing
        }
        
        this.isProcessingAudio = true;
        
        try {
            // Convert to base64
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
            
            // Send to server for translation
            this.websocket.send(JSON.stringify({
                type: 'voice_chunk',
                audio_data: base64Audio,
                language: this.userLanguage,
                chunk_size: audioBlob.size,
                timestamp: Date.now()
            }));
            
            // Show processing indicator
            this.showProcessingIndicator();
            
        } catch (error) {
            console.error('❌ Failed to process voice chunk:', error);
        } finally {
            this.isProcessingAudio = false;
        }
    }
    
    stopRealtimeTranslation() {
        console.log('⏹️ Stopping real-time translation');
        this.isTranslationActive = false;
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        // Update UI
        if (this.elements.translationToggle) {
            this.elements.translationToggle.checked = false;
        }
    }
    
    toggleTranslation(enabled) {
        if (enabled) {
            this.startRealtimeTranslation();
        } else {
            this.stopRealtimeTranslation();
        }
        
        // Send settings update to server
        this.updateTranslationSettings({ enabled });
    }
    
    changeLanguage(newLanguage) {
        if (newLanguage === this.userLanguage) {
            return;
        }
        
        console.log(`🌍 Changing language: ${this.userLanguage} → ${newLanguage}`);
        this.userLanguage = newLanguage;
        
        // Send settings update to server
        this.updateTranslationSettings({ language: newLanguage });
        
        // Show notification
        this.showNotification(`Language changed to ${this.supportedLanguages[newLanguage]}`, 'info');
    }
    
    updateTranslationSettings(settings) {
        this.websocket.send(JSON.stringify({
            type: 'translation_settings',
            settings: settings
        }));
    }
    
    addTranscription(text, language, fromUser) {
        if (!this.elements.transcriptionContainer) {
            return;
        }
        
        const transcriptionItem = document.createElement('div');
        transcriptionItem.className = `transcription-item language-${language}`;
        transcriptionItem.innerHTML = `
            <div class="transcription-header">
                <span class="language-badge">${this.supportedLanguages[language] || language}</span>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                <span class="user-id">User ${fromUser}</span>
            </div>
            <div class="transcription-text">${text}</div>
        `;
        
        this.elements.transcriptionContainer.appendChild(transcriptionItem);
        this.elements.transcriptionContainer.scrollTop = this.elements.transcriptionContainer.scrollHeight;
    }
    
    showProcessingIndicator() {
        // Create or update processing indicator
        let indicator = document.getElementById('translation-processing');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'translation-processing';
            indicator.className = 'processing-indicator';
            indicator.innerHTML = '🔄 Translating...';
            document.body.appendChild(indicator);
        }
        
        indicator.style.display = 'block';
        
        // Hide after 3 seconds
        setTimeout(() => {
            indicator.style.display = 'none';
        }, 3000);
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    updateCallStatus(status) {
        if (this.elements.callStatus) {
            this.elements.callStatus.textContent = status;
        }
        console.log(`📞 Call status: ${status}`);
    }
    
    handleParticipantJoined(message) {
        console.log(`👤 Participant ${message.user_id} joined`);
        this.updateParticipantsList();
        this.showNotification('Participant joined', 'info');
    }
    
    handleParticipantLeft(message) {
        console.log(`👤 Participant ${message.user_id} left`);
        this.updateParticipantsList();
        this.showNotification('Participant left', 'info');
    }
    
    handleCallEnded(message) {
        console.log('📞 Call ended:', message.reason);
        this.updateCallStatus('Call ended');
        this.cleanup();
        
        // Redirect to dashboard after 3 seconds
        setTimeout(() => {
            window.location.href = '/';
        }, 3000);
    }
    
    handleError(message) {
        console.error('❌ Call error:', message.message);
        this.showNotification(`Error: ${message.message}`, 'error');
    }
    
    updateParticipantsList() {
        // Update participants list UI if needed
        // Implementation depends on your UI design
    }
    
    toggleMute() {
        if (this.webrtc.localStream) {
            const audioTrack = this.webrtc.localStream.getAudioTracks()[0];
            if (audioTrack) {
                audioTrack.enabled = !audioTrack.enabled;
                
                // Update UI
                if (this.elements.muteButton) {
                    this.elements.muteButton.textContent = audioTrack.enabled ? '🔊 Mute' : '🔇 Unmute';
                    this.elements.muteButton.classList.toggle('muted', !audioTrack.enabled);
                }
                
                console.log(`🔊 Audio ${audioTrack.enabled ? 'unmuted' : 'muted'}`);
            }
        }
    }
    
    async endCall() {
        console.log('📞 Ending call...');
        
        try {
            // Send end call message
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'end_call',
                    call_id: this.callId
                }));
            }
            
            // Call API to end call
            const response = await fetch(`/api/voice-call/end/${this.callId}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token') || sessionStorage.getItem('token')}`
                }
            });
            
            if (!response.ok) {
                console.warn('Failed to end call via API, continuing with cleanup');
            }
            
        } catch (error) {
            console.error('Error ending call:', error);
        } finally {
            this.cleanup();
            window.location.href = '/';
        }
    }
    
    cleanup() {
        console.log('🧹 Cleaning up voice call...');
        
        // Stop translation
        this.stopRealtimeTranslation();
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        // Clean up WebRTC
        if (this.webrtc) {
            this.webrtc.cleanup();
        }
        
        // Stop audio visualization
        if (this.audioVisualization) {
            this.audioVisualization.stop();
        }
    }
    
    // Heartbeat to keep connection alive
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            }
        }, 30000); // Every 30 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
}

// Export for use in other scripts
window.EnhancedVoiceCall = EnhancedVoiceCall;
