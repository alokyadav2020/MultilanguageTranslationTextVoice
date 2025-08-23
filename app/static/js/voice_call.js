// filepath: app/static/js/voice_call.js

/**
 * Voice Call Main Controller
 * Manages the voice call interface and coordinates WebRTC, UI, and server communication
 */

class VoiceCall {
    constructor() {
        this.callId = null;
        this.participantName = null;
        this.isIncoming = false;
        this.token = null;
        this.apiBase = '/api/voice-call';
        this.wsUrl = null;
        
        // WebRTC and WebSocket
        this.webrtc = new WebRTCManager();
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // Call state
        this.callState = 'idle'; // idle, connecting, ringing, active, ending, ended
        this.startTime = null;
        this.durationInterval = null;
        this.isMuted = false;
        this.currentVolume = 80;
        
        // UI elements
        this.elements = {};
        
        // Event handlers
        this.setupWebRTCHandlers();
        
        // Audio elements
        this.ringtone = null;
        this.ringback = null;
        
        // Chat functionality
        this.chatMessages = [];
        this.unreadCount = 0;
        
        console.log('📞 VoiceCall initialized');
    }

    /**
     * Initialize voice call with configuration
     */
    async init(config) {
        try {
            this.callId = config.callId;
            this.participantName = config.participantName;
            this.isIncoming = config.isIncoming;
            this.token = config.token;
            this.apiBase = config.apiBase || '/api/voice-call';
            
            console.log('🚀 Initializing voice call:', config);
            
            // Cache DOM elements
            this.cacheElements();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize audio elements
            this.initializeAudio();
            
            // Update UI
            this.updateParticipantInfo();
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Handle incoming vs outgoing call
            if (this.isIncoming) {
                this.handleIncomingCall();
            } else {
                await this.initiateCall();
            }
            
            return true;
            
        } catch (error) {
            console.error('❌ Voice call initialization failed:', error);
            this.showError('Failed to initialize voice call: ' + error.message);
            return false;
        }
    }

    /**
     * Cache DOM elements for performance
     */
    cacheElements() {
        console.log('🔍 Caching DOM elements...');
        
        this.elements = {
            // Header elements
            participantName: document.getElementById('participant-name'),
            callStatus: document.getElementById('call-status'),
            callDuration: document.getElementById('call-duration'),
            
            // Control buttons
            muteBtn: document.getElementById('mute-btn'),
            speakerBtn: document.getElementById('speaker-btn'),
            chatToggleBtn: document.getElementById('chat-toggle-btn'),
            endCallBtn: document.getElementById('end-call-btn'),
            closeCallBtn: document.getElementById('close-call-btn'),
            minimizeBtn: document.getElementById('minimize-btn'),
            
            // Volume and connection
            volumeSlider: document.getElementById('volume-slider'),
            connectionStatus: document.getElementById('connection-status'),
            
            // Translation controls
            enableTranslation: document.getElementById('enable-translation'),
            translationSettings: document.getElementById('translation-settings'),
            myLanguage: document.getElementById('my-language'),
            translateTo: document.getElementById('translate-to'),
            
            // Volume indicators
            myVolume: document.getElementById('my-volume'),
            theirVolume: document.getElementById('their-volume'),
            
            // Chat elements
            chatSidebar: document.getElementById('chat-sidebar'),
            chatMessages: document.getElementById('chat-messages'),
            chatInput: document.getElementById('chat-input'),
            sendMessage: document.getElementById('send-message'),
            messageLanguage: document.getElementById('message-language'),
            autoTranslate: document.getElementById('auto-translate'),
            toggleChat: document.getElementById('toggle-chat'),
            
            // Stats elements
            callStats: document.getElementById('call-stats'),
            toggleStats: document.getElementById('toggle-stats'),
            latencyValue: document.getElementById('latency-value'),
            qualityValue: document.getElementById('quality-value'),
            packetLossValue: document.getElementById('packet-loss-value'),
            bitrateValue: document.getElementById('bitrate-value'),
            
            // Modal elements
            incomingCallModal: document.getElementById('incoming-call-modal'),
            callerName: document.getElementById('caller-name'),
            answerCall: document.getElementById('answer-call'),
            declineCall: document.getElementById('decline-call'),
            
            callEndedModal: document.getElementById('call-ended-modal'),
            callSummaryParticipant: document.getElementById('call-summary-participant'),
            callSummaryDuration: document.getElementById('call-summary-duration'),
            callEndReason: document.getElementById('call-end-reason'),
            callAgainBtn: document.getElementById('call-again-btn'),
            
            // Audio elements
            localAudio: document.getElementById('local-audio'),
            remoteAudio: document.getElementById('remote-audio'),
            ringtone: document.getElementById('ringtone'),
            ringback: document.getElementById('ringback')
        };
        
        // Log which critical elements were found
        const criticalElements = ['participantName', 'callStatus', 'muteBtn', 'endCallBtn', 'remoteAudio'];
        criticalElements.forEach(elementName => {
            if (this.elements[elementName]) {
                console.log(`✅ Found critical element: ${elementName}`);
            } else {
                console.error(`❌ Missing critical element: ${elementName}`);
            }
        });
        
        // Update status immediately
        if (this.elements.callStatus) {
            this.elements.callStatus.textContent = 'Initializing...';
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        console.log('🎛️ Setting up event listeners...');
        
        // Control buttons
        this.elements.muteBtn?.addEventListener('click', () => {
            console.log('🔇 Mute button clicked');
            this.toggleMute();
        });
        this.elements.speakerBtn?.addEventListener('click', () => {
            console.log('🔊 Speaker button clicked');
            this.toggleSpeaker();
        });
        this.elements.chatToggleBtn?.addEventListener('click', () => {
            console.log('💬 Chat toggle clicked');
            this.toggleChat();
        });
        this.elements.endCallBtn?.addEventListener('click', () => {
            console.log('📞 End call button clicked');
            this.endCall();
        });
        this.elements.closeCallBtn?.addEventListener('click', () => {
            console.log('❌ Close call button clicked');
            this.endCall();
        });
        this.elements.minimizeBtn?.addEventListener('click', () => {
            console.log('➖ Minimize button clicked');
            this.minimizeWindow();
        });
        
        // Translation controls
        this.elements.enableTranslation?.addEventListener('change', (e) => {
            console.log('🌐 Translation toggled:', e.target.checked);
            this.toggleTranslation(e.target.checked);
        });
        
        this.elements.myLanguage?.addEventListener('change', (e) => {
            console.log('🗣️ My language changed to:', e.target.value);
            this.updateTranslationSettings();
        });
        
        this.elements.translateTo?.addEventListener('change', (e) => {
            console.log('🔄 Translate to changed to:', e.target.value);
            this.updateTranslationSettings();
        });
        
        // Log which buttons were found
        const buttonElements = [
            'muteBtn', 'speakerBtn', 'chatToggleBtn', 'endCallBtn', 'closeCallBtn', 'minimizeBtn'
        ];
        
        buttonElements.forEach(btnName => {
            if (this.elements[btnName]) {
                console.log(`✅ Found ${btnName}`);
            } else {
                console.warn(`❌ Missing ${btnName}`);
            }
        });
        
        // Volume control
        this.elements.volumeSlider?.addEventListener('input', (e) => {
            this.setVolume(parseInt(e.target.value));
        });
        
        // Chat functionality
        this.elements.toggleChat?.addEventListener('click', () => this.toggleChat());
        this.elements.sendMessage?.addEventListener('click', () => this.sendChatMessage());
        this.elements.chatInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });
        
        // Stats toggle
        this.elements.toggleStats?.addEventListener('click', () => this.toggleStats());
        
        // Incoming call modal
        this.elements.answerCall?.addEventListener('click', () => this.answerCall());
        this.elements.declineCall?.addEventListener('click', () => this.declineCall());
        
        // Call ended modal
        this.elements.callAgainBtn?.addEventListener('click', () => this.callAgain());
        
        // Rating stars
        document.querySelectorAll('.star').forEach(star => {
            star.addEventListener('click', (e) => {
                this.rateCall(parseInt(e.target.dataset.rating));
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'm' || e.key === 'M') {
                this.toggleMute();
            } else if (e.key === 'h' || e.key === 'H') {
                this.endCall();
            }
        });
    }

    /**
     * Setup WebRTC event handlers
     */
    setupWebRTCHandlers() {
        this.webrtc.onConnectionStateChange = (state) => {
            this.handleConnectionStateChange(state);
        };
        
        this.webrtc.onIceConnectionStateChange = (state) => {
            this.handleIceConnectionStateChange(state);
        };
        
        this.webrtc.onRemoteStreamReceived = (stream) => {
            this.handleRemoteStream(stream);
        };
        
        this.webrtc.onLocalStreamReady = (stream) => {
            this.handleLocalStream(stream);
        };
        
        this.webrtc.onError = (type, error) => {
            this.handleWebRTCError(type, error);
        };
    }

    /**
     * Initialize audio elements
     */
    initializeAudio() {
        this.ringtone = this.elements.ringtone;
        this.ringback = this.elements.ringback;
        
        // Add error handling for missing audio files
        if (this.ringtone) {
            this.ringtone.addEventListener('error', () => {
                console.warn('📢 Ringtone audio file not found, using fallback');
                this.ringtone = null;
            });
        }
        
        if (this.ringback) {
            this.ringback.addEventListener('error', () => {
                console.warn('📢 Ringback audio file not found, using fallback');
                this.ringback = null;
            });
        }
        
        // Set initial volume
        if (this.elements.remoteAudio) {
            this.elements.remoteAudio.volume = this.currentVolume / 100;
        }
    }

    /**
     * Update participant information in UI
     */
    updateParticipantInfo() {
        if (this.elements.participantName) {
            this.elements.participantName.textContent = this.participantName || 'Unknown';
        }
        
        if (this.elements.callerName) {
            this.elements.callerName.textContent = this.participantName || 'Unknown';
        }
        
        if (this.elements.callSummaryParticipant) {
            this.elements.callSummaryParticipant.textContent = `Call with ${this.participantName || 'Unknown'}`;
        }
    }

    /**
     * Connect to WebSocket for real-time communication
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}${this.apiBase}/ws/${this.callId}?token=${this.token}`;
                
                console.log('🔌 Connecting to WebSocket:', wsUrl);
                console.log('🔌 Call ID:', this.callId);
                console.log('🔌 Token:', this.token ? this.token.substring(0, 20) + '...' : 'null');
                
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    console.log('✅ WebSocket connected successfully');
                    this.reconnectAttempts = 0;
                    
                    // Send initial join message
                    this.websocket.send(JSON.stringify({
                        type: 'join_call',
                        call_id: this.callId,
                        timestamp: new Date().toISOString()
                    }));
                    
                    resolve();
                };
                
                this.websocket.onmessage = (event) => {
                    console.log('📨 WebSocket message received:', event.data);
                    try {
                        const message = JSON.parse(event.data);
                        this.handleWebSocketMessage(message);
                    } catch (error) {
                        console.error('❌ Failed to parse WebSocket message:', error);
                    }
                };
                
                this.websocket.onclose = (event) => {
                    console.log('🔌 WebSocket closed:', event.code, event.reason);
                    this.handleWebSocketClose(event);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    reject(error);
                };
                
                // Timeout after 10 seconds
                setTimeout(() => {
                    if (this.websocket.readyState !== WebSocket.OPEN) {
                        console.error('❌ WebSocket connection timeout');
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 10000);
                
            } catch (error) {
                console.error('❌ WebSocket connection failed:', error);
                reject(error);
            }
        });
    }

    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(message) {
        console.log('📨 WebSocket message:', message.type);
        
        switch (message.type) {
            case 'call_answered':
                this.handleCallAnswered(message);
                break;
                
            case 'call_declined':
                this.handleCallDeclined(message);
                break;
                
            case 'call_ended':
                this.handleCallEnded(message);
                break;
                
            case 'webrtc_signaling':
                this.handleSignalingMessage(message);
                break;
                
            case 'ice_candidate':
                this.handleIceCandidate(message);
                break;
                
            case 'call_message':
                this.handleChatMessage(message);
                break;
                
            case 'call_status_update':
                this.handleStatusUpdate(message);
                break;
                
            case 'translated_audio':
                this.handleTranslatedAudio(message);
                break;
                
            case 'translation_state':
                this.handleTranslationState(message);
                break;
                
            case 'heartbeat_ack':
                // Keep connection alive
                break;
                
            default:
                console.log('❓ Unknown message type:', message.type);
        }
    }

    /**
     * Handle WebSocket close
     */
    handleWebSocketClose(event) {
        if (this.callState !== 'ended' && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket().catch(error => {
                    console.error('❌ WebSocket reconnection failed:', error);
                });
            }, 2000 * this.reconnectAttempts);
        }
    }

    /**
     * Initiate outgoing call
     */
    async initiateCall() {
        try {
            this.callState = 'connecting';
            this.updateCallStatus('Connecting...');
            
            console.log('📞 Initiating call...');
            
            // Initialize WebRTC
            await this.webrtc.initialize(this.callId, this.websocket, true);
            
            // Play ringback tone
            this.playRingback();
            
            this.callState = 'ringing';
            this.updateCallStatus('Ringing...');
            
        } catch (error) {
            console.error('❌ Failed to initiate call:', error);
            this.showError('Failed to initiate call: ' + error.message);
        }
    }

    /**
     * Handle incoming call
     */
    handleIncomingCall() {
        this.callState = 'ringing';
        this.updateCallStatus('Incoming call...');
        
        // Show incoming call modal
        const modal = new bootstrap.Modal(this.elements.incomingCallModal);
        modal.show();
        
        // Play ringtone
        this.playRingtone();
    }

    /**
     * Answer incoming call
     */
    async answerCall() {
        try {
            console.log('📞 Answering call...');
            
            // Hide modal and stop ringtone
            const modal = bootstrap.Modal.getInstance(this.elements.incomingCallModal);
            modal?.hide();
            this.stopRingtone();
            
            this.callState = 'connecting';
            this.updateCallStatus('Connecting...');
            
            // Initialize WebRTC
            await this.webrtc.initialize(this.callId, this.websocket, false);
            
            // Send answer to server
            const response = await fetch(`${this.apiBase}/answer/${this.callId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({
                    sdp_answer: this.webrtc.peerConnection?.localDescription?.sdp,
                    user_language: this.elements.messageLanguage?.value || 'en'
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to answer call');
            }
            
            console.log('✅ Call answered successfully');
            
        } catch (error) {
            console.error('❌ Failed to answer call:', error);
            this.showError('Failed to answer call: ' + error.message);
        }
    }

    /**
     * Decline incoming call
     */
    async declineCall() {
        try {
            console.log('📞 Declining call...');
            
            // Hide modal and stop ringtone
            const modal = bootstrap.Modal.getInstance(this.elements.incomingCallModal);
            modal?.hide();
            this.stopRingtone();
            
            // Send decline to server
            const response = await fetch(`${this.apiBase}/decline/${this.callId}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to decline call');
            }
            
            this.callState = 'ended';
            this.showCallEndedModal('Call declined');
            
        } catch (error) {
            console.error('❌ Failed to decline call:', error);
            this.showError('Failed to decline call: ' + error.message);
        }
    }

    /**
     * End active call
     */
    async endCall() {
        try {
            if (this.callState === 'ending' || this.callState === 'ended') {
                return;
            }
            
            console.log('📞 Ending call...');
            
            this.callState = 'ending';
            this.updateCallStatus('Ending call...');
            
            // Stop audio
            this.stopRingtone();
            this.stopRingback();
            
            // Close WebRTC
            this.webrtc.close();
            
            // Calculate duration
            const duration = this.startTime ? Date.now() - this.startTime : 0;
            
            // Send end call to server
            const response = await fetch(`${this.apiBase}/end/${this.callId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({
                    end_reason: 'normal',
                    quality_score: this.getCallRating(),
                    connection_quality: this.webrtc.getConnectionQuality()
                })
            });
            
            if (!response.ok) {
                console.warn('⚠️ Failed to notify server of call end');
            }
            
            this.callState = 'ended';
            this.showCallEndedModal('Call ended', duration);
            
        } catch (error) {
            console.error('❌ Failed to end call:', error);
            this.callState = 'ended';
            this.showCallEndedModal('Call ended with error');
        }
    }

    /**
     * Handle call answered notification
     */
    handleCallAnswered(message) {
        console.log('✅ Call was answered');
        
        this.stopRingback();
        this.callState = 'active';
        this.updateCallStatus('Connected');
        this.startCallTimer();
        
        // Handle SDP answer if provided
        if (message.sdp_answer) {
            this.webrtc.handleAnswer(message.sdp_answer);
        }
    }

    /**
     * Handle call declined notification
     */
    handleCallDeclined(message) {
        console.log('📞 Call was declined');
        
        this.stopRingback();
        this.callState = 'ended';
        this.showCallEndedModal('Call declined');
    }

    /**
     * Handle call ended notification
     */
    handleCallEnded(message) {
        console.log('📞 Call was ended by other party');
        
        this.stopRingtone();
        this.stopRingback();
        this.webrtc.close();
        
        this.callState = 'ended';
        this.showCallEndedModal('Call ended by other party');
    }

    /**
     * Handle WebRTC signaling messages
     */
    async handleSignalingMessage(message) {
        try {
            if (message.signaling_type === 'offer') {
                await this.webrtc.handleOffer(message.sdp);
            } else if (message.signaling_type === 'answer') {
                await this.webrtc.handleAnswer(message.sdp);
            }
        } catch (error) {
            console.error('❌ Failed to handle signaling message:', error);
        }
    }

    /**
     * Handle ICE candidate messages
     */
    async handleIceCandidate(message) {
        try {
            await this.webrtc.handleIceCandidate(message.candidate);
        } catch (error) {
            console.error('❌ Failed to handle ICE candidate:', error);
        }
    }

    /**
     * Handle connection state changes
     */
    handleConnectionStateChange(state) {
        console.log('🔄 Connection state changed:', state);
        
        switch (state) {
            case 'connecting':
                this.updateConnectionStatus('connecting', 'Connecting...');
                break;
            case 'connected':
                this.updateConnectionStatus('good', 'Connected');
                if (this.callState === 'connecting') {
                    this.callState = 'active';
                    this.updateCallStatus('Connected');
                    this.startCallTimer();
                }
                break;
            case 'disconnected':
                this.updateConnectionStatus('poor', 'Disconnected');
                break;
            case 'failed':
                this.updateConnectionStatus('poor', 'Connection failed');
                this.showError('Connection failed. Please check your internet connection.');
                break;
        }
    }

    /**
     * Handle ICE connection state changes
     */
    handleIceConnectionStateChange(state) {
        console.log('🧊 ICE connection state changed:', state);
        
        // Update connection indicator based on ICE state
        if (state === 'connected' || state === 'completed') {
            this.updateConnectionStatus('good', 'Good');
        } else if (state === 'checking') {
            this.updateConnectionStatus('fair', 'Connecting...');
        } else if (state === 'failed' || state === 'closed') {
            this.updateConnectionStatus('poor', 'Poor');
        }
    }

    /**
     * Handle remote audio stream
     */
    handleRemoteStream(stream) {
        console.log('🎵 Setting up remote audio stream');
        
        if (this.elements.remoteAudio) {
            this.elements.remoteAudio.srcObject = stream;
            this.elements.remoteAudio.volume = this.currentVolume / 100;
        }
        
        // Start audio visualization
        if (window.AudioVisualization) {
            window.AudioVisualization.setRemoteStream(stream);
        }
    }

    /**
     * Handle local audio stream
     */
    handleLocalStream(stream) {
        console.log('🎤 Setting up local audio stream');
        
        if (this.elements.localAudio) {
            this.elements.localAudio.srcObject = stream;
        }
        
        // Start audio visualization
        if (window.AudioVisualization) {
            window.AudioVisualization.setLocalStream(stream);
        }
    }

    /**
     * Handle WebRTC errors
     */
    handleWebRTCError(type, error) {
        console.error('❌ WebRTC error:', type, error);
        
        let message = 'An error occurred during the call';
        
        if (type === 'media') {
            message = error;
        } else if (type === 'connection') {
            message = 'Connection error: ' + error;
        } else if (type === 'initialization') {
            message = 'Failed to initialize call: ' + error.message;
        }
        
        this.showError(message);
    }

    /**
     * Toggle microphone mute
     */
    toggleMute() {
        this.isMuted = this.webrtc.toggleMute();
        
        // Update UI
        const muteBtn = this.elements.muteBtn;
        if (muteBtn) {
            const icon = muteBtn.querySelector('i');
            const label = muteBtn.querySelector('.control-label');
            
            if (this.isMuted) {
                muteBtn.classList.add('muted');
                icon.className = 'fas fa-microphone-slash';
                label.textContent = 'Unmute';
            } else {
                muteBtn.classList.remove('muted');
                icon.className = 'fas fa-microphone';
                label.textContent = 'Mute';
            }
        }
        
        // Send status update
        this.sendStatusUpdate('mute', { muted: this.isMuted });
    }

    /**
     * Toggle speaker
     */
    toggleSpeaker() {
        const speakerBtn = this.elements.speakerBtn;
        const isSpeakerOn = speakerBtn?.classList.contains('speaker-on');
        
        if (speakerBtn) {
            if (isSpeakerOn) {
                speakerBtn.classList.remove('speaker-on');
                this.setVolume(50);
            } else {
                speakerBtn.classList.add('speaker-on');
                this.setVolume(100);
            }
        }
    }

    /**
     * Set audio volume
     */
    setVolume(volume) {
        this.currentVolume = Math.max(0, Math.min(100, volume));
        this.webrtc.setVolume(this.currentVolume);
        
        if (this.elements.volumeSlider) {
            this.elements.volumeSlider.value = this.currentVolume;
        }
    }

    /**
     * Toggle chat sidebar
     */
    toggleChat() {
        const chatSidebar = this.elements.chatSidebar;
        const toggleBtn = this.elements.chatToggleBtn;
        
        if (chatSidebar) {
            chatSidebar.classList.toggle('collapsed');
            
            if (toggleBtn) {
                const isCollapsed = chatSidebar.classList.contains('collapsed');
                toggleBtn.classList.toggle('active', !isCollapsed);
            }
            
            // Reset unread count when opening chat
            if (!chatSidebar.classList.contains('collapsed')) {
                this.unreadCount = 0;
                this.updateChatBadge();
            }
        }
    }

    /**
     * Send chat message
     */
    async sendChatMessage() {
        const input = this.elements.chatInput;
        const message = input?.value.trim();
        
        if (!message) return;
        
        try {
            const response = await fetch(`${this.apiBase}/send-message/${this.callId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({
                    message_text: message,
                    language: this.elements.messageLanguage?.value || 'en',
                    auto_translate: this.elements.autoTranslate?.checked || false
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to send message');
            }
            
            // Clear input
            if (input) {
                input.value = '';
            }
            
        } catch (error) {
            console.error('❌ Failed to send chat message:', error);
            this.showError('Failed to send message');
        }
    }

    /**
     * Handle incoming chat message
     */
    handleChatMessage(message) {
        this.addChatMessage({
            senderId: message.sender_id,
            senderName: message.sender_name,
            text: message.message_text,
            originalLanguage: message.original_language,
            translations: message.translated_content,
            timestamp: new Date(message.sent_at),
            isOwn: false
        });
        
        // Increment unread count if chat is closed
        if (this.elements.chatSidebar?.classList.contains('collapsed')) {
            this.unreadCount++;
            this.updateChatBadge();
        }
    }

    /**
     * Add message to chat UI
     */
    addChatMessage(messageData) {
        const messagesContainer = this.elements.chatMessages;
        if (!messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${messageData.isOwn ? 'own' : 'other'}`;
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        headerDiv.textContent = `${messageData.senderName} • ${messageData.timestamp.toLocaleTimeString()}`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = messageData.text;
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(textDiv);
        
        // Add translations if available
        if (messageData.translations) {
            const translationDiv = document.createElement('div');
            translationDiv.className = 'message-translation';
            
            Object.entries(messageData.translations).forEach(([lang, translation]) => {
                const langDiv = document.createElement('div');
                langDiv.innerHTML = `<strong>${this.getLanguageName(lang)}:</strong> ${translation}`;
                translationDiv.appendChild(langDiv);
            });
            
            messageDiv.appendChild(translationDiv);
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    /**
     * Get language display name
     */
    getLanguageName(code) {
        const languages = {
            'en': 'English',
            'fr': 'Français',
            'ar': 'العربية'
        };
        return languages[code] || code;
    }

    /**
     * Handle translated audio from other participant
     */
    handleTranslatedAudio(message) {
        console.log('🔊 Received translated audio:', message);
        
        try {
            if (!message.audio_data) {
                console.warn('No audio data in translated audio message');
                return;
            }
            
            // Convert base64 audio data to blob
            const audioData = atob(message.audio_data);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioArray[i] = audioData.charCodeAt(i);
            }
            
            const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Create and play audio element
            const audio = new Audio(audioUrl);
            audio.volume = 0.8; // Slightly lower volume for translations
            
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
            
            audio.onerror = (error) => {
                console.error('❌ Failed to play translated audio:', error);
                URL.revokeObjectURL(audioUrl);
            };
            
            // Play the translated audio
            audio.play().catch(error => {
                console.error('❌ Failed to play translated audio:', error);
            });
            
            // Show translation info in UI
            this.showTranslationNotification(message);
            
        } catch (error) {
            console.error('❌ Error handling translated audio:', error);
        }
    }

    /**
     * Handle translation state changes
     */
    handleTranslationState(message) {
        console.log('🔄 Translation state update:', message);
        
        try {
            const { user_id, is_translating, source_language, target_language } = message;
            
            // Update UI to show translation status
            this.updateTranslationStatus(user_id, is_translating, source_language, target_language);
            
        } catch (error) {
            console.error('❌ Error handling translation state:', error);
        }
    }

    /**
     * Show translation notification in UI
     */
    showTranslationNotification(message) {
        const { source_text, translated_text, source_language, target_language } = message;
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'translation-notification';
        notification.innerHTML = `
            <div class="translation-header">
                <span class="translation-icon">🌐</span>
                <span class="translation-title">Translation</span>
            </div>
            <div class="translation-content">
                <div class="source-text">
                    <strong>${this.getLanguageName(source_language)}:</strong> ${source_text}
                </div>
                <div class="translated-text">
                    <strong>${this.getLanguageName(target_language)}:</strong> ${translated_text}
                </div>
            </div>
        `;
        
        // Add to UI
        const container = document.querySelector('.voice-call-container') || document.body;
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    /**
     * Update translation status in UI
     */
    updateTranslationStatus(userId, isTranslating, sourceLanguage, targetLanguage) {
        // Find or create status indicator
        let statusElement = document.querySelector(`[data-translation-status="${userId}"]`);
        
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.className = 'translation-status';
            statusElement.setAttribute('data-translation-status', userId);
            
            // Add to participant area or create one
            const participantArea = document.querySelector('.participants-area') || 
                                  document.querySelector('.voice-call-container');
            if (participantArea) {
                participantArea.appendChild(statusElement);
            }
        }
        
        if (isTranslating) {
            statusElement.innerHTML = `
                <span class="translation-indicator active">
                    🌐 Translating: ${this.getLanguageName(sourceLanguage)} → ${this.getLanguageName(targetLanguage)}
                </span>
            `;
            statusElement.classList.add('active');
        } else {
            statusElement.innerHTML = '';
            statusElement.classList.remove('active');
        }
    }

    /**
     * Update chat badge with unread count
     */
    updateChatBadge() {
        const toggleBtn = this.elements.chatToggleBtn;
        if (!toggleBtn) return;
        
        let badge = toggleBtn.querySelector('.badge');
        
        if (this.unreadCount > 0) {
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'badge bg-danger position-absolute top-0 start-100 translate-middle';
                toggleBtn.style.position = 'relative';
                toggleBtn.appendChild(badge);
            }
            badge.textContent = this.unreadCount > 99 ? '99+' : this.unreadCount.toString();
        } else if (badge) {
            badge.remove();
        }
    }

    /**
     * Send status update to other party
     */
    sendStatusUpdate(status, data) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'call_status',
                status: status,
                data: data
            }));
        }
    }

    /**
     * Handle status updates from other party
     */
    handleStatusUpdate(message) {
        console.log('📊 Status update:', message.status, message.data);
        
        if (message.status === 'mute') {
            // Update UI to show other party is muted
            // Could add a visual indicator
        }
    }

    /**
     * Toggle statistics panel
     */
    toggleStats() {
        const statsPanel = this.elements.callStats;
        if (statsPanel) {
            statsPanel.classList.toggle('collapsed');
        }
    }

    /**
     * Update call statistics display
     */
    updateStats() {
        const state = this.webrtc.getConnectionState();
        
        if (this.elements.latencyValue) {
            this.elements.latencyValue.textContent = `${state.stats.latency || 0} ms`;
        }
        
        if (this.elements.qualityValue) {
            this.elements.qualityValue.textContent = this.formatQuality(state.quality);
            this.elements.qualityValue.className = `quality-${state.quality}`;
        }
        
        if (this.elements.packetLossValue) {
            const packetLoss = state.stats.packetLossPercentage || 0;
            this.elements.packetLossValue.textContent = `${packetLoss.toFixed(1)}%`;
        }
        
        if (this.elements.bitrateValue) {
            const bitrate = Math.round((state.stats.bytesSent + state.stats.bytesReceived) / 1000);
            this.elements.bitrateValue.textContent = `${bitrate} kbps`;
        }
    }

    /**
     * Format quality text
     */
    formatQuality(quality) {
        return quality.charAt(0).toUpperCase() + quality.slice(1);
    }

    /**
     * Start call duration timer
     */
    startCallTimer() {
        this.startTime = Date.now();
        
        this.durationInterval = setInterval(() => {
            const duration = Date.now() - this.startTime;
            this.updateCallDuration(duration);
            this.updateStats();
        }, 1000);
    }

    /**
     * Update call duration display
     */
    updateCallDuration(duration) {
        const minutes = Math.floor(duration / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        const formatted = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        if (this.elements.callDuration) {
            this.elements.callDuration.textContent = formatted;
        }
    }

    /**
     * Update call status display
     */
    updateCallStatus(status) {
        if (this.elements.callStatus) {
            this.elements.callStatus.textContent = status;
        }
    }

    /**
     * Update connection status display
     */
    updateConnectionStatus(level, text) {
        const statusElement = this.elements.connectionStatus;
        if (statusElement) {
            statusElement.className = `connection-${level}`;
            statusElement.innerHTML = `<i class="fas fa-signal"></i> ${text}`;
        }
    }

    /**
     * Play ringtone for incoming calls
     */
    playRingtone() {
        if (this.ringtone) {
            this.ringtone.currentTime = 0;
            this.ringtone.play().catch(e => console.log('Could not play ringtone:', e));
        }
    }

    /**
     * Stop ringtone
     */
    stopRingtone() {
        if (this.ringtone) {
            this.ringtone.pause();
            this.ringtone.currentTime = 0;
        }
    }

    /**
     * Play ringback tone for outgoing calls
     */
    playRingback() {
        if (this.ringback) {
            this.ringback.currentTime = 0;
            this.ringback.play().catch(e => console.log('Could not play ringback:', e));
        }
    }

    /**
     * Stop ringback tone
     */
    stopRingback() {
        if (this.ringback) {
            this.ringback.pause();
            this.ringback.currentTime = 0;
        }
    }

    /**
     * Show call ended modal
     */
    showCallEndedModal(reason, duration = 0) {
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
            this.durationInterval = null;
        }
        
        // Update modal content
        if (this.elements.callEndReason) {
            this.elements.callEndReason.textContent = reason;
        }
        
        if (this.elements.callSummaryDuration) {
            const minutes = Math.floor(duration / 60000);
            const seconds = Math.floor((duration % 60000) / 1000);
            const formatted = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            this.elements.callSummaryDuration.textContent = `Duration: ${formatted}`;
        }
        
        // Show modal
        const modal = new bootstrap.Modal(this.elements.callEndedModal);
        modal.show();
        
        // Auto-close window after 30 seconds
        setTimeout(() => {
            window.close();
        }, 30000);
    }

    /**
     * Show error message
     */
    showError(message) {
        // Show bootstrap toast
        const toastElement = document.getElementById('connection-error-toast');
        if (toastElement) {
            const toastBody = toastElement.querySelector('.toast-body');
            if (toastBody) {
                toastBody.textContent = message;
            }
            
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
        } else {
            // Fallback to alert
            alert(message);
        }
    }

    /**
     * Rate call quality
     */
    rateCall(rating) {
        // Update star display
        document.querySelectorAll('.star').forEach((star, index) => {
            star.classList.toggle('active', index < rating);
        });
        
        this.callRating = rating;
    }

    /**
     * Get call rating
     */
    getCallRating() {
        return this.callRating || 3;
    }

    /**
     * Call again (redial)
     */
    callAgain() {
        // Close current window and open new call
        const newUrl = `${window.location.origin}/voice-call?participant=${encodeURIComponent(this.participantName)}`;
        window.open(newUrl, '_blank');
        window.close();
    }

    /**
     * Minimize window
     */
    minimizeWindow() {
        // This would be handled by the parent window
        if (window.opener) {
            window.opener.postMessage({ type: 'minimize_call', callId: this.callId }, '*');
        }
        window.blur();
    }

    /**
     * Check if call is active
     */
    isActive() {
        return this.callState === 'active' || this.callState === 'connecting' || this.callState === 'ringing';
    }

    /**
     * Toggle real-time translation
     */
    toggleTranslation(enabled) {
        this.translationEnabled = enabled;
        
        if (this.elements.translationSettings) {
            this.elements.translationSettings.style.display = enabled ? 'block' : 'none';
        }
        
        if (enabled) {
            this.startVoiceTranslation();
        } else {
            this.stopVoiceTranslation();
        }
        
        // Notify other participant about translation state
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'translation_state',
                enabled: enabled,
                myLanguage: this.elements.myLanguage?.value || 'en',
                translateTo: this.elements.translateTo?.value || 'en'
            }));
        }
    }

    /**
     * Update translation settings
     */
    updateTranslationSettings() {
        if (!this.translationEnabled) return;
        
        const myLang = this.elements.myLanguage?.value || 'en';
        const targetLang = this.elements.translateTo?.value || 'en';
        
        console.log(`🌐 Translation: ${myLang} → ${targetLang}`);
        
        // Notify other participant about language changes
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'translation_settings',
                myLanguage: myLang,
                translateTo: targetLang
            }));
        }
    }

    /**
     * Start voice translation
     */
    async startVoiceTranslation() {
        console.log('🌐 Starting voice translation...');
        
        try {
            // Set up audio processing for translation
            this.translationAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.translationProcessor = this.translationAudioContext.createScriptProcessor(4096, 1, 1);
            
            // Buffer for collecting audio chunks
            this.audioBuffer = [];
            this.lastTranslationTime = 0;
            this.translationDelay = 3000; // 3 seconds between translations
            
            // Connect to local audio stream for processing
            if (this.webrtc && this.webrtc.localStream) {
                const source = this.translationAudioContext.createMediaStreamSource(this.webrtc.localStream);
                source.connect(this.translationProcessor);
                this.translationProcessor.connect(this.translationAudioContext.destination);
                
                this.translationProcessor.onaudioprocess = (event) => {
                    this.processAudioForTranslation(event);
                };
                
                console.log('✅ Voice translation started');
            } else {
                console.warn('⚠️ No local audio stream available for translation');
            }
            
        } catch (error) {
            console.error('❌ Failed to start voice translation:', error);
        }
    }

    /**
     * Process audio for translation
     */
    processAudioForTranslation(event) {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Simple voice activity detection
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
            sum += Math.abs(inputData[i]);
        }
        const average = sum / inputData.length;
        
        // If there's significant audio activity
        if (average > 0.01) {
            this.audioBuffer.push(...inputData);
            
            // Process every 3 seconds or when buffer gets large
            const now = Date.now();
            if (now - this.lastTranslationTime > this.translationDelay || this.audioBuffer.length > 132300) { // ~3 seconds at 44.1kHz
                this.sendAudioForTranslation();
                this.lastTranslationTime = now;
            }
        }
    }

    /**
     * Send audio for translation
     */
    async sendAudioForTranslation() {
        if (this.audioBuffer.length === 0) return;
        
        try {
            console.log('🔊 Sending audio for translation...');
            
            // Convert audio buffer to WAV blob
            const audioBlob = this.createWavBlob(this.audioBuffer, this.translationAudioContext.sampleRate);
            
            // Create form data for API request
            const formData = new FormData();
            formData.append('audio', audioBlob, 'voice.wav');
            formData.append('language', this.elements.myLanguage?.value || 'en');
            formData.append('target_language', this.elements.translateTo?.value || 'en');
            formData.append('call_id', this.callId);
            
            // Send to translation API
            const response = await fetch('/api/voice/translate-realtime', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                },
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Translation result:', result);
                
                // Send translated audio to other participant
                if (result.translated_audio_url) {
                    this.sendTranslatedAudio(result);
                }
            }
            
            // Clear buffer
            this.audioBuffer = [];
            
        } catch (error) {
            console.error('❌ Translation failed:', error);
            this.audioBuffer = []; // Clear buffer to prevent memory buildup
        }
    }

    /**
     * Create WAV blob from audio buffer
     */
    createWavBlob(buffer, sampleRate) {
        const length = buffer.length;
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        
        // WAV file header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert float samples to 16-bit PCM
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, buffer[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
        
        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    /**
     * Send translated audio to other participant
     */
    sendTranslatedAudio(translationResult) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'translated_audio',
                audio_url: translationResult.translated_audio_url,
                original_text: translationResult.original_text,
                translated_text: translationResult.translated_text,
                source_language: translationResult.source_language,
                target_language: translationResult.target_language
            }));
        }
    }

    /**
     * Stop voice translation
     */
    stopVoiceTranslation() {
        console.log('🌐 Stopping voice translation...');
        
        if (this.translationProcessor) {
            this.translationProcessor.disconnect();
            this.translationProcessor = null;
        }
        
        if (this.translationAudioContext) {
            this.translationAudioContext.close();
            this.translationAudioContext = null;
        }
        
        this.audioBuffer = [];
        this.translationEnabled = false;
        
        console.log('✅ Voice translation stopped');
    }

    /**
     * Cleanup when call ends
     */
    cleanup() {
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
        }
        
        this.stopRingtone();
        this.stopRingback();
        
        if (this.websocket) {
            this.websocket.close();
        }
        
        this.webrtc.close();
    }
}

// Global instance
window.VoiceCall = new VoiceCall();
