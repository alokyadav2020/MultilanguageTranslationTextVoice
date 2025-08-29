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
            testAudioBtn: document.getElementById('test-audio-btn'),
            testMicBtn: document.getElementById('test-mic-btn'),
            debugAudioBtn: document.getElementById('debug-audio-btn'),
            
            // Volume indicators
            myVolume: document.getElementById('my-volume'),
            theirVolume: document.getElementById('their-volume'),
            
            // Chat elements
            chatSidebar: document.getElementById('chat-sidebar'),
            chatMessages: document.getElementById('chat-messages'),
            chatInput: document.getElementById('chat-input'),
            sendMessage: document.getElementById('send-message'),
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
            closeWindowBtn: document.getElementById('close-window-btn'),
            
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
            // Immediate visual feedback
            if (this.elements.endCallBtn) {
                this.elements.endCallBtn.style.backgroundColor = '#dc3545';
                this.elements.endCallBtn.style.opacity = '0.7';
            }
            this.endCall();
        });
        
        // Double-click end call for immediate close
        this.elements.endCallBtn?.addEventListener('dblclick', () => {
            console.log('📞 End call button double-clicked - force close');
            this.callState = 'ended';
            this.closeWindow();
        });
        
        this.elements.closeCallBtn?.addEventListener('click', () => {
            console.log('❌ Close call button clicked');
            // Immediate visual feedback
            if (this.elements.closeCallBtn) {
                this.elements.closeCallBtn.style.backgroundColor = '#dc3545';
                this.elements.closeCallBtn.style.opacity = '0.7';
            }
            this.endCall();
        });
        
        // Double-click close call for immediate close
        this.elements.closeCallBtn?.addEventListener('dblclick', () => {
            console.log('❌ Close call button double-clicked - force close');
            this.callState = 'ended';
            this.closeWindow();
        });
        this.elements.minimizeBtn?.addEventListener('click', () => {
            console.log('➖ Minimize button clicked');
            this.minimizeWindow();
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
        
        // Test audio button
        this.elements.testAudioBtn?.addEventListener('click', () => {
            this.testAudio();
        });
        
        // Test microphone button
        this.elements.testMicBtn?.addEventListener('click', () => {
            this.testMicrophone();
        });
        
        // Debug audio button
        this.elements.debugAudioBtn?.addEventListener('click', () => {
            this.debugAudioState();
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
        this.elements.closeWindowBtn?.addEventListener('click', () => {
            console.log('🚪 Close window button clicked');
            this.closeWindow();
        });
        
        // Modal close buttons
        document.querySelectorAll('[data-bs-dismiss="modal"]').forEach(btn => {
            btn.addEventListener('click', () => {
                console.log('📋 Modal close button clicked');
                setTimeout(() => this.closeWindow(), 500); // Short delay for modal to close
            });
        });
        
        // Add keyboard shortcut to close window
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.callState === 'ended') {
                this.closeWindow();
            }
        });
        
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
            console.log('📞 endCall() method called, current state:', this.callState);
            
            if (this.callState === 'ending' || this.callState === 'ended') {
                console.log('⏹️ Call already ending/ended, skipping...');
                return;
            }
            
            console.log('📞 Ending call...');
            
            this.callState = 'ending';
            this.updateCallStatus('Ending call...');
            
            // Disable end call buttons to prevent multiple clicks
            if (this.elements.endCallBtn) {
                this.elements.endCallBtn.disabled = true;
            }
            if (this.elements.closeCallBtn) {
                this.elements.closeCallBtn.disabled = true;
            }
            
            // Stop audio
            this.stopRingtone();
            this.stopRingback();
            
            // Close WebRTC
            if (this.webrtc) {
                this.webrtc.close();
            }
            
            // Calculate duration
            const duration = this.startTime ? Date.now() - this.startTime : 0;
            
            // Send end call to server
            try {
                console.log('📤 Sending end call request to server...');
                const response = await fetch(`${this.apiBase}/end/${this.callId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.token}`
                    },
                    body: JSON.stringify({
                        end_reason: 'normal',
                        quality_score: this.getCallRating(),
                        connection_quality: this.webrtc ? this.webrtc.getConnectionQuality() : null
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('✅ Server confirmed call end:', result);
                } else {
                    console.warn('⚠️ Server responded with error:', response.status, response.statusText);
                }
            } catch (apiError) {
                console.warn('⚠️ Failed to notify server of call end:', apiError);
                // Continue with local cleanup even if server notification fails
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
            
            // Check if we have audio tracks
            const audioTracks = stream.getAudioTracks();
            console.log('🎵 Remote audio tracks:', audioTracks.length);
            audioTracks.forEach((track, index) => {
                console.log(`🎵 Remote audio track ${index}:`, {
                    enabled: track.enabled,
                    muted: track.muted,
                    readyState: track.readyState,
                    settings: track.getSettings()
                });
            });
            
            // Explicitly play the remote audio
            this.elements.remoteAudio.play().then(() => {
                console.log('✅ Remote audio playback started');
            }).catch(error => {
                console.error('❌ Failed to play remote audio:', error);
                console.log('🔊 Trying to enable audio after user interaction...');
                
                // Show a message to user to click to enable audio
                this.showAudioPermissionMessage();
                
                // Retry after a short delay
                setTimeout(() => {
                    this.elements.remoteAudio.play().catch(err => {
                        console.error('❌ Remote audio retry failed:', err);
                    });
                }, 1000);
            });
            
            // Add additional event listeners for debugging
            this.elements.remoteAudio.addEventListener('loadstart', () => {
                console.log('🎵 Remote audio load started');
            });
            
            this.elements.remoteAudio.addEventListener('loadeddata', () => {
                console.log('🎵 Remote audio data loaded');
            });
            
            this.elements.remoteAudio.addEventListener('canplay', () => {
                console.log('🎵 Remote audio can play');
                // Try to play again when ready
                this.elements.remoteAudio.play().catch(err => {
                    console.log('🔊 Still cannot autoplay, user interaction needed');
                });
            });
            
            this.elements.remoteAudio.addEventListener('playing', () => {
                console.log('✅ Remote audio is playing');
                this.hideAudioPermissionMessage();
            });
            
            this.elements.remoteAudio.addEventListener('pause', () => {
                console.log('⏸️ Remote audio paused');
            });
            
            this.elements.remoteAudio.addEventListener('error', (e) => {
                console.error('❌ Remote audio error:', e);
            });
        }
        
        // Start audio visualization
        if (window.AudioVisualization) {
            window.AudioVisualization.setRemoteStream(stream);
        }
    }

    /**
     * Show message to click for audio permission
     */
    showAudioPermissionMessage() {
        if (!this.audioPermissionShown) {
            this.audioPermissionShown = true;
            
            // Create a floating message
            const message = document.createElement('div');
            message.id = 'audio-permission-message';
            message.className = 'alert alert-warning position-fixed';
            message.style.cssText = `
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 9999;
                min-width: 300px;
                text-align: center;
                cursor: pointer;
            `;
            message.innerHTML = `
                <strong>🔊 Click here to enable audio</strong><br>
                <small>Browser requires user interaction to play audio</small>
            `;
            
            message.addEventListener('click', () => {
                if (this.elements.remoteAudio) {
                    this.elements.remoteAudio.play().then(() => {
                        console.log('✅ Audio enabled after user click');
                        this.hideAudioPermissionMessage();
                    }).catch(err => {
                        console.error('❌ Still failed to play audio:', err);
                    });
                }
            });
            
            document.body.appendChild(message);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                this.hideAudioPermissionMessage();
            }, 10000);
        }
    }

    /**
     * Hide audio permission message
     */
    hideAudioPermissionMessage() {
        const message = document.getElementById('audio-permission-message');
        if (message) {
            message.remove();
        }
        this.audioPermissionShown = false;
    }

    /**
     * Test audio playback
     */
    testAudio() {
        console.log('🔊 Testing audio...');
        
        // Try to play remote audio if available
        if (this.elements.remoteAudio && this.elements.remoteAudio.srcObject) {
            console.log('🔊 Testing remote audio playback...');
            this.elements.remoteAudio.play().then(() => {
                console.log('✅ Remote audio test successful');
                this.showTemporaryMessage('✅ Audio test successful', 'success');
            }).catch(error => {
                console.error('❌ Remote audio test failed:', error);
                this.showTemporaryMessage('❌ Audio test failed - ' + error.message, 'danger');
            });
        } else {
            // Play a test tone
            this.playTestTone();
        }
    }

    /**
     * Play a test tone to verify audio
     */
    playTestTone() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            
            oscillator.start();
            oscillator.stop(audioContext.currentTime + 0.5);
            
            console.log('✅ Test tone played');
            this.showTemporaryMessage('✅ Test tone played - speakers working', 'success');
            
        } catch (error) {
            console.error('❌ Failed to play test tone:', error);
            this.showTemporaryMessage('❌ Audio test failed - ' + error.message, 'danger');
        }
    }

    /**
     * Show temporary message
     */
    showTemporaryMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} position-fixed`;
        alertDiv.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
        `;
        alertDiv.textContent = message;
        
        document.body.appendChild(alertDiv);
        
        setTimeout(() => {
            alertDiv.remove();
        }, 3000);
    }

    /**
     * Test microphone
     */
    testMicrophone() {
        console.log('🎤 Testing microphone...');
        
        if (this.webrtc && this.webrtc.localStream) {
            const audioTracks = this.webrtc.localStream.getAudioTracks();
            
            if (audioTracks.length > 0) {
                const track = audioTracks[0];
                console.log('🎤 Microphone track status:', {
                    enabled: track.enabled,
                    muted: track.muted,
                    readyState: track.readyState
                });
                
                if (track.readyState === 'live' && track.enabled && !track.muted) {
                    this.showTemporaryMessage('✅ Microphone is working', 'success');
                    
                    // Try to measure audio level
                    this.measureMicrophoneLevel();
                } else {
                    this.showTemporaryMessage('❌ Microphone issue detected', 'warning');
                }
            } else {
                this.showTemporaryMessage('❌ No microphone track found', 'danger');
            }
        } else {
            this.showTemporaryMessage('❌ No audio stream available', 'danger');
        }
    }

    /**
     * Measure microphone audio level
     */
    measureMicrophoneLevel() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(this.webrtc.localStream);
            
            source.connect(analyser);
            analyser.fftSize = 256;
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            let measurements = 0;
            let maxLevel = 0;
            
            const measureLevel = () => {
                analyser.getByteFrequencyData(dataArray);
                
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const average = sum / dataArray.length;
                maxLevel = Math.max(maxLevel, average);
                
                measurements++;
                
                if (measurements < 30) { // Measure for about 1 second
                    requestAnimationFrame(measureLevel);
                } else {
                    if (maxLevel > 5) {
                        this.showTemporaryMessage(`✅ Microphone level detected: ${Math.round(maxLevel)}`, 'success');
                    } else {
                        this.showTemporaryMessage('⚠️ Very low microphone level detected', 'warning');
                    }
                    
                    audioContext.close();
                }
            };
            
            measureLevel();
            
        } catch (error) {
            console.error('❌ Failed to measure microphone level:', error);
            this.showTemporaryMessage('❌ Could not measure microphone level', 'warning');
        }
    }

    /**
     * Debug current audio state
     */
    debugAudioState() {
        console.log('🔍 === AUDIO DEBUG REPORT ===');
        
        // Debug WebRTC state
        if (this.webrtc) {
            console.log('📞 WebRTC State:', {
                connectionState: this.webrtc.connectionState,
                iceConnectionState: this.webrtc.iceConnectionState,
                signallingState: this.webrtc.signallingState
            });
            
            // Call WebRTC debug method
            this.webrtc.debugAudioTracks();
        } else {
            console.error('❌ No WebRTC instance available');
        }
        
        // Debug audio elements
        if (this.elements.remoteAudio) {
            console.log('🎵 Remote Audio Element:', {
                src: this.elements.remoteAudio.src,
                srcObject: !!this.elements.remoteAudio.srcObject,
                volume: this.elements.remoteAudio.volume,
                muted: this.elements.remoteAudio.muted,
                paused: this.elements.remoteAudio.paused,
                readyState: this.elements.remoteAudio.readyState,
                networkState: this.elements.remoteAudio.networkState,
                currentTime: this.elements.remoteAudio.currentTime,
                duration: this.elements.remoteAudio.duration
            });
        } else {
            console.error('❌ No remote audio element found');
        }
        
        if (this.elements.localAudio) {
            console.log('🎤 Local Audio Element:', {
                src: this.elements.localAudio.src,
                srcObject: !!this.elements.localAudio.srcObject,
                volume: this.elements.localAudio.volume,
                muted: this.elements.localAudio.muted,
                paused: this.elements.localAudio.paused
            });
        } else {
            console.error('❌ No local audio element found');
        }
        
        // Debug call state
        console.log('📞 Call State:', {
            callState: this.callState,
            callId: this.callId,
            currentVolume: this.currentVolume,
            isInitiator: this.isInitiator
        });
        
        console.log('🔍 === END AUDIO DEBUG REPORT ===');
        
        this.showTemporaryMessage('🔍 Audio debug completed - check console', 'info');
    }

    /**
     * Handle local audio stream
     */
    handleLocalStream(stream) {
        console.log('🎤 Setting up local audio stream');
        
        if (this.elements.localAudio) {
            this.elements.localAudio.srcObject = stream;
            // Don't play local audio to avoid feedback
            this.elements.localAudio.muted = true;
        }
        
        // Verify microphone is working
        const audioTrack = stream.getAudioTracks()[0];
        if (audioTrack) {
            console.log('🎤 Audio track details:', {
                enabled: audioTrack.enabled,
                muted: audioTrack.muted,
                readyState: audioTrack.readyState,
                settings: audioTrack.getSettings()
            });
            
            // Check if audio is actually being captured
            if (audioTrack.readyState === 'live') {
                console.log('✅ Microphone is live and capturing audio');
            } else {
                console.warn('⚠️ Microphone track not in live state:', audioTrack.readyState);
            }
        } else {
            console.error('❌ No audio track found in local stream');
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
                    message_text: message
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
            if (!message.audio_url) {
                console.warn('No audio URL in translated audio message');
                return;
            }
            
            // Create and play audio element using the URL
            const audio = new Audio(message.audio_url);
            audio.volume = 0.8; // Slightly lower volume for translations
            
            audio.onerror = (error) => {
                console.error('❌ Failed to play translated audio:', error);
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
        console.log('📋 Showing call ended modal:', reason, duration);
        
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
        
        // Show modal if available
        if (this.elements.callEndedModal) {
            try {
                const modal = new bootstrap.Modal(this.elements.callEndedModal);
                modal.show();
                console.log('✅ Call ended modal displayed');
            } catch (error) {
                console.error('❌ Failed to show modal:', error);
                // Fallback to immediate close
                this.closeWindow();
                return;
            }
        } else {
            console.warn('⚠️ Call ended modal not found, closing immediately');
            this.closeWindow();
            return;
        }
        
        // Auto-close window after 5 seconds (reduced from 30)
        setTimeout(() => {
            console.log('⏰ Auto-closing window after 5 seconds');
            this.closeWindow();
        }, 5000);
    }

    /**
     * Close the call window safely
     */
    closeWindow() {
        try {
            // Cleanup before closing
            this.cleanup();
            
            // Try different methods to close the window
            if (window.opener) {
                window.close();
            } else if (window.parent !== window) {
                window.parent.postMessage('closeVoiceCall', '*');
            } else {
                // Last resort - redirect back to chat
                window.location.href = '/chat';
            }
        } catch (error) {
            console.error('❌ Error closing window:', error);
            // Final fallback
            window.location.href = '/chat';
        }
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
