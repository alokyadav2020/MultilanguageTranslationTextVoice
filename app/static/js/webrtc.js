// filepath: app/static/js/webrtc.js

/**
 * WebRTC Voice Call Implementation
 * Handles peer-to-peer audio communication for voice        // Connection state change han        // ICE connection state change handler
        this.peerConnection.oniceconnectionstatechange = () => {
            this.iceConnectionState = this.peerConnection.iceConnectionState;
            console.log('🧊 ICE connection state:', this.iceConnectionState);
            
            // Add detailed logging for different ICE states
            switch (this.iceConnectionState) {
                case 'connected':
                    console.log('✅ ICE connected - checking audio tracks...');
                    this.debugAudioTracks();
                    // Start monitoring audio transmission
                    this.startAudioLevelMonitoring();
                    break;
                case 'disconnected':
                    console.warn('⚠️ ICE disconnected - audio may stop working');
                    this.stopAudioLevelMonitoring();
                    break;
                case 'failed':
                    console.error('❌ ICE connection failed - audio will not work');
                    this.stopAudioLevelMonitoring();
                    break;
                case 'checking':
                    console.log('🔍 ICE checking connectivity...');
                    break;
            }
            
            if (this.onIceConnectionStateChange) {
                this.onIceConnectionStateChange(this.iceConnectionState);
            }
        };     this.peerConnection.onconnectionstatechange = () => {
            this.connectionState = this.peerConnection.connectionState;
            console.log('🔄 Connection state:', this.connectionState);
            
            // Debug all senders and receivers when connection state changes
            const senders = this.peerConnection.getSenders();
            const receivers = this.peerConnection.getReceivers();
            console.log(`📊 Connection state ${this.connectionState}: ${senders.length} senders, ${receivers.length} receivers`);
            
            senders.forEach((sender, i) => {
                if (sender.track) {
                    console.log(`📤 Sender ${i}:`, sender.track.kind, sender.track.enabled, sender.track.readyState);
                }
            });
            
            receivers.forEach((receiver, i) => {
                if (receiver.track) {
                    console.log(`📥 Receiver ${i}:`, receiver.track.kind, receiver.track.enabled, receiver.track.readyState);
                }
            });
            
            if (this.onConnectionStateChange) {
                this.onConnectionStateChange(this.connectionState);
            }
        };*/

class WebRTCManager {
    constructor() {
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.websocket = null;
        this.isInitiator = false;
        this.callId = null;
        this.isInitialized = false;
        this.initializationPromise = null;
        this.config = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
                { urls: 'stun:stun2.l.google.com:19302' }
            ],
            iceCandidatePoolSize: 10
        };
        
        // Event handlers
        this.onConnectionStateChange = null;
        this.onIceConnectionStateChange = null;
        this.onRemoteStreamReceived = null;
        this.onLocalStreamReady = null;
        this.onDataChannelMessage = null;
        this.onError = null;
        
        // Connection state tracking
        this.connectionState = 'new';
        this.iceConnectionState = 'new';
        this.signallingState = 'stable';
        
        // Audio constraints
        this.audioConstraints = {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 44100,
                sampleSize: 16,
                channelCount: 1
            },
            video: false
        };
        
        // Statistics tracking
        this.stats = {
            bytesSent: 0,
            bytesReceived: 0,
            packetsLost: 0,
            jitter: 0,
            latency: 0,
            audioLevel: 0
        };
        
        this.statsInterval = null;
    }

    /**
     * Initialize WebRTC connection
     */
    async initialize(callId, websocket, isInitiator = false) {
        // Prevent multiple initializations
        if (this.isInitialized) {
            console.log('⚠️ WebRTC already initialized');
            return true;
        }
        
        if (this.initializationPromise) {
            console.log('⏳ WebRTC initialization in progress, waiting...');
            return await this.initializationPromise;
        }
        
        this.initializationPromise = this._performInitialization(callId, websocket, isInitiator);
        return await this.initializationPromise;
    }

    /**
     * Internal initialization method
     */
    async _performInitialization(callId, websocket, isInitiator) {
        try {
            this.callId = callId;
            this.websocket = websocket;
            this.isInitiator = isInitiator;
            
            console.log('🔗 Initializing WebRTC for call:', callId, 'isInitiator:', isInitiator);
            
            // Create peer connection FIRST
            this.peerConnection = new RTCPeerConnection(this.config);
            this.setupPeerConnectionHandlers();
            
            // Get user media and add tracks BEFORE creating offer
            console.log('🎤 Getting user media...');
            await this.getUserMedia();
            
            // Verify tracks were added
            const senders = this.peerConnection.getSenders();
            console.log('📤 Senders after getUserMedia:', senders.length);
            
            if (senders.length === 0) {
                throw new Error('No audio tracks were added to peer connection');
            }
            
            // Add a small delay to ensure everything is set up
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Mark as initialized before creating offer
            this.isInitialized = true;
            
            // If initiator, create and send offer AFTER tracks are added
            if (this.isInitiator) {
                console.log('📞 Creating offer as initiator...');
                await this.createOffer();
            } else {
                console.log('📞 Waiting for offer as responder...');
            }
            
            // Start collecting statistics
            this.startStatsCollection();
            
            console.log('✅ WebRTC initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ WebRTC initialization failed:', error);
            this.isInitialized = false;
            this.initializationPromise = null;
            if (this.onError) {
                this.onError('initialization', error);
            }
            return false;
        }
    }

    /**
     * Setup peer connection event handlers
     */
    setupPeerConnectionHandlers() {
        // Ice candidate handler
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('🧊 Sending ICE candidate');
                this.sendSignalingMessage({
                    type: 'ice_candidate',
                    candidate: event.candidate
                });
            }
        };

        // Remote stream handler
        this.peerConnection.ontrack = (event) => {
            console.log('🎵 Received remote track:', event.track.kind);
            console.log('🎵 Track details:', {
                kind: event.track.kind,
                enabled: event.track.enabled,
                muted: event.track.muted,
                readyState: event.track.readyState,
                settings: event.track.getSettings()
            });
            
            this.remoteStream = event.streams[0];
            console.log('🎵 Remote stream tracks:', this.remoteStream.getTracks().length);
            
            if (this.onRemoteStreamReceived) {
                this.onRemoteStreamReceived(this.remoteStream);
            }
        };

        // Connection state change handler
        this.peerConnection.onconnectionstatechange = () => {
            this.connectionState = this.peerConnection.connectionState;
            console.log('🔄 Connection state:', this.connectionState);
            
            if (this.onConnectionStateChange) {
                this.onConnectionStateChange(this.connectionState);
            }
            
            if (this.connectionState === 'failed') {
                this.handleConnectionFailure();
            }
        };

        // ICE connection state change handler
        this.peerConnection.oniceconnectionstatechange = () => {
            this.iceConnectionState = this.peerConnection.iceConnectionState;
            console.log('🧊 ICE connection state:', this.iceConnectionState);
            
            if (this.onIceConnectionStateChange) {
                this.onIceConnectionStateChange(this.iceConnectionState);
            }
            
            if (this.iceConnectionState === 'failed') {
                this.handleIceConnectionFailure();
            }
        };

        // Signaling state change handler
        this.peerConnection.onsignalingstatechange = () => {
            this.signallingState = this.peerConnection.signalingState;
            console.log('📡 Signaling state:', this.signallingState);
        };

        // ICE gathering state change handler
        this.peerConnection.onicegatheringstatechange = () => {
            console.log('🧊 ICE gathering state:', this.peerConnection.iceGatheringState);
        };
    }

    /**
     * Get user media (microphone access)
     */
    async getUserMedia() {
        try {
            console.log('🎤 Requesting microphone access...');
            console.log('🎤 Audio constraints:', this.audioConstraints);
            
            this.localStream = await navigator.mediaDevices.getUserMedia(this.audioConstraints);
            console.log('✅ Got local stream with tracks:', this.localStream.getTracks().length);
            
            // Log each track
            this.localStream.getTracks().forEach((track, index) => {
                console.log(`🎤 Local track ${index}:`, {
                    kind: track.kind,
                    enabled: track.enabled,
                    muted: track.muted,
                    readyState: track.readyState,
                    label: track.label,
                    settings: track.getSettings()
                });
            });
            
            // Add local stream to peer connection
            this.localStream.getTracks().forEach(track => {
                console.log('🎤 Adding local track:', track.kind, track.enabled, track.readyState);
                const sender = this.peerConnection.addTrack(track, this.localStream);
                console.log('📤 Track added to sender:', sender);
            });
            
            // Debug: Check all senders
            const senders = this.peerConnection.getSenders();
            console.log('📤 Total senders after adding tracks:', senders.length);
            senders.forEach((sender, index) => {
                if (sender.track) {
                    console.log(`📤 Sender ${index}:`, {
                        kind: sender.track.kind,
                        enabled: sender.track.enabled,
                        readyState: sender.track.readyState
                    });
                } else {
                    console.log(`📤 Sender ${index}: No track`);
                }
            });
            
            console.log('✅ Microphone access granted');
            
            if (this.onLocalStreamReady) {
                this.onLocalStreamReady(this.localStream);
            }
            
            return this.localStream;
            
        } catch (error) {
            console.error('❌ Failed to get user media:', error);
            
            let errorMessage = 'Failed to access microphone';
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Microphone access denied. Please allow microphone access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No microphone found. Please connect a microphone and try again.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Microphone is being used by another application.';
            }
            
            if (this.onError) {
                this.onError('media', errorMessage);
            }
            
            throw error;
        }
    }

    /**
     * Create and send offer (for call initiator)
     */
    async createOffer() {
        try {
            console.log('📞 Creating offer...');
            
            // Check peer connection state to avoid "wrong state" errors
            if (this.peerConnection.signalingState !== 'stable') {
                console.warn(`⚠️ Cannot create offer in signaling state: ${this.peerConnection.signalingState}`);
                if (this.peerConnection.signalingState === 'have-remote-offer') {
                    console.log('📞 Already have remote offer, should create answer instead');
                    return;
                }
                throw new Error(`Cannot create offer in state: ${this.peerConnection.signalingState}`);
            }
            
            const offer = await this.peerConnection.createOffer({
                offerToReceiveAudio: true,
                offerToReceiveVideo: false
            });
            
            await this.peerConnection.setLocalDescription(offer);
            
            console.log('📞 Sending offer');
            this.sendSignalingMessage({
                type: 'webrtc_signaling',
                signaling_type: 'offer',
                sdp: offer.sdp
            });
            
        } catch (error) {
            console.error('❌ Failed to create offer:', error);
            if (this.onError) {
                this.onError('offer', error);
            }
            throw error;
        }
    }

    /**
     * Handle incoming offer (for call receiver)
     */
    async handleOffer(sdp) {
        try {
            // Wait for initialization if it's in progress
            if (!this.isInitialized && this.initializationPromise) {
                console.log('⏳ Waiting for WebRTC initialization before handling offer...');
                await this.initializationPromise;
            }
            
            if (!this.isInitialized || !this.peerConnection) {
                throw new Error('WebRTC not initialized when handling offer');
            }
            
            console.log('📞 Handling incoming offer...');
            console.log(`📊 Current signaling state: ${this.peerConnection.signalingState}`);
            
            // Check if we can handle an offer in current state
            if (this.peerConnection.signalingState !== 'stable' && 
                this.peerConnection.signalingState !== 'have-local-offer') {
                console.warn(`⚠️ Cannot handle offer in signaling state: ${this.peerConnection.signalingState}`);
                throw new Error(`Cannot handle offer in state: ${this.peerConnection.signalingState}`);
            }
            
            const offer = new RTCSessionDescription({
                type: 'offer',
                sdp: sdp
            });
            
            await this.peerConnection.setRemoteDescription(offer);
            console.log('✅ Remote offer set successfully');
            
            // Ensure we have our own audio tracks added (for call receiver)
            if (!this.localStream) {
                console.log('🎤 Getting user media for call receiver...');
                await this.getUserMedia();
            }
            
            // Verify tracks are added to peer connection
            const senders = this.peerConnection.getSenders();
            console.log(`📊 Current senders before answer: ${senders.length}`);
            
            if (senders.length === 0 && this.localStream) {
                console.log('🔗 Adding tracks to peer connection for receiver...');
                this.localStream.getTracks().forEach(track => {
                    this.peerConnection.addTrack(track, this.localStream);
                    console.log(`➕ Added ${track.kind} track for receiver`);
                });
            }
            
            // Create and send answer
            const answer = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(answer);
            
            console.log('📞 Sending answer');
            this.sendSignalingMessage({
                type: 'webrtc_signaling',
                signaling_type: 'answer',
                sdp: answer.sdp
            });
            
        } catch (error) {
            console.error('❌ Failed to handle offer:', error);
            if (this.onError) {
                this.onError('answer', error);
            }
            throw error;
        }
    }

    /**
     * Handle incoming answer (for call initiator)
     */
    async handleAnswer(sdp) {
        try {
            console.log('📞 Handling incoming answer...');
            
            const answer = new RTCSessionDescription({
                type: 'answer',
                sdp: sdp
            });
            
            await this.peerConnection.setRemoteDescription(answer);
            
            console.log('✅ Answer processed successfully');
            
        } catch (error) {
            console.error('❌ Failed to handle answer:', error);
            if (this.onError) {
                this.onError('answer', error);
            }
            throw error;
        }
    }

    /**
     * Handle incoming ICE candidate
     */
    async handleIceCandidate(candidate) {
        try {
            console.log('🧊 Adding ICE candidate');
            
            const iceCandidate = new RTCIceCandidate(candidate);
            await this.peerConnection.addIceCandidate(iceCandidate);
            
        } catch (error) {
            console.error('❌ Failed to add ICE candidate:', error);
            // Don't throw error for ICE candidates as they're not critical
        }
    }

    /**
     * Send signaling message via WebSocket
     */
    sendSignalingMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                ...message,
                call_id: this.callId
            }));
        } else {
            console.error('❌ WebSocket not available for signaling');
        }
    }

    /**
     * Toggle microphone mute
     */
    toggleMute() {
        if (this.localStream) {
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (audioTrack) {
                audioTrack.enabled = !audioTrack.enabled;
                console.log('🎤', audioTrack.enabled ? 'Unmuted' : 'Muted');
                return !audioTrack.enabled;
            }
        }
        return false;
    }

    /**
     * Check if microphone is muted
     */
    isMuted() {
        if (this.localStream) {
            const audioTrack = this.localStream.getAudioTracks()[0];
            return audioTrack ? !audioTrack.enabled : true;
        }
        return true;
    }

    /**
     * Set output volume
     */
    setVolume(volume) {
        const remoteAudio = document.getElementById('remote-audio');
        if (remoteAudio) {
            remoteAudio.volume = Math.max(0, Math.min(1, volume / 100));
        }
    }

    /**
     * Get audio statistics
     */
    async getAudioStats() {
        if (!this.peerConnection) return null;
        
        try {
            const stats = await this.peerConnection.getStats();
            const audioStats = {};
            
            stats.forEach(report => {
                if (report.type === 'inbound-rtp' && report.mediaType === 'audio') {
                    audioStats.bytesReceived = report.bytesReceived || 0;
                    audioStats.packetsReceived = report.packetsReceived || 0;
                    audioStats.packetsLost = report.packetsLost || 0;
                    audioStats.jitter = report.jitter || 0;
                    audioStats.audioLevel = report.audioLevel || 0;
                } else if (report.type === 'outbound-rtp' && report.mediaType === 'audio') {
                    audioStats.bytesSent = report.bytesSent || 0;
                    audioStats.packetsSent = report.packetsSent || 0;
                } else if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                    audioStats.currentRoundTripTime = report.currentRoundTripTime || 0;
                }
            });
            
            return audioStats;
            
        } catch (error) {
            console.error('❌ Failed to get audio stats:', error);
            return null;
        }
    }

    /**
     * Start statistics collection
     */
    startStatsCollection() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
        }
        
        this.statsInterval = setInterval(async () => {
            const audioStats = await this.getAudioStats();
            if (audioStats) {
                this.stats = { ...this.stats, ...audioStats };
                
                // Calculate packet loss percentage
                if (audioStats.packetsReceived > 0) {
                    this.stats.packetLossPercentage = 
                        (audioStats.packetsLost / (audioStats.packetsReceived + audioStats.packetsLost)) * 100;
                }
                
                // Calculate latency in milliseconds
                if (audioStats.currentRoundTripTime) {
                    this.stats.latency = Math.round(audioStats.currentRoundTripTime * 1000);
                }
            }
        }, 1000);
    }

    /**
     * Stop statistics collection
     */
    stopStatsCollection() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
    }

    /**
     * Debug audio tracks status
     */
    debugAudioTracks() {
        console.log('🔍 Debugging audio tracks...');
        
        // Check local stream
        if (this.localStream) {
            const localAudioTracks = this.localStream.getAudioTracks();
            console.log(`🎤 Local audio tracks: ${localAudioTracks.length}`);
            localAudioTracks.forEach((track, i) => {
                console.log(`🎤 Local track ${i}:`, {
                    kind: track.kind,
                    enabled: track.enabled,
                    muted: track.muted,
                    readyState: track.readyState,
                    label: track.label
                });
            });
        } else {
            console.warn('⚠️ No local stream available');
        }
        
        // Check remote stream
        if (this.remoteStream) {
            const remoteAudioTracks = this.remoteStream.getAudioTracks();
            console.log(`🎵 Remote audio tracks: ${remoteAudioTracks.length}`);
            remoteAudioTracks.forEach((track, i) => {
                console.log(`🎵 Remote track ${i}:`, {
                    kind: track.kind,
                    enabled: track.enabled,
                    muted: track.muted,
                    readyState: track.readyState,
                    label: track.label
                });
            });
        } else {
            console.warn('⚠️ No remote stream available');
        }
        
        // Check senders and receivers
        const senders = this.peerConnection.getSenders();
        const receivers = this.peerConnection.getReceivers();
        
        console.log(`📊 WebRTC Stats: ${senders.length} senders, ${receivers.length} receivers`);
        
        senders.forEach((sender, i) => {
            if (sender.track && sender.track.kind === 'audio') {
                console.log(`📤 Audio sender ${i}:`, {
                    enabled: sender.track.enabled,
                    readyState: sender.track.readyState,
                    muted: sender.track.muted
                });
            }
        });
        
        receivers.forEach((receiver, i) => {
            if (receiver.track && receiver.track.kind === 'audio') {
                console.log(`📥 Audio receiver ${i}:`, {
                    enabled: receiver.track.enabled,
                    readyState: receiver.track.readyState,
                    muted: receiver.track.muted
                });
            }
        });
    }

    /**
     * Handle connection failure
     */
    async handleConnectionFailure() {
        console.log('🔄 Attempting to restart ICE...');
        
        try {
            // Restart ICE
            await this.peerConnection.restartIce();
        } catch (error) {
            console.error('❌ ICE restart failed:', error);
            if (this.onError) {
                this.onError('connection', 'Connection failed and could not be restored');
            }
        }
    }

    /**
     * Handle ICE connection failure
     */
    async handleIceConnectionFailure() {
        console.log('🧊 ICE connection failed, attempting recovery...');
        
        // Wait a bit and try to restart ICE
        setTimeout(async () => {
            try {
                if (this.peerConnection.iceConnectionState === 'failed') {
                    await this.peerConnection.restartIce();
                }
            } catch (error) {
                console.error('❌ ICE recovery failed:', error);
            }
        }, 2000);
    }

    /**
     * Get connection quality assessment
     */
    getConnectionQuality() {
        const latency = this.stats.latency || 0;
        const packetLoss = this.stats.packetLossPercentage || 0;
        
        if (latency < 100 && packetLoss < 1) {
            return 'excellent';
        } else if (latency < 200 && packetLoss < 3) {
            return 'good';
        } else if (latency < 400 && packetLoss < 5) {
            return 'fair';
        } else if (latency < 800 && packetLoss < 10) {
            return 'poor';
        } else {
            return 'bad';
        }
    }

    /**
     * Close WebRTC connection
     */
    close() {
        console.log('🔌 Closing WebRTC connection...');
        
        // Reset initialization state
        this.isInitialized = false;
        this.initializationPromise = null;
        
        // Stop statistics collection
        this.stopStatsCollection();
        
        // Stop audio level monitoring
        this.stopAudioLevelMonitoring();
        
        // Stop local stream tracks
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => {
                track.stop();
            });
            this.localStream = null;
        }
        
        // Close peer connection
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        
        // Reset states
        this.connectionState = 'closed';
        this.iceConnectionState = 'closed';
        this.remoteStream = null;
        
        console.log('✅ WebRTC connection closed');
    }

    /**
     * Get current connection state
     */
    getConnectionState() {
        return {
            connection: this.connectionState,
            ice: this.iceConnectionState,
            signaling: this.signallingState,
            stats: this.stats,
            quality: this.getConnectionQuality()
        };
    }

    /**
     * Monitor audio levels for both local and remote streams
     */
    startAudioLevelMonitoring() {
        if (this.audioLevelMonitor) {
            return; // Already monitoring
        }
        
        console.log('📊 Starting audio level monitoring...');
        
        this.audioLevelMonitor = setInterval(() => {
            this.checkAudioLevels();
        }, 3000); // Check every 3 seconds
    }

    /**
     * Stop audio level monitoring
     */
    stopAudioLevelMonitoring() {
        if (this.audioLevelMonitor) {
            clearInterval(this.audioLevelMonitor);
            this.audioLevelMonitor = null;
            console.log('📊 Stopped audio level monitoring');
        }
    }

    /**
     * Check current audio levels and transmission stats
     */
    async checkAudioLevels() {
        try {
            console.log('📊 === AUDIO TRANSMISSION CHECK ===');
            
            // Check local audio track status
            if (this.localStream) {
                const audioTrack = this.localStream.getAudioTracks()[0];
                if (audioTrack) {
                    console.log('🎤 Local audio:', {
                        enabled: audioTrack.enabled,
                        muted: audioTrack.muted,
                        readyState: audioTrack.readyState,
                        label: audioTrack.label
                    });
                }
            }

            // Check remote audio track status
            if (this.remoteStream) {
                const audioTrack = this.remoteStream.getAudioTracks()[0];
                if (audioTrack) {
                    console.log('🔊 Remote audio:', {
                        enabled: audioTrack.enabled,
                        muted: audioTrack.muted,
                        readyState: audioTrack.readyState,
                        label: audioTrack.label
                    });
                }
            }

            // Check WebRTC transmission stats
            if (this.peerConnection && this.peerConnection.connectionState === 'connected') {
                const stats = await this.peerConnection.getStats();
                let inboundAudio = null;
                let outboundAudio = null;
                
                stats.forEach(report => {
                    if (report.type === 'inbound-rtp' && report.mediaType === 'audio') {
                        inboundAudio = {
                            packetsReceived: report.packetsReceived || 0,
                            packetsLost: report.packetsLost || 0,
                            bytesReceived: report.bytesReceived || 0,
                            jitter: report.jitter || 0
                        };
                    } else if (report.type === 'outbound-rtp' && report.mediaType === 'audio') {
                        outboundAudio = {
                            packetsSent: report.packetsSent || 0,
                            bytesSent: report.bytesSent || 0,
                            headerBytesSent: report.headerBytesSent || 0
                        };
                    }
                });
                
                if (outboundAudio && outboundAudio.packetsSent > 0) {
                    console.log('📤 Audio SENDING:', outboundAudio);
                } else {
                    console.log('📤 Audio NOT SENDING - no outbound packets');
                }
                
                if (inboundAudio && inboundAudio.packetsReceived > 0) {
                    console.log('📥 Audio RECEIVING:', inboundAudio);
                } else {
                    console.log('📥 Audio NOT RECEIVING - no inbound packets');
                }
                
                // Show overall transmission status
                const isSending = outboundAudio && outboundAudio.packetsSent > 0;
                const isReceiving = inboundAudio && inboundAudio.packetsReceived > 0;
                
                console.log(`📊 TRANSMISSION STATUS: Sending: ${isSending ? '✅' : '❌'}, Receiving: ${isReceiving ? '✅' : '❌'}`);
            }
            
            console.log('📊 === END TRANSMISSION CHECK ===');
        } catch (error) {
            console.warn('⚠️ Error checking audio levels:', error);
        }
    }
}

// Export for use in other scripts
window.WebRTCManager = WebRTCManager;
