// filepath: app/static/js/webrtc.js

/**
 * WebRTC Voice Call Implementation
 * Handles peer-to-peer audio communication for voice calls
 */

class WebRTCManager {
    constructor() {
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.websocket = null;
        this.isInitiator = false;
        this.callId = null;
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
        try {
            this.callId = callId;
            this.websocket = websocket;
            this.isInitiator = isInitiator;
            
            console.log('🔗 Initializing WebRTC for call:', callId);
            
            // Create peer connection
            this.peerConnection = new RTCPeerConnection(this.config);
            this.setupPeerConnectionHandlers();
            
            // Get user media
            await this.getUserMedia();
            
            // If initiator, create and send offer
            if (this.isInitiator) {
                await this.createOffer();
            }
            
            // Start collecting statistics
            this.startStatsCollection();
            
            console.log('✅ WebRTC initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ WebRTC initialization failed:', error);
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
            console.log('🎵 Received remote stream');
            this.remoteStream = event.streams[0];
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
            
            this.localStream = await navigator.mediaDevices.getUserMedia(this.audioConstraints);
            
            // Add local stream to peer connection
            this.localStream.getTracks().forEach(track => {
                this.peerConnection.addTrack(track, this.localStream);
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
            console.log('📞 Handling incoming offer...');
            
            const offer = new RTCSessionDescription({
                type: 'offer',
                sdp: sdp
            });
            
            await this.peerConnection.setRemoteDescription(offer);
            
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
        
        // Stop statistics collection
        this.stopStatsCollection();
        
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
}

// Export for use in other scripts
window.WebRTCManager = WebRTCManager;
