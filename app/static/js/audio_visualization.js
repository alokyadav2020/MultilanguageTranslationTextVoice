// filepath: app/static/js/audio_visualization.js

/**
 * Audio Visualization for Voice Calls
 * Provides real-time audio waveform visualization and volume level indicators
 */

class AudioVisualization {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.localStream = null;
        this.remoteStream = null;
        this.localAnalyser = null;
        this.remoteAnalyser = null;
        this.animationId = null;
        
        // Audio context
        this.audioContext = null;
        this.localSource = null;
        this.remoteSource = null;
        
        // Visualization settings
        this.fftSize = 256;
        this.bufferLength = this.fftSize / 2;
        this.dataArray = new Uint8Array(this.bufferLength);
        this.remoteDataArray = new Uint8Array(this.bufferLength);
        
        // Canvas settings
        this.width = 0;
        this.height = 0;
        this.centerY = 0;
        
        // Colors
        this.colors = {
            background: 'rgba(255, 255, 255, 0.1)',
            localWave: '#4ade80',
            remoteWave: '#60a5fa',
            gridLines: 'rgba(255, 255, 255, 0.2)'
        };
        
        // Volume level elements
        this.myVolumeElement = null;
        this.theirVolumeElement = null;
        
        // Volume detection
        this.localVolumeLevel = 0;
        this.remoteVolumeLevel = 0;
        this.volumeThreshold = 30; // Minimum volume to register as speaking
        
        this.init();
    }

    /**
     * Initialize audio visualization
     */
    init() {
        this.canvas = document.getElementById('waveform-canvas');
        this.myVolumeElement = document.getElementById('my-volume');
        this.theirVolumeElement = document.getElementById('their-volume');
        
        if (!this.canvas) {
            console.warn('Waveform canvas not found');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();
        this.startVisualization();
        
        console.log('🎨 Audio visualization initialized');
    }

    /**
     * Setup canvas dimensions and styling
     */
    setupCanvas() {
        const resizeCanvas = () => {
            const rect = this.canvas.getBoundingClientRect();
            this.canvas.width = rect.width * window.devicePixelRatio;
            this.canvas.height = rect.height * window.devicePixelRatio;
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            this.width = rect.width;
            this.height = rect.height;
            this.centerY = this.height / 2;
        };
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
    }

    /**
     * Set local audio stream (user's microphone)
     */
    setLocalStream(stream) {
        try {
            this.localStream = stream;
            this.setupLocalAnalyser();
            console.log('🎤 Local stream set for visualization');
        } catch (error) {
            console.error('❌ Failed to set local stream:', error);
        }
    }

    /**
     * Set remote audio stream (other party's audio)
     */
    setRemoteStream(stream) {
        try {
            this.remoteStream = stream;
            this.setupRemoteAnalyser();
            console.log('🔊 Remote stream set for visualization');
        } catch (error) {
            console.error('❌ Failed to set remote stream:', error);
        }
    }

    /**
     * Setup audio analyser for local stream
     */
    setupLocalAnalyser() {
        if (!this.localStream) return;
        
        try {
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            this.localSource = this.audioContext.createMediaStreamSource(this.localStream);
            this.localAnalyser = this.audioContext.createAnalyser();
            
            this.localAnalyser.fftSize = this.fftSize;
            this.localAnalyser.smoothingTimeConstant = 0.85;
            this.localAnalyser.minDecibels = -90;
            this.localAnalyser.maxDecibels = -10;
            
            this.localSource.connect(this.localAnalyser);
            
        } catch (error) {
            console.error('❌ Failed to setup local analyser:', error);
        }
    }

    /**
     * Setup audio analyser for remote stream
     */
    setupRemoteAnalyser() {
        if (!this.remoteStream) return;
        
        try {
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            this.remoteSource = this.audioContext.createMediaStreamSource(this.remoteStream);
            this.remoteAnalyser = this.audioContext.createAnalyser();
            
            this.remoteAnalyser.fftSize = this.fftSize;
            this.remoteAnalyser.smoothingTimeConstant = 0.85;
            this.remoteAnalyser.minDecibels = -90;
            this.remoteAnalyser.maxDecibels = -10;
            
            this.remoteSource.connect(this.remoteAnalyser);
            
        } catch (error) {
            console.error('❌ Failed to setup remote analyser:', error);
        }
    }

    /**
     * Start visualization animation loop
     */
    startVisualization() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            this.draw();
            this.updateVolumeIndicators();
        };
        
        animate();
    }

    /**
     * Stop visualization
     */
    stopVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    /**
     * Main drawing function
     */
    draw() {
        if (!this.ctx) return;
        
        // Clear canvas
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Draw grid lines
        this.drawGrid();
        
        // Draw waveforms
        this.drawLocalWaveform();
        this.drawRemoteWaveform();
        
        // Draw center line
        this.drawCenterLine();
        
        // Draw labels
        this.drawLabels();
    }

    /**
     * Draw grid background
     */
    drawGrid() {
        this.ctx.strokeStyle = this.colors.gridLines;
        this.ctx.lineWidth = 0.5;
        this.ctx.globalAlpha = 0.3;
        
        // Horizontal lines
        for (let i = 0; i <= 4; i++) {
            const y = (this.height / 4) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
        
        // Vertical lines
        for (let i = 0; i <= 8; i++) {
            const x = (this.width / 8) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }

    /**
     * Draw local audio waveform (top half)
     */
    drawLocalWaveform() {
        if (!this.localAnalyser) {
            this.drawIdleWave(this.colors.localWave, 0, this.centerY);
            return;
        }
        
        this.localAnalyser.getByteFrequencyData(this.dataArray);
        
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = this.colors.localWave;
        this.ctx.fillStyle = this.colors.localWave + '40'; // With alpha
        
        this.ctx.beginPath();
        
        const sliceWidth = this.width / this.bufferLength;
        let x = 0;
        
        // Calculate average volume for this frame
        let sum = 0;
        for (let i = 0; i < this.bufferLength; i++) {
            sum += this.dataArray[i];
        }
        this.localVolumeLevel = sum / this.bufferLength;
        
        // Draw the waveform
        for (let i = 0; i < this.bufferLength; i++) {
            const amplitude = this.dataArray[i] / 255.0;
            const y = this.centerY - (amplitude * this.centerY * 0.8);
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        // Complete the filled area
        this.ctx.lineTo(this.width, this.centerY);
        this.ctx.lineTo(0, this.centerY);
        this.ctx.closePath();
        
        this.ctx.fill();
        this.ctx.stroke();
    }

    /**
     * Draw remote audio waveform (bottom half)
     */
    drawRemoteWaveform() {
        if (!this.remoteAnalyser) {
            this.drawIdleWave(this.colors.remoteWave, this.centerY, this.height);
            return;
        }
        
        this.remoteAnalyser.getByteFrequencyData(this.remoteDataArray);
        
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = this.colors.remoteWave;
        this.ctx.fillStyle = this.colors.remoteWave + '40'; // With alpha
        
        this.ctx.beginPath();
        
        const sliceWidth = this.width / this.bufferLength;
        let x = 0;
        
        // Calculate average volume for this frame
        let sum = 0;
        for (let i = 0; i < this.bufferLength; i++) {
            sum += this.remoteDataArray[i];
        }
        this.remoteVolumeLevel = sum / this.bufferLength;
        
        // Draw the waveform (inverted for bottom half)
        for (let i = 0; i < this.bufferLength; i++) {
            const amplitude = this.remoteDataArray[i] / 255.0;
            const y = this.centerY + (amplitude * this.centerY * 0.8);
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        // Complete the filled area
        this.ctx.lineTo(this.width, this.centerY);
        this.ctx.lineTo(0, this.centerY);
        this.ctx.closePath();
        
        this.ctx.fill();
        this.ctx.stroke();
    }

    /**
     * Draw idle wave animation when no audio data
     */
    drawIdleWave(color, startY, endY) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.5;
        
        const amplitude = 10;
        const frequency = 0.02;
        const time = Date.now() * 0.005;
        
        this.ctx.beginPath();
        
        for (let x = 0; x <= this.width; x += 2) {
            const y = startY + (endY - startY) / 2 + 
                Math.sin(x * frequency + time) * amplitude * 
                Math.sin(x * 0.01 + time * 0.5);
            
            if (x === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        this.ctx.globalAlpha = 1;
    }

    /**
     * Draw center divider line
     */
    drawCenterLine() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.centerY);
        this.ctx.lineTo(this.width, this.centerY);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }

    /**
     * Draw labels for local and remote audio
     */
    drawLabels() {
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'right';
        
        // Local audio label
        this.ctx.fillText('Your Audio', this.width - 10, 20);
        
        // Remote audio label
        this.ctx.fillText('Their Audio', this.width - 10, this.height - 10);
    }

    /**
     * Update volume level indicators
     */
    updateVolumeIndicators() {
        // Update local volume indicator
        if (this.myVolumeElement) {
            const level = this.getVolumeLevel(this.localVolumeLevel);
            this.updateVolumeElement(this.myVolumeElement, level);
        }
        
        // Update remote volume indicator
        if (this.theirVolumeElement) {
            const level = this.getVolumeLevel(this.remoteVolumeLevel);
            this.updateVolumeElement(this.theirVolumeElement, level);
        }
    }

    /**
     * Convert raw volume to discrete level (1-5)
     */
    getVolumeLevel(rawVolume) {
        if (rawVolume < this.volumeThreshold) return 0;
        if (rawVolume < 50) return 1;
        if (rawVolume < 100) return 2;
        if (rawVolume < 150) return 3;
        if (rawVolume < 200) return 4;
        return 5;
    }

    /**
     * Update volume element visual state
     */
    updateVolumeElement(element, level) {
        // Remove existing level classes
        element.classList.remove('level-1', 'level-2', 'level-3', 'level-4', 'level-5');
        
        // Add current level class
        if (level > 0) {
            element.classList.add(`level-${level}`);
        }
    }

    /**
     * Set volume threshold for voice activity detection
     */
    setVolumeThreshold(threshold) {
        this.volumeThreshold = threshold;
    }

    /**
     * Get current volume levels
     */
    getCurrentVolumeLevels() {
        return {
            local: this.localVolumeLevel,
            remote: this.remoteVolumeLevel,
            localLevel: this.getVolumeLevel(this.localVolumeLevel),
            remoteLevel: this.getVolumeLevel(this.remoteVolumeLevel)
        };
    }

    /**
     * Check if someone is speaking
     */
    isSpeaking() {
        return {
            local: this.localVolumeLevel > this.volumeThreshold,
            remote: this.remoteVolumeLevel > this.volumeThreshold
        };
    }

    /**
     * Update visualization colors
     */
    setColors(newColors) {
        this.colors = { ...this.colors, ...newColors };
    }

    /**
     * Enable/disable visualization
     */
    setEnabled(enabled) {
        if (enabled && !this.animationId) {
            this.startVisualization();
        } else if (!enabled && this.animationId) {
            this.stopVisualization();
        }
    }

    /**
     * Clean up resources
     */
    cleanup() {
        this.stopVisualization();
        
        if (this.localSource) {
            this.localSource.disconnect();
            this.localSource = null;
        }
        
        if (this.remoteSource) {
            this.remoteSource.disconnect();
            this.remoteSource = null;
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.localAnalyser = null;
        this.remoteAnalyser = null;
        this.localStream = null;
        this.remoteStream = null;
    }
}

// Global instance
window.AudioVisualization = new AudioVisualization();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.AudioVisualization) {
        window.AudioVisualization.cleanup();
    }
});
