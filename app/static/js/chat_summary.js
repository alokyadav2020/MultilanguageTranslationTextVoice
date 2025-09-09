/**
 * Chat Summary JavaScript Module
 * Handles AI-powered chat summarization functionality
 */

class ChatSummaryManager {
    constructor() {
        this.currentSummaryData = null;
        this.isGenerating = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.createSummaryModal();
    }

    setupEventListeners() {
        // Direct chat summary buttons
        const generateBtn = document.getElementById('generateSummaryBtn');
        const downloadBtn = document.getElementById('downloadSummaryBtn');
        
        if (generateBtn) {
            generateBtn.addEventListener('click', (e) => this.handleGenerateSummary(e));
        }
        
        if (downloadBtn) {
            downloadBtn.addEventListener('click', (e) => this.handleDownloadSummary(e));
        }

        // Group chat summary buttons (if present)
        const groupGenerateBtn = document.getElementById('generateGroupSummaryBtn');
        const groupDownloadBtn = document.getElementById('downloadGroupSummaryBtn');
        
        if (groupGenerateBtn) {
            groupGenerateBtn.addEventListener('click', (e) => this.handleGenerateGroupSummary(e));
        }
        
        if (groupDownloadBtn) {
            groupDownloadBtn.addEventListener('click', (e) => this.handleDownloadGroupSummary(e));
        }
    }

    createSummaryModal() {
        // Check if modal already exists
        if (document.getElementById('summaryModal')) {
            return;
        }

        const modalHTML = `
            <div class="modal fade" id="summaryModal" tabindex="-1" aria-labelledby="summaryModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title" id="summaryModalLabel">
                                <i class="bi bi-lightbulb-fill"></i> 
                                AI Chat Summary
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <!-- Loading State -->
                            <div id="summaryLoading" class="text-center py-5">
                                <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                                    <span class="visually-hidden">Generating summary...</span>
                                </div>
                                <h5 class="mt-3 text-primary">AI is analyzing your conversation...</h5>
                                <p class="text-muted">This may take a few moments</p>
                                <div class="progress mt-3" style="height: 6px;">
                                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                         role="progressbar" style="width: 100%"></div>
                                </div>
                            </div>
                            
                            <!-- Summary Content -->
                            <div id="summaryContent" style="display: none;">
                                <!-- Content will be inserted here -->
                            </div>
                            
                            <!-- Error State -->
                            <div id="summaryError" style="display: none;">
                                <div class="alert alert-danger">
                                    <i class="bi bi-exclamation-triangle-fill"></i>
                                    <strong>Summary Generation Failed</strong>
                                    <p class="mb-0 mt-2" id="errorMessage">An error occurred while generating the summary.</p>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                <i class="bi bi-x-circle"></i> Close
                            </button>
                            <div class="btn-group" id="downloadButtons" style="display: none;">
                                <button type="button" class="btn btn-success" id="downloadMarkdownBtn">
                                    <i class="bi bi-download"></i> Download Markdown
                                </button>
                                <button type="button" class="btn btn-outline-success" id="downloadTextBtn">
                                    <i class="bi bi-file-text"></i> Download Text
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Setup modal download buttons
        document.getElementById('downloadMarkdownBtn')?.addEventListener('click', () => {
            this.downloadSummary('markdown');
        });

        document.getElementById('downloadTextBtn')?.addEventListener('click', () => {
            this.downloadSummary('txt');
        });
    }

    async handleGenerateSummary(event) {
        event.preventDefault();
        
        if (this.isGenerating) {
            return;
        }

        // Get user ID from page data
        const otherUserId = this.getOtherUserId();
        if (!otherUserId) {
            this.showError('Unable to identify conversation participant');
            return;
        }

        // Get user's preferred language
        const userLanguage = this.getUserLanguage();

        this.showSummaryModal();
        this.isGenerating = true;

        try {
            const response = await fetch(`/api/chat-summary/direct/${otherUserId}?language=${userLanguage}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.currentSummaryData = data;
                this.displaySummary(data);
                this.showDownloadButtons();
            } else {
                throw new Error(data.error || 'Failed to generate summary');
            }

        } catch (error) {
            console.error('Summary generation error:', error);
            this.showError(error.message);
        } finally {
            this.isGenerating = false;
        }
    }

    async handleGenerateGroupSummary(event) {
        event.preventDefault();
        
        if (this.isGenerating) {
            return;
        }

        // Get group ID from page data
        const groupId = this.getGroupId();
        if (!groupId) {
            this.showError('Unable to identify group conversation');
            return;
        }

        // Get user's preferred language
        const userLanguage = this.getUserLanguage();

        this.showSummaryModal();
        this.isGenerating = true;

        try {
            const response = await fetch(`/api/chat-summary/group/${groupId}?language=${userLanguage}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.currentSummaryData = data;
                this.displaySummary(data);
                this.showDownloadButtons();
            } else {
                throw new Error(data.error || 'Failed to generate group summary');
            }

        } catch (error) {
            console.error('Group summary generation error:', error);
            this.showError(error.message);
        } finally {
            this.isGenerating = false;
        }
    }

    showSummaryModal() {
        const modal = new bootstrap.Modal(document.getElementById('summaryModal'));
        modal.show();
        
        // Reset modal state
        document.getElementById('summaryLoading').style.display = 'block';
        document.getElementById('summaryContent').style.display = 'none';
        document.getElementById('summaryError').style.display = 'none';
        document.getElementById('downloadButtons').style.display = 'none';
    }

    displaySummary(data) {
        const summaryContent = document.getElementById('summaryContent');
        const summaryLoading = document.getElementById('summaryLoading');

        // Hide loading
        summaryLoading.style.display = 'none';

        // Create summary HTML
        const summaryHTML = `
            <div class="row">
                <div class="col-lg-8">
                    <div class="summary-text-container">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h6 class="text-primary mb-0">
                                <i class="bi bi-chat-dots"></i> 
                                ${data.group ? 'Group' : 'Direct'} Chat Summary
                            </h6>
                            <small class="text-muted">
                                <i class="bi bi-calendar"></i> 
                                ${new Date(data.generated_at).toLocaleString()}
                            </small>
                        </div>
                        
                        <div class="summary-text bg-light p-4 rounded">
                            ${this.formatSummaryText(data.summary)}
                        </div>
                        
                        <div class="mt-3">
                            <small class="text-muted">
                                <i class="bi bi-people"></i> 
                                Participants: ${data.participants.join(', ')}
                            </small>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-4">
                    <div class="summary-stats-container">
                        <h6 class="text-success mb-3">
                            <i class="bi bi-bar-chart"></i> 
                            Conversation Statistics
                        </h6>
                        ${this.formatStatistics(data.statistics)}
                    </div>
                </div>
            </div>
        `;

        summaryContent.innerHTML = summaryHTML;
        summaryContent.style.display = 'block';
    }

    formatSummaryText(summary) {
        if (!summary) {
            return '<p class="text-muted">No summary available</p>';
        }

        // Convert markdown-style formatting to HTML
        let formatted = summary
            .replace(/### (.*?)$/gm, '<h5 class="text-primary mt-4 mb-2">$1</h5>')
            .replace(/## (.*?)$/gm, '<h4 class="text-dark mt-4 mb-3">$1</h4>')
            .replace(/# (.*?)$/gm, '<h3 class="text-primary mt-4 mb-3">$1</h3>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/- (.*?)$/gm, '<li>$1</li>')
            .replace(/\n(?=<li>)/g, '')
            .replace(/(<li>.*?<\/li>)/gs, '<ul class="list-unstyled">$1</ul>')
            .replace(/\n/g, '<br>');

        return formatted;
    }

    formatStatistics(stats) {
        if (!stats) {
            return '<p class="text-muted">No statistics available</p>';
        }

        return `
            <div class="stats-grid">
                <div class="stat-card mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="stat-label">Total Messages</span>
                        <span class="stat-value badge bg-primary">${stats.total_messages || 0}</span>
                    </div>
                </div>
                
                <div class="stat-card mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="stat-label">Voice Messages</span>
                        <span class="stat-value badge bg-info">${stats.voice_messages || 0}</span>
                    </div>
                    <div class="progress mt-1" style="height: 4px;">
                        <div class="progress-bar bg-info" style="width: ${stats.voice_percentage || 0}%"></div>
                    </div>
                    <small class="text-muted">${stats.voice_percentage || 0}% of conversation</small>
                </div>
                
                <div class="stat-card mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="stat-label">Voice Duration</span>
                        <span class="stat-value badge bg-success">${stats.total_voice_duration_formatted || '0s'}</span>
                    </div>
                </div>
                
                <div class="stat-card mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="stat-label">Most Active</span>
                        <span class="stat-value text-primary">${stats.most_active_participant || 'N/A'}</span>
                    </div>
                </div>
                
                <div class="stat-card mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="stat-label">Languages</span>
                        <span class="stat-value badge bg-warning text-dark">${stats.language_count || 1}</span>
                    </div>
                    <small class="text-muted">${(stats.languages_used || []).join(', ') || 'Not specified'}</small>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Date Range</div>
                    <div class="stat-value text-muted small">${stats.date_range || 'Unknown'}</div>
                </div>
            </div>
        `;
    }

    showDownloadButtons() {
        document.getElementById('downloadButtons').style.display = 'block';
    }

    showError(message) {
        const errorDiv = document.getElementById('summaryError');
        const errorMessage = document.getElementById('errorMessage');
        const loadingDiv = document.getElementById('summaryLoading');
        const contentDiv = document.getElementById('summaryContent');

        loadingDiv.style.display = 'none';
        contentDiv.style.display = 'none';
        errorMessage.textContent = message;
        errorDiv.style.display = 'block';
    }

    async handleDownloadSummary(event) {
        event.preventDefault();
        this.downloadSummary('markdown');
    }

    async handleDownloadGroupSummary(event) {
        event.preventDefault();
        this.downloadGroupSummary('markdown');
    }

    async downloadSummary(format = 'markdown') {
        if (!this.currentSummaryData) {
            this.showError('No summary data available for download');
            return;
        }

        try {
            const otherUserId = this.getOtherUserId();
            const userLanguage = this.getUserLanguage();
            
            const url = `/api/chat-summary/direct/${otherUserId}/download?format=${format}&language=${userLanguage}`;
            window.open(url, '_blank');
            
        } catch (error) {
            console.error('Download error:', error);
            this.showError('Failed to download summary');
        }
    }

    async downloadGroupSummary(format = 'markdown') {
        if (!this.currentSummaryData) {
            this.showError('No summary data available for download');
            return;
        }

        try {
            const groupId = this.getGroupId();
            const userLanguage = this.getUserLanguage();
            
            const url = `/api/chat-summary/group/${groupId}/download?format=${format}&language=${userLanguage}`;
            window.open(url, '_blank');
            
        } catch (error) {
            console.error('Group download error:', error);
            this.showError('Failed to download summary');
        }
    }

    // Utility methods
    getOtherUserId() {
        // Try multiple methods to get the other user ID
        const urlMatch = window.location.pathname.match(/\/chat\/(\d+)/);
        if (urlMatch) {
            return urlMatch[1];
        }

        // Check for data attribute
        const chatContainer = document.querySelector('[data-other-user-id]');
        if (chatContainer) {
            return chatContainer.getAttribute('data-other-user-id');
        }

        // Check for global variable
        if (window.remoteUserId) {
            return window.remoteUserId;
        }

        return null;
    }

    getGroupId() {
        // Try multiple methods to get the group ID
        const urlMatch = window.location.pathname.match(/\/groups\/(\d+)/);
        if (urlMatch) {
            return urlMatch[1];
        }

        // Check for data attribute
        const groupContainer = document.querySelector('[data-group-id]');
        if (groupContainer) {
            return groupContainer.getAttribute('data-group-id');
        }

        // Check for global variable
        if (window.currentGroupId) {
            return window.currentGroupId;
        }

        return null;
    }

    getUserLanguage() {
        // Get user's preferred language for summary
        const langSelect = document.getElementById('summaryLanguageSelect');
        if (langSelect) {
            return langSelect.value;
        }

        // Check user's general language preference
        const userLangElement = document.querySelector('[data-user-language]');
        if (userLangElement) {
            return userLangElement.getAttribute('data-user-language');
        }

        // Check global variable
        if (window.userLanguage) {
            return window.userLanguage;
        }

        // Default to English
        return 'en';
    }

    // Static method to add summary buttons to existing chat interfaces
    static addSummaryButtons() {
        // Add to direct chat header
        const chatHeader = document.querySelector('.chat-header, .message-header');
        if (chatHeader && !document.getElementById('generateSummaryBtn')) {
            const buttonsHTML = `
                <div class="summary-buttons d-flex gap-2 ms-2">
                    <button id="generateSummaryBtn" class="btn btn-outline-primary btn-sm" title="Generate AI Summary">
                        <i class="bi bi-lightbulb"></i> <span class="d-none d-md-inline">Summary</span>
                    </button>
                    <button id="downloadSummaryBtn" class="btn btn-outline-success btn-sm" style="display: none;" title="Download Summary">
                        <i class="bi bi-download"></i> <span class="d-none d-md-inline">Download</span>
                    </button>
                </div>
            `;
            chatHeader.insertAdjacentHTML('beforeend', buttonsHTML);
        }

        // Add to group chat interface
        const groupHeader = document.querySelector('.group-header, .group-controls');
        if (groupHeader && !document.getElementById('generateGroupSummaryBtn')) {
            const groupButtonsHTML = `
                <div class="group-summary-buttons d-flex gap-2 ms-2">
                    <button id="generateGroupSummaryBtn" class="btn btn-outline-primary btn-sm" title="Generate Group Summary">
                        <i class="bi bi-lightbulb"></i> <span class="d-none d-md-inline">Group Summary</span>
                    </button>
                    <button id="downloadGroupSummaryBtn" class="btn btn-outline-success btn-sm" style="display: none;" title="Download Group Summary">
                        <i class="bi bi-download"></i> <span class="d-none d-md-inline">Download</span>
                    </button>
                </div>
            `;
            groupHeader.insertAdjacentHTML('beforeend', groupButtonsHTML);
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add summary buttons to existing interfaces
    ChatSummaryManager.addSummaryButtons();
    
    // Initialize the summary manager
    window.chatSummaryManager = new ChatSummaryManager();
    
    console.log('✅ Chat Summary Manager initialized');
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatSummaryManager;
}
