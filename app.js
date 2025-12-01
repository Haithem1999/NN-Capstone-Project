/**
 * BraTS 2020 Brain Tumor Segmentation Analyzer
 * A comprehensive visualization tool for brain MRI analysis
 * 
 * Dataset: BraTS 2020 - Multimodal Brain Tumor Segmentation Challenge
 * Modalities: T1, T1-CE (Contrast Enhanced), T2, FLAIR
 * Labels: 0 = Background, 1 = NCR/NET (Necrotic), 2 = ED (Edema), 4 = ET (Enhancing)
 */

// ============================================================================
// Configuration Constants
// ============================================================================
const CONFIG = {
    IMAGE_SIZE: 240,
    SLICE_COUNT: 155,
    DEFAULT_SLICE: 78,
    VOXEL_SIZE: 1.0, // mm³
    
    // Color mapping for tumor regions (RGBA)
    COLORS: {
        NCR: { r: 255, g: 68, b: 68, label: 'Necrotic Core' },     // Label 1
        ED: { r: 68, g: 255, b: 68, label: 'Peritumoral Edema' },  // Label 2
        ET: { r: 255, g: 255, b: 68, label: 'Enhancing Tumor' }    // Label 4
    },
    
    // Window/Level presets
    PRESETS: {
        brain: { window: 400, level: 200 },
        contrast: { window: 300, level: 150 },
        bone: { window: 2000, level: 500 },
        tumor: { window: 250, level: 125 }
    },
    
    // Animation settings
    ANIMATION: {
        playSpeed: 150, // ms per slice
        transitionDuration: 300
    }
};

// ============================================================================
// Application State
// ============================================================================
const state = {
    // Current data
    currentPatient: null,
    currentSlice: CONFIG.DEFAULT_SLICE,
    
    // MRI data (simulated 3D volumes)
    volumes: {
        t1: null,
        t1ce: null,
        t2: null,
        flair: null,
        segmentation: null
    },
    
    // Display settings
    windowWidth: 400,
    windowLevel: 200,
    overlayOpacity: 0.6,
    
    // Region visibility
    showNCR: true,
    showED: true,
    showET: true,
    
    // View settings
    activeView: 'quad',
    activeTool: 'pan',
    syncViews: true,
    showCrosshair: true,
    isPlaying: false,
    
    // Interaction state
    isDragging: false,
    lastMousePos: { x: 0, y: 0 },
    zoom: 1,
    pan: { x: 0, y: 0 },
    
    // Metrics
    metrics: {
        dice: { wt: 0, tc: 0, et: 0 },
        volumes: { ncr: 0, ed: 0, et: 0, total: 0 },
        percentages: { ncr: 0, ed: 0, et: 0 },
        additional: { sensitivity: 0, specificity: 0, precision: 0, iou: 0, hausdorff: 0, sphericity: 0, surfaceArea: 0 }
    },
    
    // Activity log
    activityLog: [],
    
    // Animation
    playInterval: null
};

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    logActivity('System', 'Initializing BraTS 2020 Analyzer...');
    
    // Initialize canvases
    initializeCanvases();
    
    // Set up event listeners
    setupEventListeners();
    
    // Generate demo data
    generateDemoData();
    
    // Initial render
    renderAllViews();
    updateMetricsDisplay();
    
    logActivity('System', 'Ready - Demo mode active');
    showToast('info', 'Welcome', 'BraTS 2020 Analyzer initialized in demo mode');
}

// ============================================================================
// Canvas Management
// ============================================================================
const canvases = {};
const contexts = {};

function initializeCanvases() {
    const modalities = ['t1', 't1ce', 't2', 'flair'];
    
    modalities.forEach(mod => {
        canvases[mod] = document.getElementById(`canvas-${mod}`);
        contexts[mod] = canvases[mod].getContext('2d');
        
        // Enable image smoothing control
        contexts[mod].imageSmoothingEnabled = false;
    });
}

// ============================================================================
// Data Generation (Demo Mode)
// ============================================================================
function generateDemoData(patientId = 'BraTS20_Training_001') {
    showLoading('Generating synthetic MRI data...');
    
    const size = CONFIG.IMAGE_SIZE;
    const slices = CONFIG.SLICE_COUNT;
    
    // Initialize volumes
    state.volumes.t1 = new Float32Array(size * size * slices);
    state.volumes.t1ce = new Float32Array(size * size * slices);
    state.volumes.t2 = new Float32Array(size * size * slices);
    state.volumes.flair = new Float32Array(size * size * slices);
    state.volumes.segmentation = new Uint8Array(size * size * slices);
    
    // Generate brain structure and tumor
    const centerX = size / 2;
    const centerY = size / 2;
    const centerZ = slices / 2;
    
    // Tumor parameters (randomized based on patient)
    const seed = hashString(patientId);
    const random = seededRandom(seed);
    
    const tumorCenterX = centerX + (random() - 0.5) * 40;
    const tumorCenterY = centerY + (random() - 0.5) * 40;
    const tumorCenterZ = centerZ + (random() - 0.5) * 20;
    const tumorRadiusX = 15 + random() * 25;
    const tumorRadiusY = 15 + random() * 25;
    const tumorRadiusZ = 10 + random() * 15;
    
    // Generate volumes
    let progress = 0;
    const totalVoxels = size * size * slices;
    
    for (let z = 0; z < slices; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const idx = z * size * size + y * size + x;
                
                // Distance from brain center (ellipsoid)
                const brainDist = Math.sqrt(
                    Math.pow((x - centerX) / 90, 2) +
                    Math.pow((y - centerY) / 100, 2) +
                    Math.pow((z - centerZ) / 60, 2)
                );
                
                // Distance from tumor center
                const tumorDist = Math.sqrt(
                    Math.pow((x - tumorCenterX) / tumorRadiusX, 2) +
                    Math.pow((y - tumorCenterY) / tumorRadiusY, 2) +
                    Math.pow((z - tumorCenterZ) / tumorRadiusZ, 2)
                );
                
                // Brain tissue simulation
                if (brainDist < 1.0) {
                    // Base brain signal with tissue variations
                    const noise = (perlinNoise(x * 0.05, y * 0.05, z * 0.05) + 1) * 0.5;
                    const grayWhite = perlinNoise(x * 0.02, y * 0.02, z * 0.02) > 0 ? 1.2 : 0.8;
                    
                    // T1 - Gray/white matter contrast
                    state.volumes.t1[idx] = (150 + noise * 60) * grayWhite * (1 - brainDist * 0.3);
                    
                    // T1-CE - Similar to T1 but with enhancement
                    state.volumes.t1ce[idx] = state.volumes.t1[idx] * 1.1;
                    
                    // T2 - Inverted contrast
                    state.volumes.t2[idx] = (180 + noise * 50) * (2 - grayWhite) * (1 - brainDist * 0.3);
                    
                    // FLAIR - CSF suppressed
                    state.volumes.flair[idx] = (160 + noise * 55) * grayWhite * (1 - brainDist * 0.4);
                    
                    // Tumor segmentation
                    if (tumorDist < 1.0) {
                        // Enhancing tumor (core)
                        if (tumorDist < 0.3) {
                            state.volumes.segmentation[idx] = 4; // ET
                            state.volumes.t1ce[idx] = 220 + noise * 35;
                            state.volumes.t2[idx] = 200 + noise * 40;
                            state.volumes.flair[idx] = 210 + noise * 30;
                        }
                        // Necrotic core (center of some tumors)
                        else if (tumorDist < 0.5 && random() > 0.4) {
                            state.volumes.segmentation[idx] = 1; // NCR
                            state.volumes.t1[idx] = 50 + noise * 30;
                            state.volumes.t1ce[idx] = 60 + noise * 25;
                            state.volumes.t2[idx] = 180 + noise * 40;
                            state.volumes.flair[idx] = 80 + noise * 30;
                        }
                        // Enhancing tumor ring
                        else if (tumorDist < 0.6) {
                            state.volumes.segmentation[idx] = 4; // ET
                            state.volumes.t1ce[idx] = 230 + noise * 25;
                            state.volumes.t2[idx] = 190 + noise * 35;
                            state.volumes.flair[idx] = 200 + noise * 35;
                        }
                        // Peritumoral edema
                        else {
                            state.volumes.segmentation[idx] = 2; // ED
                            state.volumes.t2[idx] = 230 + noise * 25;
                            state.volumes.flair[idx] = 220 + noise * 30;
                        }
                    }
                } else {
                    // Background (outside brain)
                    state.volumes.t1[idx] = noise * 20;
                    state.volumes.t1ce[idx] = noise * 20;
                    state.volumes.t2[idx] = noise * 25;
                    state.volumes.flair[idx] = noise * 15;
                    state.volumes.segmentation[idx] = 0;
                }
            }
        }
        
        // Update progress
        progress = Math.round((z / slices) * 100);
        updateLoadingProgress(progress);
    }
    
    // Calculate metrics
    calculateMetrics();
    
    // Update state
    state.currentPatient = patientId;
    const isHGG = patientId.includes('001') || patientId.includes('002') || patientId.includes('010');
    
    // Update UI
    document.getElementById('currentPatient').textContent = patientId.split('_')[2];
    document.getElementById('tumorGrade').textContent = isHGG ? 'HGG' : 'LGG';
    document.getElementById('totalSlices').textContent = slices;
    
    hideLoading();
    logActivity('Data', `Loaded ${patientId}`);
}

// ============================================================================
// Metrics Calculation
// ============================================================================
function calculateMetrics() {
    const seg = state.volumes.segmentation;
    const size = CONFIG.IMAGE_SIZE;
    const slices = CONFIG.SLICE_COUNT;
    
    // Count voxels for each region
    let ncrCount = 0, edCount = 0, etCount = 0, totalTumor = 0;
    
    for (let i = 0; i < seg.length; i++) {
        switch (seg[i]) {
            case 1: ncrCount++; totalTumor++; break;
            case 2: edCount++; totalTumor++; break;
            case 4: etCount++; totalTumor++; break;
        }
    }
    
    // Volume calculations (voxel size is 1mm³)
    const voxelVolume = Math.pow(CONFIG.VOXEL_SIZE, 3) / 1000; // Convert to cm³
    
    state.metrics.volumes = {
        ncr: (ncrCount * voxelVolume).toFixed(2),
        ed: (edCount * voxelVolume).toFixed(2),
        et: (etCount * voxelVolume).toFixed(2),
        total: (totalTumor * voxelVolume).toFixed(2)
    };
    
    // Percentages
    if (totalTumor > 0) {
        state.metrics.percentages = {
            ncr: ((ncrCount / totalTumor) * 100).toFixed(1),
            ed: ((edCount / totalTumor) * 100).toFixed(1),
            et: ((etCount / totalTumor) * 100).toFixed(1)
        };
    }
    
    // Simulated Dice scores (would be compared with ground truth in real scenario)
    const random = seededRandom(hashString(state.currentPatient || 'demo'));
    state.metrics.dice = {
        wt: (0.88 + random() * 0.08).toFixed(3),
        tc: (0.82 + random() * 0.10).toFixed(3),
        et: (0.75 + random() * 0.12).toFixed(3)
    };
    
    // Additional metrics
    state.metrics.additional = {
        sensitivity: (0.85 + random() * 0.10).toFixed(3),
        specificity: (0.97 + random() * 0.02).toFixed(3),
        precision: (0.83 + random() * 0.10).toFixed(3),
        iou: (0.70 + random() * 0.12).toFixed(3),
        hausdorff: (2 + random() * 5).toFixed(2),
        sphericity: (0.6 + random() * 0.25).toFixed(2),
        surfaceArea: (80 + random() * 100).toFixed(1)
    };
    
    // Calculate overall dice as average
    const avgDice = (parseFloat(state.metrics.dice.wt) + 
                    parseFloat(state.metrics.dice.tc) + 
                    parseFloat(state.metrics.dice.et)) / 3;
    state.metrics.dice.overall = avgDice.toFixed(3);
}

function updateMetricsDisplay() {
    // Update Dice scores
    document.getElementById('diceWT').textContent = state.metrics.dice.wt;
    document.getElementById('diceTC').textContent = state.metrics.dice.tc;
    document.getElementById('diceET').textContent = state.metrics.dice.et;
    
    // Color code Dice scores
    ['WT', 'TC', 'ET'].forEach(region => {
        const el = document.getElementById(`dice${region}`);
        const value = parseFloat(el.textContent);
        el.className = 'dice-score ' + getDiceScoreClass(value);
    });
    
    // Update overall progress ring
    const overallDice = parseFloat(state.metrics.dice.overall);
    document.getElementById('overallDice').textContent = overallDice.toFixed(3);
    
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - overallDice);
    document.getElementById('overallProgress').style.strokeDashoffset = offset;
    
    // Update composition bars
    updateCompositionBar('ncr', state.metrics.percentages.ncr);
    updateCompositionBar('ed', state.metrics.percentages.ed);
    updateCompositionBar('et', state.metrics.percentages.et);
    
    // Update volume metrics
    document.getElementById('totalVolume').textContent = state.metrics.volumes.total;
    document.getElementById('surfaceArea').textContent = state.metrics.additional.surfaceArea;
    document.getElementById('sphericity').textContent = state.metrics.additional.sphericity;
    document.getElementById('hausdorff').textContent = state.metrics.additional.hausdorff;
    
    // Update additional metrics
    document.getElementById('sensitivity').textContent = state.metrics.additional.sensitivity;
    document.getElementById('specificity').textContent = state.metrics.additional.specificity;
    document.getElementById('precision').textContent = state.metrics.additional.precision;
    document.getElementById('iou').textContent = state.metrics.additional.iou;
}

function getDiceScoreClass(value) {
    if (value >= 0.9) return 'score-excellent';
    if (value >= 0.8) return 'score-good';
    if (value >= 0.7) return 'score-moderate';
    return 'score-low';
}

function updateCompositionBar(region, percentage) {
    const valueEl = document.getElementById(`${region}Volume`);
    if (valueEl) {
        valueEl.textContent = percentage + '%';
        const barFill = valueEl.closest('.composition-bar').querySelector('.bar-fill');
        if (barFill) {
            barFill.style.width = percentage + '%';
        }
    }
}

// ============================================================================
// Rendering
// ============================================================================
function renderAllViews() {
    if (state.activeView === 'quad') {
        renderView('t1');
        renderView('t1ce');
        renderView('t2');
        renderView('flair');
    } else {
        renderView(state.activeView);
    }
}

function renderView(modality) {
    const canvas = canvases[modality];
    const ctx = contexts[modality];
    const volume = state.volumes[modality];
    const seg = state.volumes.segmentation;
    
    if (!volume || !canvas) return;
    
    const size = CONFIG.IMAGE_SIZE;
    const slice = state.currentSlice;
    const sliceOffset = slice * size * size;
    
    // Create image data
    const imageData = ctx.createImageData(size, size);
    const data = imageData.data;
    
    // Window/Level parameters
    const ww = state.windowWidth;
    const wl = state.windowLevel;
    const wMin = wl - ww / 2;
    const wMax = wl + ww / 2;
    
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const srcIdx = sliceOffset + y * size + x;
            const dstIdx = (y * size + x) * 4;
            
            // Get intensity and apply windowing
            let intensity = volume[srcIdx];
            intensity = Math.max(0, Math.min(255, ((intensity - wMin) / ww) * 255));
            
            // Set grayscale base
            data[dstIdx] = intensity;
            data[dstIdx + 1] = intensity;
            data[dstIdx + 2] = intensity;
            data[dstIdx + 3] = 255;
            
            // Apply segmentation overlay
            const segValue = seg[srcIdx];
            if (segValue > 0 && state.overlayOpacity > 0) {
                let color = null;
                
                switch (segValue) {
                    case 1: // NCR
                        if (state.showNCR) color = CONFIG.COLORS.NCR;
                        break;
                    case 2: // ED
                        if (state.showED) color = CONFIG.COLORS.ED;
                        break;
                    case 4: // ET
                        if (state.showET) color = CONFIG.COLORS.ET;
                        break;
                }
                
                if (color) {
                    const alpha = state.overlayOpacity;
                    data[dstIdx] = intensity * (1 - alpha) + color.r * alpha;
                    data[dstIdx + 1] = intensity * (1 - alpha) + color.g * alpha;
                    data[dstIdx + 2] = intensity * (1 - alpha) + color.b * alpha;
                }
            }
        }
    }
    
    // Put image data to canvas
    ctx.putImageData(imageData, 0, 0);
}

// ============================================================================
// Event Handlers
// ============================================================================
function setupEventListeners() {
    // File upload
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Patient selection
    document.getElementById('patientSelect').addEventListener('change', (e) => {
        if (e.target.value) {
            generateDemoData(e.target.value);
            renderAllViews();
            updateMetricsDisplay();
        }
    });
    
    // Slice slider
    const sliceSlider = document.getElementById('sliceSlider');
    sliceSlider.addEventListener('input', (e) => {
        state.currentSlice = parseInt(e.target.value);
        document.getElementById('sliceValue').textContent = state.currentSlice;
        renderAllViews();
    });
    
    // Window/Level sliders
    document.getElementById('windowSlider').addEventListener('input', (e) => {
        state.windowWidth = parseInt(e.target.value);
        document.getElementById('windowValue').textContent = state.windowWidth;
        renderAllViews();
    });
    
    document.getElementById('levelSlider').addEventListener('input', (e) => {
        state.windowLevel = parseInt(e.target.value);
        document.getElementById('levelValue').textContent = state.windowLevel;
        renderAllViews();
    });
    
    // Opacity slider
    document.getElementById('opacitySlider').addEventListener('input', (e) => {
        state.overlayOpacity = parseInt(e.target.value) / 100;
        document.getElementById('opacityValue').textContent = state.overlayOpacity.toFixed(2);
        renderAllViews();
    });
    
    // Region toggles
    document.getElementById('toggleNCR').addEventListener('change', (e) => {
        state.showNCR = e.target.checked;
        renderAllViews();
        logActivity('Overlay', `NCR ${e.target.checked ? 'shown' : 'hidden'}`);
    });
    
    document.getElementById('toggleED').addEventListener('change', (e) => {
        state.showED = e.target.checked;
        renderAllViews();
        logActivity('Overlay', `ED ${e.target.checked ? 'shown' : 'hidden'}`);
    });
    
    document.getElementById('toggleET').addEventListener('change', (e) => {
        state.showET = e.target.checked;
        renderAllViews();
        logActivity('Overlay', `ET ${e.target.checked ? 'shown' : 'hidden'}`);
    });
    
    // View tabs
    document.querySelectorAll('.viewer-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.viewer-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            
            const view = e.target.dataset.view;
            state.activeView = view;
            
            const container = document.getElementById('viewerContainer');
            if (view === 'quad') {
                container.classList.remove('viewer-single');
                document.querySelectorAll('.mri-view').forEach(v => v.style.display = '');
            } else {
                container.classList.add('viewer-single');
                document.querySelectorAll('.mri-view').forEach(v => {
                    v.style.display = v.id === `view-${view}` ? '' : 'none';
                });
            }
            
            renderAllViews();
            logActivity('View', `Switched to ${view.toUpperCase()}`);
        });
    });
    
    // Toolbar buttons
    document.getElementById('prevSlice').addEventListener('click', () => navigateSlice(-1));
    document.getElementById('nextSlice').addEventListener('click', () => navigateSlice(1));
    document.getElementById('playSlices').addEventListener('click', togglePlayback);
    
    document.getElementById('toolPan').addEventListener('click', () => setTool('pan'));
    document.getElementById('toolZoom').addEventListener('click', () => setTool('zoom'));
    document.getElementById('toolMeasure').addEventListener('click', () => setTool('measure'));
    document.getElementById('toolCrosshair').addEventListener('click', () => setTool('crosshair'));
    
    document.getElementById('toggleSync').addEventListener('click', toggleSync);
    document.getElementById('toggleGrid').addEventListener('click', toggleGrid);
    document.getElementById('fullscreen').addEventListener('click', toggleFullscreen);
    
    // Action buttons
    document.getElementById('runAnalysis').addEventListener('click', runAnalysis);
    document.getElementById('resetView').addEventListener('click', resetView);
    document.getElementById('exportBtn').addEventListener('click', openExportModal);
    document.getElementById('screenshotBtn').addEventListener('click', takeScreenshot);
    
    // Export modal
    document.getElementById('closeExportModal').addEventListener('click', closeExportModal);
    document.getElementById('cancelExport').addEventListener('click', closeExportModal);
    document.getElementById('confirmExport').addEventListener('click', performExport);
    
    // Canvas mouse events
    Object.keys(canvases).forEach(mod => {
        const canvas = canvases[mod];
        canvas.addEventListener('mousemove', (e) => handleCanvasMouseMove(e, mod));
        canvas.addEventListener('mousedown', (e) => handleCanvasMouseDown(e, mod));
        canvas.addEventListener('mouseup', handleCanvasMouseUp);
        canvas.addEventListener('wheel', (e) => handleCanvasWheel(e, mod));
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
    
    // Window resize
    window.addEventListener('resize', debounce(handleResize, 250));
}

// ============================================================================
// File Handling
// ============================================================================
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    logActivity('Upload', `${files.length} file(s) selected`);
    showToast('info', 'Files Received', `Processing ${files.length} file(s)...`);
    
    // In a real implementation, this would parse NIfTI files
    // For demo, we'll show a message about the expected format
    setTimeout(() => {
        showToast('info', 'Demo Mode', 'Using synthetic data. Real NIfTI parsing requires a server-side component.');
    }, 1000);
}

// ============================================================================
// Navigation & Playback
// ============================================================================
function navigateSlice(delta) {
    const newSlice = Math.max(0, Math.min(CONFIG.SLICE_COUNT - 1, state.currentSlice + delta));
    if (newSlice !== state.currentSlice) {
        state.currentSlice = newSlice;
        document.getElementById('sliceSlider').value = newSlice;
        document.getElementById('sliceValue').textContent = newSlice;
        renderAllViews();
    }
}

function togglePlayback() {
    const btn = document.getElementById('playSlices');
    
    if (state.isPlaying) {
        clearInterval(state.playInterval);
        state.isPlaying = false;
        btn.textContent = '▶';
        btn.classList.remove('active');
        logActivity('Playback', 'Stopped');
    } else {
        state.isPlaying = true;
        btn.textContent = '⏸';
        btn.classList.add('active');
        logActivity('Playback', 'Started');
        
        state.playInterval = setInterval(() => {
            state.currentSlice = (state.currentSlice + 1) % CONFIG.SLICE_COUNT;
            document.getElementById('sliceSlider').value = state.currentSlice;
            document.getElementById('sliceValue').textContent = state.currentSlice;
            renderAllViews();
        }, CONFIG.ANIMATION.playSpeed);
    }
}

// ============================================================================
// Tool Management
// ============================================================================
function setTool(tool) {
    state.activeTool = tool;
    
    document.querySelectorAll('.toolbar-btn').forEach(btn => {
        if (btn.id === `tool${tool.charAt(0).toUpperCase() + tool.slice(1)}`) {
            btn.classList.add('active');
        } else if (btn.id.startsWith('tool')) {
            btn.classList.remove('active');
        }
    });
    
    // Update cursor
    const cursors = {
        pan: 'grab',
        zoom: 'zoom-in',
        measure: 'crosshair',
        crosshair: 'crosshair'
    };
    
    Object.values(canvases).forEach(canvas => {
        canvas.style.cursor = cursors[tool] || 'default';
    });
    
    logActivity('Tool', `Selected ${tool}`);
}

function toggleSync() {
    state.syncViews = !state.syncViews;
    const btn = document.getElementById('toggleSync');
    btn.classList.toggle('active', state.syncViews);
    logActivity('Settings', `View sync ${state.syncViews ? 'enabled' : 'disabled'}`);
}

function toggleGrid() {
    const grid = document.querySelector('.grid-overlay');
    grid.style.display = grid.style.display === 'none' ? '' : 'none';
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

// ============================================================================
// Canvas Interaction
// ============================================================================
function handleCanvasMouseMove(e, modality) {
    const canvas = canvases[modality];
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);
    
    // Update coordinates display
    const coordsEl = document.getElementById(`${modality}-coords`);
    const intensityEl = document.getElementById(`${modality}-intensity`);
    
    if (coordsEl) coordsEl.textContent = `X: ${x} Y: ${y}`;
    
    // Get intensity value
    if (state.volumes[modality] && x >= 0 && x < CONFIG.IMAGE_SIZE && y >= 0 && y < CONFIG.IMAGE_SIZE) {
        const idx = state.currentSlice * CONFIG.IMAGE_SIZE * CONFIG.IMAGE_SIZE + y * CONFIG.IMAGE_SIZE + x;
        const intensity = Math.round(state.volumes[modality][idx]);
        if (intensityEl) intensityEl.textContent = `I: ${intensity}`;
    }
    
    // Update crosshairs
    if (state.showCrosshair) {
        updateCrosshairs(modality, x, y);
    }
    
    // Handle dragging
    if (state.isDragging) {
        const dx = e.clientX - state.lastMousePos.x;
        const dy = e.clientY - state.lastMousePos.y;
        
        if (state.activeTool === 'pan') {
            state.pan.x += dx;
            state.pan.y += dy;
        } else if (state.activeTool === 'zoom') {
            const zoomDelta = -dy * 0.01;
            state.zoom = Math.max(0.5, Math.min(4, state.zoom + zoomDelta));
        }
        
        state.lastMousePos = { x: e.clientX, y: e.clientY };
    }
}

function handleCanvasMouseDown(e, modality) {
    state.isDragging = true;
    state.lastMousePos = { x: e.clientX, y: e.clientY };
    
    const canvas = canvases[modality];
    if (state.activeTool === 'pan') {
        canvas.style.cursor = 'grabbing';
    }
}

function handleCanvasMouseUp() {
    state.isDragging = false;
    
    Object.values(canvases).forEach(canvas => {
        if (state.activeTool === 'pan') {
            canvas.style.cursor = 'grab';
        }
    });
}

function handleCanvasWheel(e, modality) {
    e.preventDefault();
    
    if (e.ctrlKey) {
        // Zoom
        const zoomDelta = -e.deltaY * 0.001;
        state.zoom = Math.max(0.5, Math.min(4, state.zoom + zoomDelta));
    } else {
        // Scroll through slices
        const delta = e.deltaY > 0 ? 1 : -1;
        navigateSlice(delta);
    }
}

function updateCrosshairs(modality, x, y) {
    const modalities = state.syncViews ? ['t1', 't1ce', 't2', 'flair'] : [modality];
    
    modalities.forEach(mod => {
        const hCrosshair = document.getElementById(`crosshair-${mod}-h`);
        const vCrosshair = document.getElementById(`crosshair-${mod}-v`);
        
        if (hCrosshair && vCrosshair) {
            const canvas = canvases[mod];
            const rect = canvas.getBoundingClientRect();
            
            hCrosshair.style.top = `${(y / CONFIG.IMAGE_SIZE) * rect.height}px`;
            vCrosshair.style.left = `${(x / CONFIG.IMAGE_SIZE) * rect.width}px`;
        }
    });
}

// ============================================================================
// Keyboard Handling
// ============================================================================
function handleKeyboard(e) {
    // Ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    
    switch (e.key) {
        case 'ArrowLeft':
            navigateSlice(-1);
            break;
        case 'ArrowRight':
            navigateSlice(1);
            break;
        case ' ':
            e.preventDefault();
            togglePlayback();
            break;
        case 'r':
        case 'R':
            resetView();
            break;
        case 'f':
        case 'F':
            toggleFullscreen();
            break;
        case 'p':
        case 'P':
            setTool('pan');
            break;
        case 'z':
        case 'Z':
            setTool('zoom');
            break;
        case 'm':
        case 'M':
            setTool('measure');
            break;
        case 'c':
        case 'C':
            setTool('crosshair');
            break;
        case '1':
            document.querySelector('[data-view="t1"]').click();
            break;
        case '2':
            document.querySelector('[data-view="t1ce"]').click();
            break;
        case '3':
            document.querySelector('[data-view="t2"]').click();
            break;
        case '4':
            document.querySelector('[data-view="flair"]').click();
            break;
        case '0':
            document.querySelector('[data-view="quad"]').click();
            break;
    }
}

// ============================================================================
// Analysis & Export
// ============================================================================
function runAnalysis() {
    showLoading('Running tumor analysis...');
    logActivity('Analysis', 'Started');
    
    // Simulate analysis time
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        updateLoadingProgress(Math.min(progress, 100));
        
        if (progress >= 100) {
            clearInterval(interval);
            calculateMetrics();
            updateMetricsDisplay();
            hideLoading();
            showToast('success', 'Analysis Complete', 'Tumor metrics have been updated');
            logActivity('Analysis', 'Complete');
        }
    }, 200);
}

function resetView() {
    // Reset all settings to default
    state.currentSlice = CONFIG.DEFAULT_SLICE;
    state.windowWidth = CONFIG.PRESETS.brain.window;
    state.windowLevel = CONFIG.PRESETS.brain.level;
    state.overlayOpacity = 0.6;
    state.zoom = 1;
    state.pan = { x: 0, y: 0 };
    state.showNCR = true;
    state.showED = true;
    state.showET = true;
    
    // Update UI
    document.getElementById('sliceSlider').value = state.currentSlice;
    document.getElementById('sliceValue').textContent = state.currentSlice;
    document.getElementById('windowSlider').value = state.windowWidth;
    document.getElementById('windowValue').textContent = state.windowWidth;
    document.getElementById('levelSlider').value = state.windowLevel;
    document.getElementById('levelValue').textContent = state.windowLevel;
    document.getElementById('opacitySlider').value = state.overlayOpacity * 100;
    document.getElementById('opacityValue').textContent = state.overlayOpacity.toFixed(2);
    document.getElementById('toggleNCR').checked = true;
    document.getElementById('toggleED').checked = true;
    document.getElementById('toggleET').checked = true;
    
    renderAllViews();
    logActivity('View', 'Reset to defaults');
    showToast('info', 'View Reset', 'All settings restored to defaults');
}

function openExportModal() {
    document.getElementById('exportModal').classList.add('visible');
}

function closeExportModal() {
    document.getElementById('exportModal').classList.remove('visible');
}

function performExport() {
    const format = document.getElementById('exportFormat').value;
    
    switch (format) {
        case 'json':
            exportJSON();
            break;
        case 'csv':
            exportCSV();
            break;
        case 'png':
            takeScreenshot();
            break;
        case 'pdf':
            exportPDF();
            break;
    }
    
    closeExportModal();
}

function exportJSON() {
    const data = {
        patient: state.currentPatient,
        timestamp: new Date().toISOString(),
        slice: state.currentSlice,
        metrics: state.metrics,
        settings: {
            windowWidth: state.windowWidth,
            windowLevel: state.windowLevel,
            overlayOpacity: state.overlayOpacity
        }
    };
    
    downloadFile(JSON.stringify(data, null, 2), `brats_analysis_${state.currentPatient}.json`, 'application/json');
    logActivity('Export', 'JSON file generated');
    showToast('success', 'Export Complete', 'JSON file downloaded');
}

function exportCSV() {
    const rows = [
        ['Metric', 'Region', 'Value'],
        ['Dice Score', 'Whole Tumor', state.metrics.dice.wt],
        ['Dice Score', 'Tumor Core', state.metrics.dice.tc],
        ['Dice Score', 'Enhancing Tumor', state.metrics.dice.et],
        ['Volume (cm³)', 'Necrotic Core', state.metrics.volumes.ncr],
        ['Volume (cm³)', 'Edema', state.metrics.volumes.ed],
        ['Volume (cm³)', 'Enhancing Tumor', state.metrics.volumes.et],
        ['Volume (cm³)', 'Total', state.metrics.volumes.total],
        ['Percentage', 'Necrotic Core', state.metrics.percentages.ncr],
        ['Percentage', 'Edema', state.metrics.percentages.ed],
        ['Percentage', 'Enhancing Tumor', state.metrics.percentages.et],
        ['Additional', 'Sensitivity', state.metrics.additional.sensitivity],
        ['Additional', 'Specificity', state.metrics.additional.specificity],
        ['Additional', 'Precision', state.metrics.additional.precision],
        ['Additional', 'IoU', state.metrics.additional.iou],
        ['Additional', 'Hausdorff Distance', state.metrics.additional.hausdorff],
        ['Additional', 'Sphericity', state.metrics.additional.sphericity],
        ['Additional', 'Surface Area', state.metrics.additional.surfaceArea]
    ];
    
    const csv = rows.map(row => row.join(',')).join('\n');
    downloadFile(csv, `brats_metrics_${state.currentPatient}.csv`, 'text/csv');
    logActivity('Export', 'CSV file generated');
    showToast('success', 'Export Complete', 'CSV file downloaded');
}

function exportPDF() {
    showToast('info', 'PDF Export', 'PDF generation requires additional libraries. Using screenshot instead.');
    takeScreenshot();
}

function takeScreenshot() {
    // Create a combined canvas
    const combinedCanvas = document.createElement('canvas');
    const padding = 20;
    const canvasSize = CONFIG.IMAGE_SIZE;
    
    if (state.activeView === 'quad') {
        combinedCanvas.width = canvasSize * 2 + padding * 3;
        combinedCanvas.height = canvasSize * 2 + padding * 3;
    } else {
        combinedCanvas.width = canvasSize + padding * 2;
        combinedCanvas.height = canvasSize + padding * 2;
    }
    
    const ctx = combinedCanvas.getContext('2d');
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, combinedCanvas.width, combinedCanvas.height);
    
    if (state.activeView === 'quad') {
        ctx.drawImage(canvases.t1, padding, padding);
        ctx.drawImage(canvases.t1ce, canvasSize + padding * 2, padding);
        ctx.drawImage(canvases.t2, padding, canvasSize + padding * 2);
        ctx.drawImage(canvases.flair, canvasSize + padding * 2, canvasSize + padding * 2);
        
        // Add labels
        ctx.fillStyle = '#00d4ff';
        ctx.font = '14px JetBrains Mono';
        ctx.fillText('T1', padding + 5, padding + 20);
        ctx.fillText('T1-CE', canvasSize + padding * 2 + 5, padding + 20);
        ctx.fillText('T2', padding + 5, canvasSize + padding * 2 + 20);
        ctx.fillText('FLAIR', canvasSize + padding * 2 + 5, canvasSize + padding * 2 + 20);
    } else {
        ctx.drawImage(canvases[state.activeView], padding, padding);
    }
    
    // Download
    const link = document.createElement('a');
    link.download = `brats_${state.currentPatient}_slice${state.currentSlice}.png`;
    link.href = combinedCanvas.toDataURL('image/png');
    link.click();
    
    logActivity('Export', 'Screenshot saved');
    showToast('success', 'Screenshot Saved', 'Image downloaded to your device');
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

// ============================================================================
// UI Utilities
// ============================================================================
function showLoading(message = 'Loading...') {
    document.getElementById('loadingText').textContent = message;
    document.getElementById('loadingProgress').style.width = '0%';
    document.getElementById('loadingOverlay').classList.add('visible');
}

function updateLoadingProgress(percent) {
    document.getElementById('loadingProgress').style.width = `${percent}%`;
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('visible');
}

function showToast(type, title, message) {
    const container = document.getElementById('toastContainer');
    
    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type]}</span>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function logActivity(action, value) {
    const now = new Date();
    const time = now.toTimeString().split(' ')[0];
    
    state.activityLog.unshift({ time, action, value });
    
    // Keep only last 50 entries
    if (state.activityLog.length > 50) {
        state.activityLog.pop();
    }
    
    // Update UI
    const logContainer = document.getElementById('activityLog');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-action">${action}</span>
        <span class="log-value">${value}</span>
    `;
    
    logContainer.insertBefore(entry, logContainer.firstChild);
    
    // Keep only visible entries
    while (logContainer.children.length > 20) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

function handleResize() {
    renderAllViews();
}

// ============================================================================
// Utility Functions
// ============================================================================
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash);
}

function seededRandom(seed) {
    let state = seed;
    return function() {
        state = (state * 1103515245 + 12345) & 0x7fffffff;
        return state / 0x7fffffff;
    };
}

// Simple Perlin-like noise function
function perlinNoise(x, y, z) {
    const p = new Array(512);
    const permutation = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
        140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,
        197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,
        136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,
        122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,
        1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,
        164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,
        255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
        119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,
        19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,
        193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,
        214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,
        236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180];
    
    for (let i = 0; i < 256; i++) {
        p[256 + i] = p[i] = permutation[i];
    }
    
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const Z = Math.floor(z) & 255;
    
    x -= Math.floor(x);
    y -= Math.floor(y);
    z -= Math.floor(z);
    
    const u = fade(x);
    const v = fade(y);
    const w = fade(z);
    
    const A = p[X] + Y;
    const AA = p[A] + Z;
    const AB = p[A + 1] + Z;
    const B = p[X + 1] + Y;
    const BA = p[B] + Z;
    const BB = p[B + 1] + Z;
    
    return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                   grad(p[BA], x - 1, y, z)),
                           lerp(u, grad(p[AB], x, y - 1, z),
                                   grad(p[BB], x - 1, y - 1, z))),
                   lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
                                   grad(p[BA + 1], x - 1, y, z - 1)),
                           lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                                   grad(p[BB + 1], x - 1, y - 1, z - 1))));
}

function fade(t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(t, a, b) {
    return a + t * (b - a);
}

function grad(hash, x, y, z) {
    const h = hash & 15;
    const u = h < 8 ? x : y;
    const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}

// ============================================================================
// Console Information
// ============================================================================
console.log('%c BraTS 2020 Brain Tumor Segmentation Analyzer ', 
    'background: linear-gradient(90deg, #00d4ff, #ff00aa); color: white; font-size: 16px; padding: 10px; border-radius: 5px;');
console.log('%c Dataset Information:', 'font-weight: bold; color: #00d4ff;');
console.log('• Modalities: T1, T1-CE, T2, FLAIR');
console.log('• Image Size: 240 × 240 × 155');
console.log('• Voxel Resolution: 1mm³ isotropic');
console.log('• Tumor Labels: NCR/NET (1), ED (2), ET (4)');
console.log('%c Evaluation Metrics:', 'font-weight: bold; color: #ff00aa;');
console.log('• Dice Score (WT, TC, ET)');
console.log('• Hausdorff Distance (95th percentile)');
console.log('• Sensitivity, Specificity, Precision');
console.log('• IoU (Jaccard Index)');
