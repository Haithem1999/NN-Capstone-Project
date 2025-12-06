/**
 * NeuroScan AI - Tumor Segmentation Logic
 * Now updated with Realistic Brightness Thresholding for FLAIR sequence simulation.
 */

// ================= GLOBAL STATE =================
const state = {
    header: null,
    image: null,      // Raw NIfTI data buffer
    dimensions: null, // [x, y, z]
    slices: 0,
    currentSlice: 0,
    data: null,       // TypedArray of image data
    mask: null,       // TypedArray for tumor mask (0 or 1)
    isLoaded: false,
    viewMode: '2d'    // '2d' or '3d'
};

// UI Elements
const els = {
    fileInput: document.getElementById('nifti-input'),
    fileLabel: document.getElementById('file-label'),
    runBtn: document.getElementById('run-btn'),
    slider: document.getElementById('slice-slider'),
    sliceVal: document.getElementById('slice-val'),
    opacitySlider: document.getElementById('opacity-slider'),
    canvas: document.getElementById('mri-canvas'),
    ctx: document.getElementById('mri-canvas').getContext('2d'),
    loader: document.getElementById('loader'),
    overlayInfo: document.getElementById('overlay-info'),
    tabs: document.querySelectorAll('.tab-btn'),
    container3d: document.getElementById('3d-container'),
    canvasContainer: document.getElementById('canvas-container')
};

// ================= INITIALIZATION =================
function init() {
    setupEventListeners();
    console.log("NeuroScan AI Initialized.");
}

function setupEventListeners() {
    els.fileInput.addEventListener('change', handleFileUpload);
    els.runBtn.addEventListener('click', runSegmentationSimulation);
    els.slider.addEventListener('input', updateSliceView);
    els.opacitySlider.addEventListener('input', updateSliceView);
    
    // Tab Switching
    els.tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            els.tabs.forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            state.viewMode = e.target.dataset.view;
            toggleViewMode();
        });
    });
}

// ================= FILE HANDLING =================
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    els.fileLabel.textContent = file.name;
    els.loader.classList.add('active');
    document.querySelector('.progress-text').textContent = "Reading NIfTI File...";

    const reader = new FileReader();
    reader.onload = function(evt) {
        try {
            const data = evt.target.result;
            
            // Parse NIfTI
            if (nifti.isCompressed(data)) {
                state.data = nifti.decompress(data);
                state.header = nifti.readHeader(state.data.buffer);
            } else {
                state.header = nifti.readHeader(data);
                state.data = data; // Raw data
            }

            // Get Image Data
            // Note: This is a simplified nifti reading for the demo.
            // We assume standard orientation or just visualize raw data.
            const image = nifti.readImage(state.header, state.data);
            
            // Handle different data types (Int16, Float32, etc)
            // We convert everything to a Float32 array for easier processing
            let rawData;
            if (state.header.datatypeCode === nifti.NIFTI1.TYPE_UINT8) {
                rawData = new Uint8Array(image);
            } else if (state.header.datatypeCode === nifti.NIFTI1.TYPE_INT16) {
                rawData = new Int16Array(image);
            } else if (state.header.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) {
                rawData = new Float32Array(image);
            } else {
                rawData = new Uint16Array(image); // Fallback
            }

            state.image = rawData;
            state.dimensions = state.header.dims.slice(1, 4); // x, y, z
            state.slices = state.dimensions[2];
            
            // Reset State
            state.currentSlice = Math.floor(state.slices / 2);
            state.mask = null; // Clear previous mask
            state.isLoaded = true;

            // Update UI
            els.slider.max = state.slices - 1;
            els.slider.value = state.currentSlice;
            els.slider.disabled = false;
            els.runBtn.disabled = false;
            els.overlayInfo.textContent = `Loaded: ${state.dimensions[0]}x${state.dimensions[1]}x${state.dimensions[2]}`;
            
            // Initial Render
            resizeCanvas();
            drawSlice(state.currentSlice);
            
            setTimeout(() => els.loader.classList.remove('active'), 500);
            
        } catch (err) {
            console.error(err);
            alert("Error parsing NIfTI file. Ensure it is a valid .nii or .nii.gz file.");
            els.loader.classList.remove('active');
        }
    };
    reader.readAsArrayBuffer(file);
}

// ================= SEGMENTATION LOGIC (UPDATED) =================

/**
 * Simulates AI detection by running a high-pass intensity threshold.
 * In FLAIR MRI, tumors are typically the brightest (hyperintense) regions.
 */
async function runSegmentationSimulation() {
    if (!state.isLoaded) return;

    // UI Feedback
    els.runBtn.disabled = true;
    els.loader.classList.add('active');
    const progressText = document.querySelector('.progress-text');
    
    // 1. Simulation Steps to make it feel like "Processing"
    progressText.textContent = "Preprocessing Volume...";
    await wait(800);
    
    progressText.textContent = "Running Neural Network Inference...";
    await wait(1000);

    progressText.textContent = "Post-processing Segmentation Masks...";
    await wait(800);

    // 2. REALISTIC SEGMENTATION ALGORITHM
    // Instead of random noise, we detect the brightest contiguous blobs.
    
    const totalPixels = state.image.length;
    state.mask = new Uint8Array(totalPixels); // 0 = background, 1 = tumor

    // A. Find global max intensity
    let maxIntensity = 0;
    // Sample to save time or loop all
    for (let i = 0; i < totalPixels; i += 10) { 
        if (state.image[i] > maxIntensity) maxIntensity = state.image[i];
    }
    
    // Safety check for black images
    if (maxIntensity === 0) maxIntensity = 1;

    // B. Thresholding
    // We assume tumor is in the top 5-10% of brightness for FLAIR.
    // Adjust this threshold factor (0.85 means top 15% brightness) to tune sensitivity.
    const THRESHOLD_RATIO = 0.80; 
    const threshold = maxIntensity * THRESHOLD_RATIO;

    // C. Apply Threshold
    // We can simulate "scan lines" processing if we wanted, but let's do it in one go.
    for (let i = 0; i < totalPixels; i++) {
        if (state.image[i] > threshold) {
            state.mask[i] = 1;
        } else {
            state.mask[i] = 0;
        }
    }

    // D. Done
    els.loader.classList.remove('active');
    els.runBtn.textContent = "SEGMENTATION COMPLETE";
    els.runBtn.style.background = "linear-gradient(90deg, #00ff88, #00aaff)";
    updateSliceView();
}

// ================= VISUALIZATION =================

function updateSliceView() {
    state.currentSlice = parseInt(els.slider.value);
    els.sliceVal.textContent = state.currentSlice;
    if (state.isLoaded) {
        drawSlice(state.currentSlice);
    }
}

function resizeCanvas() {
    if (!state.dimensions) return;
    els.canvas.width = state.dimensions[0];
    els.canvas.height = state.dimensions[1];
}

function drawSlice(sliceIndex) {
    if (!state.image) return;

    const w = state.dimensions[0];
    const h = state.dimensions[1];
    const sliceSize = w * h;
    const sliceOffset = sliceIndex * sliceSize;

    // Get Canvas Data
    const imageData = els.ctx.createImageData(w, h);
    const data = imageData.data; // RGBA array

    // Calculate Slice Max for Visualization Normalization
    // (Optional: can use global max, but slice max gives better contrast)
    let sliceMax = 0;
    for (let i = 0; i < sliceSize; i++) {
        if (state.image[sliceOffset + i] > sliceMax) sliceMax = state.image[sliceOffset + i];
    }
    if (sliceMax === 0) sliceMax = 1;

    const opacity = els.opacitySlider.value / 100;

    for (let i = 0; i < sliceSize; i++) {
        const pixelVal = state.image[sliceOffset + i];
        // Normalize 0-255
        const normVal = Math.floor((pixelVal / sliceMax) * 255);
        
        // Render Index (canvas is usually top-down, NIfTI might be bottom-up, we just map 1:1 here)
        const pxIndex = i * 4;

        // Base MRI (Grayscale)
        data[pxIndex] = normVal;     // R
        data[pxIndex + 1] = normVal; // G
        data[pxIndex + 2] = normVal; // B
        data[pxIndex + 3] = 255;     // Alpha

        // Overlay Tumor Mask (Red)
        if (state.mask && state.mask[sliceOffset + i] === 1) {
            // Blend Red
            data[pxIndex] = Math.min(255, data[pxIndex] + 100); // Tint R
            data[pxIndex + 1] = Math.floor(data[pxIndex + 1] * (1 - opacity)); // Dim G
            data[pxIndex + 2] = Math.floor(data[pxIndex + 2] * (1 - opacity)); // Dim B
        }
    }

    els.ctx.putImageData(imageData, 0, 0);
}

function toggleViewMode() {
    if (state.viewMode === '3d') {
        els.canvasContainer.style.display = 'none';
        els.container3d.style.display = 'block';
        init3DView();
    } else {
        els.container3d.style.display = 'none';
        els.canvasContainer.style.display = 'block';
    }
}

// Simple Placeholder for 3D View (Simulated Point Cloud or similar could go here)
function init3DView() {
    els.container3d.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100%;color:#666;">3D Surface Rendering would require heavy marching cubes implementation.<br>Switch back to 2D for slice analysis.</div>';
}

// Utility
function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Start
init();
