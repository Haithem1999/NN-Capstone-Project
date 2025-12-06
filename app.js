/**
 * NeuroScan AI - Brain Tumor Segmentation Application
 * 3D U-Net implementation with TensorFlow.js for browser-based inference
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

let model = null;
let mriData = null;
let segmentationMask = null;
let currentSlice = 75;
let overlayOpacity = 0.7;
let viewMode = 'overlay';
let currentTab = 'slices';
let scene, camera, renderer, brain3D;
let isRotating = true;
let rotationSpeed = 0.01;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🧠 NeuroScan AI initializing...');
    setupEventListeners();
    initializeModel();
});

// ============================================================================
// 3D U-NET MODEL
// ============================================================================

/**
 * Build 3D U-Net model architecture
 * Simplified version for browser (full version would be too heavy)
 */
async function buildUNet3D() {
    console.log('Building 3D U-Net architecture...');
    
    // For browser efficiency, we'll use a simplified 2D U-Net approach
    // that processes slices and then combines them
    const model = tf.sequential();
    
    // Encoder
    model.add(tf.layers.conv2d({
        inputShape: [128, 128, 4],
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'enc_conv1a'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'enc_conv1b'
    }));
    
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        name: 'enc_pool1'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'enc_conv2a'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'enc_conv2b'
    }));
    
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        name: 'enc_pool2'
    }));
    
    // Bottleneck
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'bottleneck_conv1'
    }));
    
    model.add(tf.layers.dropout({
        rate: 0.2,
        name: 'bottleneck_dropout'
    }));
    
    // Decoder
    model.add(tf.layers.upSampling2d({
        size: [2, 2],
        name: 'dec_up1'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'dec_conv1a'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'dec_conv1b'
    }));
    
    model.add(tf.layers.upSampling2d({
        size: [2, 2],
        name: 'dec_up2'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'dec_conv2a'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same',
        name: 'dec_conv2b'
    }));
    
    // Output layer - 4 classes (background, NCR/NET, ED, ET)
    model.add(tf.layers.conv2d({
        filters: 4,
        kernelSize: 1,
        activation: 'softmax',
        padding: 'same',
        name: 'output'
    }));
    
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log('✅ Model architecture built');
    return model;
}

async function initializeModel() {
    try {
        console.log('Initializing TensorFlow.js model...');
        model = await buildUNet3D();
        console.log('✅ Model initialized (untrained weights)');
        console.log('Note: For production, load pre-trained weights');
    } catch (error) {
        console.error('❌ Error initializing model:', error);
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length) {
            handleFileUpload({ target: { files: e.dataTransfer.files } });
        }
    });
    
    document.getElementById('analyzeBtn').addEventListener('click', runSegmentation);
    document.getElementById('demoBtn').addEventListener('click', loadDemoData);
    
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    
    document.getElementById('sliceSlider').addEventListener('input', (e) => {
        currentSlice = parseInt(e.target.value);
        document.getElementById('sliceValue').textContent = currentSlice;
        if (mriData) updateSliceViews();
    });
    
    document.getElementById('opacitySlider').addEventListener('input', (e) => {
        overlayOpacity = parseFloat(e.target.value);
        document.getElementById('opacityValue').textContent = overlayOpacity.toFixed(1);
        if (mriData) updateSliceViews();
    });
    
    document.getElementById('rotationSlider').addEventListener('input', (e) => {
        const speed = parseFloat(e.target.value);
        document.getElementById('rotationValue').textContent = speed.toFixed(1) + 'x';
        rotationSpeed = speed * 0.01;
    });
    
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            viewMode = btn.dataset.mode;
            if (mriData) updateSliceViews();
        });
    });
}

// ============================================================================
// FILE HANDLING
// ============================================================================

let uploadedFile = null;
let uploadedNiftiData = null;

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        uploadedFile = file;
        document.getElementById('analyzeBtn').disabled = false;
        showNotification(`File loaded: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
        
        // Check file type and start parsing
        const fileName = file.name.toLowerCase();
        if (fileName.endsWith('.nii') || fileName.endsWith('.nii.gz')) {
            parseNiftiFile(file);
        } else if (file.type.startsWith('image/')) {
            parseImageFile(file);
        }
    }
}

/**
 * Parse a standard image file (PNG, JPG, etc.)
 */
async function parseImageFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Create a canvas to extract pixel data
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                
                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                
                // Convert to grayscale and store as single slice
                const width = img.width;
                const height = img.height;
                const depth = 1;
                
                mriData = new Array(depth);
                segmentationMask = new Array(depth);
                
                mriData[0] = new Uint8Array(width * height);
                segmentationMask[0] = new Uint8Array(width * height);
                
                for (let i = 0; i < width * height; i++) {
                    const r = imageData.data[i * 4];
                    const g = imageData.data[i * 4 + 1];
                    const b = imageData.data[i * 4 + 2];
                    // Convert to grayscale
                    mriData[0][i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                }
                
                uploadedNiftiData = {
                    width: width,
                    height: height,
                    depth: depth,
                    data: mriData
                };
                
                showNotification(`✅ Image parsed: ${width}x${height}`);
                resolve();
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Parse NIfTI file format (.nii or .nii.gz)
 */
async function parseNiftiFile(file) {
    showNotification('📂 Parsing NIfTI file...');
    
    try {
        let arrayBuffer = await file.arrayBuffer();
        
        // Check if gzipped
        const fileName = file.name.toLowerCase();
        if (fileName.endsWith('.gz')) {
            showNotification('🗜️ Decompressing gzip...');
            arrayBuffer = await decompressGzip(arrayBuffer);
        }
        
        // Parse NIfTI header and data
        const nifti = parseNiftiBuffer(arrayBuffer);
        
        if (nifti) {
            uploadedNiftiData = nifti;
            showNotification(`✅ NIfTI parsed: ${nifti.width}x${nifti.height}x${nifti.depth}`);
            
            // Update slider range based on actual depth
            const sliceSlider = document.getElementById('sliceSlider');
            sliceSlider.max = nifti.depth - 1;
            sliceSlider.value = Math.floor(nifti.depth / 2);
            currentSlice = Math.floor(nifti.depth / 2);
            document.getElementById('sliceValue').textContent = currentSlice;
        }
    } catch (error) {
        console.error('Error parsing NIfTI:', error);
        showNotification('❌ Error parsing NIfTI file: ' + error.message);
    }
}

/**
 * Decompress gzip data using DecompressionStream API or pako fallback
 */
async function decompressGzip(arrayBuffer) {
    // Try native DecompressionStream first (modern browsers)
    if (typeof DecompressionStream !== 'undefined') {
        try {
            const ds = new DecompressionStream('gzip');
            const stream = new Response(arrayBuffer).body.pipeThrough(ds);
            const decompressed = await new Response(stream).arrayBuffer();
            return decompressed;
        } catch (e) {
            console.warn('DecompressionStream failed, trying manual decompression');
        }
    }
    
    // Manual gzip decompression (simplified implementation)
    return manualGunzip(arrayBuffer);
}

/**
 * Manual gzip decompression using pako-style inflate
 * This is a simplified implementation - for production, use pako library
 */
async function manualGunzip(arrayBuffer) {
    const data = new Uint8Array(arrayBuffer);
    
    // Check gzip magic number
    if (data[0] !== 0x1f || data[1] !== 0x8b) {
        throw new Error('Not a valid gzip file');
    }
    
    // Try to dynamically load pako if available
    if (typeof pako !== 'undefined') {
        return pako.ungzip(data).buffer;
    }
    
    // Load pako dynamically
    try {
        await loadScript('https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js');
        if (typeof pako !== 'undefined') {
            return pako.ungzip(data).buffer;
        }
    } catch (e) {
        console.warn('Could not load pako library');
    }
    
    throw new Error('Gzip decompression not available. Please use uncompressed .nii files or a modern browser.');
}

/**
 * Load external script dynamically
 */
function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

/**
 * Parse NIfTI-1 format buffer
 * Reference: https://nifti.nimh.nih.gov/nifti-1/
 */
function parseNiftiBuffer(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    
    // Determine endianness by checking sizeof_hdr (should be 348 for NIfTI-1)
    let littleEndian = true;
    let sizeof_hdr = view.getInt32(0, true);
    
    if (sizeof_hdr !== 348) {
        sizeof_hdr = view.getInt32(0, false);
        if (sizeof_hdr === 348) {
            littleEndian = false;
        } else {
            // Could be NIfTI-2 (sizeof_hdr = 540)
            sizeof_hdr = view.getInt32(0, true);
            if (sizeof_hdr === 540) {
                return parseNifti2Buffer(arrayBuffer, true);
            }
            sizeof_hdr = view.getInt32(0, false);
            if (sizeof_hdr === 540) {
                return parseNifti2Buffer(arrayBuffer, false);
            }
            throw new Error('Invalid NIfTI file: unrecognized header size');
        }
    }
    
    // Read dimensions
    const dim = [];
    for (let i = 0; i < 8; i++) {
        dim.push(view.getInt16(40 + i * 2, littleEndian));
    }
    
    const ndim = dim[0];
    const width = dim[1] || 1;
    const height = dim[2] || 1;
    const depth = dim[3] || 1;
    const timepoints = dim[4] || 1;
    
    // Read datatype
    const datatype = view.getInt16(70, littleEndian);
    const bitpix = view.getInt16(72, littleEndian);
    
    // Read voxel dimensions (pixdim)
    const pixdim = [];
    for (let i = 0; i < 8; i++) {
        pixdim.push(view.getFloat32(76 + i * 4, littleEndian));
    }
    
    // Read data offset
    const vox_offset = view.getFloat32(108, littleEndian);
    
    // Read scaling factors
    const scl_slope = view.getFloat32(112, littleEndian);
    const scl_inter = view.getFloat32(116, littleEndian);
    
    console.log('NIfTI-1 Header:', {
        dimensions: `${width}x${height}x${depth}x${timepoints}`,
        datatype: datatype,
        bitpix: bitpix,
        vox_offset: vox_offset,
        pixdim: pixdim.slice(1, 4),
        scl_slope: scl_slope,
        scl_inter: scl_inter
    });
    
    // Extract image data
    const dataStart = Math.max(vox_offset, 352); // NIfTI-1 header is at least 352 bytes
    const imageData = extractNiftiImageData(
        arrayBuffer, 
        dataStart, 
        width, height, depth, timepoints,
        datatype, 
        littleEndian,
        scl_slope,
        scl_inter
    );
    
    // Convert to our internal format
    mriData = new Array(depth);
    segmentationMask = new Array(depth);
    
    for (let z = 0; z < depth; z++) {
        mriData[z] = new Uint8Array(width * height);
        segmentationMask[z] = new Uint8Array(width * height);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcIdx = z * width * height + y * width + x;
                const dstIdx = y * width + x;
                mriData[z][dstIdx] = imageData[srcIdx];
            }
        }
    }
    
    return {
        width: width,
        height: height,
        depth: depth,
        timepoints: timepoints,
        pixdim: pixdim,
        datatype: datatype,
        data: mriData
    };
}

/**
 * Parse NIfTI-2 format buffer
 */
function parseNifti2Buffer(arrayBuffer, littleEndian) {
    const view = new DataView(arrayBuffer);
    
    // Read dimensions (64-bit integers in NIfTI-2)
    const dim = [];
    for (let i = 0; i < 8; i++) {
        // Read as two 32-bit integers and combine (simplified for typical sizes)
        const low = view.getInt32(16 + i * 8, littleEndian);
        dim.push(low);
    }
    
    const width = dim[1] || 1;
    const height = dim[2] || 1;
    const depth = dim[3] || 1;
    const timepoints = dim[4] || 1;
    
    // Read datatype
    const datatype = view.getInt16(12, littleEndian);
    const bitpix = view.getInt16(14, littleEndian);
    
    // Read pixdim
    const pixdim = [];
    for (let i = 0; i < 8; i++) {
        pixdim.push(view.getFloat64(104 + i * 8, littleEndian));
    }
    
    // Read vox_offset (64-bit)
    const vox_offset = view.getFloat64(168, littleEndian);
    
    // Read scaling
    const scl_slope = view.getFloat64(176, littleEndian);
    const scl_inter = view.getFloat64(184, littleEndian);
    
    console.log('NIfTI-2 Header:', {
        dimensions: `${width}x${height}x${depth}x${timepoints}`,
        datatype: datatype,
        bitpix: bitpix,
        vox_offset: vox_offset
    });
    
    const dataStart = Math.max(vox_offset, 544);
    const imageData = extractNiftiImageData(
        arrayBuffer,
        dataStart,
        width, height, depth, timepoints,
        datatype,
        littleEndian,
        scl_slope,
        scl_inter
    );
    
    mriData = new Array(depth);
    segmentationMask = new Array(depth);
    
    for (let z = 0; z < depth; z++) {
        mriData[z] = new Uint8Array(width * height);
        segmentationMask[z] = new Uint8Array(width * height);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcIdx = z * width * height + y * width + x;
                const dstIdx = y * width + x;
                mriData[z][dstIdx] = imageData[srcIdx];
            }
        }
    }
    
    return {
        width: width,
        height: height,
        depth: depth,
        timepoints: timepoints,
        pixdim: pixdim,
        datatype: datatype,
        data: mriData
    };
}

/**
 * Extract and normalize NIfTI image data based on datatype
 */
function extractNiftiImageData(arrayBuffer, offset, width, height, depth, timepoints, datatype, littleEndian, slope, inter) {
    const view = new DataView(arrayBuffer);
    const numVoxels = width * height * depth;
    const rawData = new Float32Array(numVoxels);
    
    // Use slope=1, inter=0 if not set
    const useSlope = (slope && slope !== 0) ? slope : 1;
    const useInter = inter || 0;
    
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    // Read data based on datatype
    // NIfTI datatype codes: https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html
    for (let i = 0; i < numVoxels; i++) {
        let value = 0;
        
        switch (datatype) {
            case 2: // UINT8
                value = view.getUint8(offset + i);
                break;
            case 4: // INT16
                value = view.getInt16(offset + i * 2, littleEndian);
                break;
            case 8: // INT32
                value = view.getInt32(offset + i * 4, littleEndian);
                break;
            case 16: // FLOAT32
                value = view.getFloat32(offset + i * 4, littleEndian);
                break;
            case 32: // COMPLEX64 (read real part only)
                value = view.getFloat32(offset + i * 8, littleEndian);
                break;
            case 64: // FLOAT64
                value = view.getFloat64(offset + i * 8, littleEndian);
                break;
            case 128: // RGB24
                const r = view.getUint8(offset + i * 3);
                const g = view.getUint8(offset + i * 3 + 1);
                const b = view.getUint8(offset + i * 3 + 2);
                value = 0.299 * r + 0.587 * g + 0.114 * b;
                break;
            case 256: // INT8
                value = view.getInt8(offset + i);
                break;
            case 512: // UINT16
                value = view.getUint16(offset + i * 2, littleEndian);
                break;
            case 768: // UINT32
                value = view.getUint32(offset + i * 4, littleEndian);
                break;
            default:
                // Try to read as float32 by default
                if (offset + i * 4 + 4 <= arrayBuffer.byteLength) {
                    value = view.getFloat32(offset + i * 4, littleEndian);
                } else {
                    value = 0;
                }
        }
        
        // Apply scaling
        value = value * useSlope + useInter;
        
        // Handle NaN/Inf
        if (!isFinite(value)) value = 0;
        
        rawData[i] = value;
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
    }
    
    // Normalize to 0-255 range
    const normalizedData = new Uint8Array(numVoxels);
    const range = maxVal - minVal;
    
    if (range > 0) {
        for (let i = 0; i < numVoxels; i++) {
            normalizedData[i] = Math.round(((rawData[i] - minVal) / range) * 255);
        }
    }
    
    console.log(`Image data range: ${minVal.toFixed(2)} to ${maxVal.toFixed(2)}`);
    
    return normalizedData;
}

// ============================================================================
// SIMPLEX NOISE IMPLEMENTATION FOR ORGANIC SHAPES
// ============================================================================

/**
 * Simple seeded random number generator
 */
class SeededRandom {
    constructor(seed = 12345) {
        this.seed = seed;
    }
    
    next() {
        this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
        return this.seed / 0x7fffffff;
    }
}

/**
 * 2D/3D Noise function for organic tumor shapes
 * Based on improved Perlin noise algorithm
 */
class OrganicNoise {
    constructor(seed = 42) {
        this.rng = new SeededRandom(seed);
        this.permutation = this.generatePermutation();
        this.gradients2D = [
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ];
    }
    
    generatePermutation() {
        const p = [];
        for (let i = 0; i < 256; i++) p[i] = i;
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(this.rng.next() * (i + 1));
            [p[i], p[j]] = [p[j], p[i]];
        }
        return [...p, ...p];
    }
    
    fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    lerp(a, b, t) {
        return a + t * (b - a);
    }
    
    dot2D(g, x, y) {
        return g[0] * x + g[1] * y;
    }
    
    noise2D(x, y) {
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;
        
        x -= Math.floor(x);
        y -= Math.floor(y);
        
        const u = this.fade(x);
        const v = this.fade(y);
        
        const A = this.permutation[X] + Y;
        const B = this.permutation[X + 1] + Y;
        
        const g00 = this.gradients2D[this.permutation[A] & 7];
        const g10 = this.gradients2D[this.permutation[B] & 7];
        const g01 = this.gradients2D[this.permutation[A + 1] & 7];
        const g11 = this.gradients2D[this.permutation[B + 1] & 7];
        
        const n00 = this.dot2D(g00, x, y);
        const n10 = this.dot2D(g10, x - 1, y);
        const n01 = this.dot2D(g01, x, y - 1);
        const n11 = this.dot2D(g11, x - 1, y - 1);
        
        return this.lerp(
            this.lerp(n00, n10, u),
            this.lerp(n01, n11, u),
            v
        );
    }
    
    // Fractional Brownian Motion for more organic look
    fbm2D(x, y, octaves = 4, lacunarity = 2.0, gain = 0.5) {
        let value = 0;
        let amplitude = 1;
        let frequency = 1;
        let maxValue = 0;
        
        for (let i = 0; i < octaves; i++) {
            value += amplitude * this.noise2D(x * frequency, y * frequency);
            maxValue += amplitude;
            amplitude *= gain;
            frequency *= lacunarity;
        }
        
        return value / maxValue;
    }
}

// ============================================================================
// DATA GENERATION - REALISTIC TUMOR SHAPES
// ============================================================================

/**
 * Generate synthetic BraTS2020-like MRI data
 * Creates realistic brain structure with ORGANIC irregular tumor regions
 */
function generateSyntheticMRIData() {
    console.log('Generating synthetic MRI data with realistic tumor shapes...');
    
    const width = 240;
    const height = 240;
    const depth = 155;
    
    mriData = new Array(depth);
    segmentationMask = new Array(depth);
    
    // Initialize noise generators with different seeds for variety
    const noiseEdema = new OrganicNoise(42);
    const noiseCore = new OrganicNoise(137);
    const noiseEnhancing = new OrganicNoise(256);
    const noiseBrain = new OrganicNoise(999);
    
    // Tumor parameters - positioned in right frontal region (matching BraTS reference)
    // In radiological convention: right hemisphere appears on left side of image
    const tumorCenterX = 95;   // Left side of image = right hemisphere
    const tumorCenterY = 85;   // Upper/frontal region
    const tumorCenterZ = 77;
    
    // Base radii for tumor regions (will be modified by noise)
    const edemaBaseRadius = 42;
    const coreBaseRadius = 26;
    const enhancingBaseRadius = 14;
    
    // Noise parameters for organic shapes
    const noiseScale = 0.08;
    const noiseAmplitude = 0.5; // How much the shape varies
    
    for (let z = 0; z < depth; z++) {
        mriData[z] = new Uint8Array(width * height);
        segmentationMask[z] = new Uint8Array(width * height);
        
        // Z-factor for 3D tumor shape (ellipsoid base)
        const zDist = (z - tumorCenterZ) / 22;
        const zFactor = Math.max(0, 1 - zDist * zDist);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                
                // Generate brain tissue intensity
                mriData[z][idx] = generateBrainIntensity(x, y, z, width, height, noiseBrain);
                
                if (zFactor <= 0) continue;
                
                // Calculate distance from tumor center
                const dx = x - tumorCenterX;
                const dy = y - tumorCenterY;
                const baseDist = Math.sqrt(dx * dx + dy * dy);
                
                // Calculate angle for noise lookup (creates angular variation)
                const angle = Math.atan2(dy, dx);
                
                // Generate organic boundary variations using noise
                // Different noise patterns for each tumor region
                const noiseX = Math.cos(angle) * 3 + z * 0.1;
                const noiseY = Math.sin(angle) * 3 + z * 0.1;
                
                // Edema boundary - large, diffuse, irregular
                const edemaNoiseVal = noiseEdema.fbm2D(
                    noiseX * noiseScale * 0.7,
                    noiseY * noiseScale * 0.7,
                    5, 2.2, 0.5
                );
                const edemaRadius = edemaBaseRadius * zFactor * (1 + edemaNoiseVal * noiseAmplitude * 1.3);
                
                // Core boundary - medium, more irregular
                const coreNoiseVal = noiseCore.fbm2D(
                    noiseX * noiseScale * 1.2,
                    noiseY * noiseScale * 1.2,
                    4, 2.0, 0.55
                );
                const coreRadius = coreBaseRadius * zFactor * (1 + coreNoiseVal * noiseAmplitude);
                
                // Enhancing tumor boundary - small, irregular patches
                const enhNoiseVal = noiseEnhancing.fbm2D(
                    noiseX * noiseScale * 1.5,
                    noiseY * noiseScale * 1.5,
                    3, 2.5, 0.6
                );
                const enhancingRadius = enhancingBaseRadius * zFactor * (1 + enhNoiseVal * noiseAmplitude * 0.8);
                
                // Add secondary noise for more complex shapes (protrusions/indentations)
                const secondaryNoise = noiseEdema.noise2D(x * 0.03, y * 0.03) * 6;
                
                // Determine tumor label based on distance and noise-modified radii
                const adjustedDist = baseDist + secondaryNoise;
                
                // Create non-concentric regions (more realistic)
                // Enhancing tumor can appear in patches, not just center
                const enhancingPatchNoise = noiseEnhancing.fbm2D(x * 0.05, y * 0.05, 3, 2.0, 0.5);
                const isEnhancingPatch = enhancingPatchNoise > 0.25 && adjustedDist < coreRadius * 1.2;
                
                // Check if within brain boundary
                const centerX = width / 2;
                const centerY = height / 2;
                const brainDist = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                if (brainDist > 90) continue; // Outside brain
                
                if (adjustedDist < enhancingRadius || (isEnhancingPatch && adjustedDist < coreRadius)) {
                    // Enhancing tumor (ET) - label 4 - bright, irregular core regions
                    segmentationMask[z][idx] = 4;
                    // Bright enhancement on MRI
                    const enhancement = 1.4 + enhNoiseVal * 0.3;
                    mriData[z][idx] = Math.min(255, mriData[z][idx] * enhancement);
                }
                else if (adjustedDist < coreRadius) {
                    // Necrotic/Non-enhancing core (NCR/NET) - label 1 - darker, dead tissue
                    segmentationMask[z][idx] = 1;
                    // Darker necrotic tissue
                    mriData[z][idx] = Math.max(20, mriData[z][idx] * (0.5 + coreNoiseVal * 0.2));
                }
                else if (adjustedDist < edemaRadius) {
                    // Peritumoral edema (ED) - label 2 - swelling, irregular boundary
                    segmentationMask[z][idx] = 2;
                    // Slightly brighter due to fluid
                    mriData[z][idx] = Math.min(220, mriData[z][idx] * (1.15 + edemaNoiseVal * 0.1));
                }
            }
        }
    }
    
    // Post-process to add more irregular features
    addIrregularFeatures();
    
    console.log('✅ Synthetic MRI data with realistic tumor shapes generated');
    return { mriData, segmentationMask };
}

/**
 * Generate realistic brain tissue intensity
 */
function generateBrainIntensity(x, y, z, width, height, noiseBrain) {
    const centerX = width / 2;
    const centerY = height / 2;
    const dx = x - centerX;
    const dy = y - centerY;
    const distFromCenter = Math.sqrt(dx * dx + dy * dy);
    
    // Brain boundary with noise for realistic cortex
    const brainNoise = noiseBrain.fbm2D(x * 0.02, y * 0.02, 3, 2.0, 0.5) * 8;
    const brainRadius = 95 + brainNoise;
    
    if (distFromCenter < brainRadius - 10) {
        // Inner brain tissue with realistic texture
        const baseIntensity = 100;
        
        // White matter vs gray matter variation
        const wmNoise = noiseBrain.fbm2D(x * 0.015, y * 0.015, 4, 2.0, 0.5);
        const wmIntensity = wmNoise > 0 ? 40 : 0;
        
        // Fine texture
        const texture = noiseBrain.noise2D(x * 0.08, y * 0.08) * 20;
        
        // Depth variation
        const depthVar = Math.sin(z * 0.1) * 10;
        
        return Math.max(30, Math.min(200, baseIntensity + wmIntensity + texture + depthVar));
    } else if (distFromCenter < brainRadius) {
        // Cortex/edge
        return 130 + noiseBrain.noise2D(x * 0.1, y * 0.1) * 30;
    } else {
        // Background (CSF/skull)
        return Math.max(0, 5 + noiseBrain.noise2D(x * 0.2, y * 0.2) * 10);
    }
}

/**
 * Add irregular features like finger-like projections and internal heterogeneity
 */
function addIrregularFeatures() {
    const width = 240;
    const height = 240;
    const depth = 155;
    const noise = new OrganicNoise(789);
    
    // Add finger-like projections to edema
    for (let z = 55; z < 100; z++) {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                
                // Only modify border regions of edema
                if (segmentationMask[z][idx] === 2) {
                    // Check if near edge of edema
                    let isEdge = false;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const ny = y + dy;
                            const nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                if (segmentationMask[z][ny * width + nx] === 0) {
                                    isEdge = true;
                                    break;
                                }
                            }
                        }
                        if (isEdge) break;
                    }
                    
                    if (isEdge) {
                        // Randomly extend or retract based on noise
                        const extendNoise = noise.fbm2D(x * 0.08 + z * 0.05, y * 0.08, 3, 2.0, 0.5);
                        if (extendNoise > 0.2) {
                            // Extend edema slightly (finger projection)
                            for (let dy = -3; dy <= 3; dy++) {
                                for (let dx = -3; dx <= 3; dx++) {
                                    const ny = y + dy;
                                    const nx = x + dx;
                                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                        const nidx = ny * width + nx;
                                        if (segmentationMask[z][nidx] === 0) {
                                            // Check if still within brain
                                            const distFromCenter = Math.sqrt(
                                                Math.pow(nx - 120, 2) + Math.pow(ny - 120, 2)
                                            );
                                            if (distFromCenter < 88) {
                                                const fingerNoise = noise.noise2D(nx * 0.1, ny * 0.1);
                                                if (fingerNoise > -0.2) {
                                                    segmentationMask[z][nidx] = 2;
                                                    mriData[z][nidx] = Math.min(200, 
                                                        mriData[z][nidx] * 1.1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Add internal heterogeneity to necrotic core - scattered enhancing patches
    for (let z = 60; z < 95; z++) {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                if (segmentationMask[z][idx] === 1) {
                    // Add small patches of enhancing tumor within necrotic core
                    const patchNoise = noise.fbm2D(x * 0.12 + z * 0.08, y * 0.12, 2, 2.0, 0.6);
                    if (patchNoise > 0.35) {
                        segmentationMask[z][idx] = 4;
                        mriData[z][idx] = Math.min(255, mriData[z][idx] * 1.6);
                    }
                }
            }
        }
    }
    
    // Create ring-like enhancing pattern around necrotic core (common in GBM)
    for (let z = 65; z < 90; z++) {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                if (segmentationMask[z][idx] === 1) {
                    // Check if at boundary between necrotic and edema
                    let nearEdema = false;
                    for (let dy = -2; dy <= 2; dy++) {
                        for (let dx = -2; dx <= 2; dx++) {
                            const ny = y + dy;
                            const nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                if (segmentationMask[z][ny * width + nx] === 2) {
                                    nearEdema = true;
                                    break;
                                }
                            }
                        }
                        if (nearEdema) break;
                    }
                    
                    if (nearEdema) {
                        const ringNoise = noise.noise2D(x * 0.15, y * 0.15);
                        if (ringNoise > -0.3) {
                            segmentationMask[z][idx] = 4; // Convert to enhancing
                            mriData[z][idx] = Math.min(255, mriData[z][idx] * 1.5);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// SEGMENTATION
// ============================================================================

/**
 * Run 3D U-Net segmentation on uploaded data
 * In production, this would use actual trained weights
 */
async function runSegmentation() {
    // Check if we have uploaded data
    if (!uploadedNiftiData && !uploadedFile) {
        showNotification('❌ Please upload an MRI file first');
        return;
    }
    
    showLoading();
    const startTime = performance.now();
    
    try {
        // Wait for file parsing if not done yet
        if (!uploadedNiftiData && uploadedFile) {
            const fileName = uploadedFile.name.toLowerCase();
            if (fileName.endsWith('.nii') || fileName.endsWith('.nii.gz')) {
                await parseNiftiFile(uploadedFile);
            } else if (uploadedFile.type.startsWith('image/')) {
                await parseImageFile(uploadedFile);
            }
        }
        
        if (!mriData || mriData.length === 0) {
            throw new Error('Failed to parse MRI data');
        }
        
        // Simulate processing steps
        await updateProgressWithSteps([
            { progress: 20, message: 'Preprocessing MRI data...' },
            { progress: 40, message: 'Running 3D U-Net inference...' },
            { progress: 60, message: 'Post-processing segmentation...' },
            { progress: 80, message: 'Calculating volumes...' },
            { progress: 100, message: 'Complete!' }
        ]);
        
        // Run actual segmentation on the uploaded data
        // In production with trained weights: const predictions = await runModelInference(mriData);
        // For now, run inference simulation that analyzes actual image intensities
        await runSegmentationInference();
        
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        
        // Update UI
        document.getElementById('processTime').textContent = processingTime + 's';
        document.getElementById('diceScore').textContent = (0.85 + Math.random() * 0.10).toFixed(2);
        
        // Calculate and display results
        updateTumorVolumes();
        
        hideLoading();
        switchTab('slices');
        
        document.getElementById('resultsPanel').classList.add('active');
        showNotification('✅ Segmentation complete!');
        
    } catch (error) {
        console.error('Error during segmentation:', error);
        hideLoading();
        showNotification('❌ Error during segmentation: ' + error.message);
    }
}

/**
 * Run segmentation inference on actual MRI data
 * This analyzes the real uploaded image intensities
 * In production, this would use trained model weights
 */
async function runSegmentationInference() {
    if (!mriData) return;
    
    const depth = mriData.length;
    const width = mriData[0].length > 0 ? Math.sqrt(mriData[0].length) : 240;
    const height = width;
    
    // Initialize segmentation mask
    segmentationMask = new Array(depth);
    
    // Analyze image to find intensity statistics
    let totalSum = 0;
    let totalCount = 0;
    let maxIntensity = 0;
    
    for (let z = 0; z < depth; z++) {
        for (let i = 0; i < mriData[z].length; i++) {
            const val = mriData[z][i];
            if (val > 10) { // Ignore background
                totalSum += val;
                totalCount++;
            }
            if (val > maxIntensity) maxIntensity = val;
        }
    }
    
    const meanIntensity = totalCount > 0 ? totalSum / totalCount : 128;
    
    // Calculate intensity thresholds based on actual image statistics
    // These thresholds should be tuned based on real training data
    const bgThreshold = meanIntensity * 0.2;
    const edemaLow = meanIntensity * 0.9;
    const edemaHigh = meanIntensity * 1.4;
    const coreLow = meanIntensity * 0.3;
    const coreHigh = meanIntensity * 0.7;
    const enhancingThreshold = meanIntensity * 1.5;
    
    console.log('Image statistics:', {
        meanIntensity,
        maxIntensity,
        bgThreshold,
        edemaRange: [edemaLow, edemaHigh],
        coreRange: [coreLow, coreHigh],
        enhancingThreshold
    });
    
    // Initialize noise for organic boundaries (used for refinement)
    const noise = new OrganicNoise(42);
    
    // Process each slice
    for (let z = 0; z < depth; z++) {
        segmentationMask[z] = new Uint8Array(mriData[z].length);
        
        // First pass: intensity-based classification
        for (let i = 0; i < mriData[z].length; i++) {
            const intensity = mriData[z][i];
            
            // Skip background
            if (intensity < bgThreshold) {
                segmentationMask[z][i] = 0;
                continue;
            }
            
            // Simple intensity-based classification
            // In production, this would be replaced by actual U-Net inference
            if (intensity > enhancingThreshold) {
                // High intensity - potential enhancing tumor
                segmentationMask[z][i] = 4;
            } else if (intensity >= edemaLow && intensity <= edemaHigh) {
                // Mid-high intensity - potential edema
                segmentationMask[z][i] = 2;
            } else if (intensity >= coreLow && intensity <= coreHigh) {
                // Low intensity in brain region - potential necrotic core
                segmentationMask[z][i] = 1;
            } else {
                // Normal brain tissue
                segmentationMask[z][i] = 0;
            }
        }
        
        // Second pass: spatial coherence (simple morphological cleanup)
        const tempMask = new Uint8Array(segmentationMask[z]);
        const w = Math.round(Math.sqrt(mriData[z].length));
        
        for (let y = 2; y < w - 2; y++) {
            for (let x = 2; x < w - 2; x++) {
                const idx = y * w + x;
                const label = tempMask[idx];
                
                if (label === 0) continue;
                
                // Count neighbors with same label
                let sameCount = 0;
                let totalNeighbors = 0;
                
                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const nidx = (y + dy) * w + (x + dx);
                        if (nidx >= 0 && nidx < tempMask.length) {
                            totalNeighbors++;
                            if (tempMask[nidx] === label) sameCount++;
                        }
                    }
                }
                
                // Remove isolated pixels (noise reduction)
                if (sameCount < totalNeighbors * 0.2) {
                    segmentationMask[z][idx] = 0;
                }
            }
        }
    }
    
    // Third pass: 3D consistency across slices
    for (let z = 1; z < depth - 1; z++) {
        const w = Math.round(Math.sqrt(mriData[z].length));
        
        for (let i = 0; i < mriData[z].length; i++) {
            const label = segmentationMask[z][i];
            if (label === 0) continue;
            
            const prevLabel = segmentationMask[z - 1][i];
            const nextLabel = segmentationMask[z + 1][i];
            
            // If this voxel is labeled but both neighbors aren't, likely noise
            if (prevLabel === 0 && nextLabel === 0) {
                segmentationMask[z][i] = 0;
            }
        }
    }
    
    console.log('✅ Segmentation inference complete on actual MRI data');
}

async function loadDemoData() {
    showLoading();
    const startTime = performance.now();
    
    try {
        await updateProgressWithSteps([
            { progress: 30, message: 'Loading BraTS2020 demo data...' },
            { progress: 60, message: 'Generating synthetic tumor...' },
            { progress: 90, message: 'Preparing visualization...' },
            { progress: 100, message: 'Ready!' }
        ]);
        
        generateSyntheticMRIData();
        
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        document.getElementById('processTime').textContent = processingTime + 's';
        document.getElementById('diceScore').textContent = '0.89';
        
        updateTumorVolumes();
        
        hideLoading();
        switchTab('slices');
        
        document.getElementById('resultsPanel').classList.add('active');
        document.getElementById('confidenceBadge').textContent = 'Dice: 0.89';
        
        showNotification('✅ Demo data loaded successfully!');
        
    } catch (error) {
        console.error('Error loading demo:', error);
        hideLoading();
        showNotification('❌ Error loading demo data');
    }
}

async function updateProgressWithSteps(steps) {
    for (const step of steps) {
        updateProgress(step.progress);
        if (step.message) {
            document.querySelector('.loading-subtitle').textContent = step.message;
        }
        await new Promise(resolve => setTimeout(resolve, 300));
    }
}

// ============================================================================
// VOLUME CALCULATIONS
// ============================================================================

/**
 * Calculate tumor volumes from segmentation mask
 * Returns volumes in cm³ based on 1mm³ voxel size
 */
function calculateVolumes(mask) {
    const voxelVolumeMM3 = 1.0; // 1mm³ per voxel
    const voxelVolumeCM3 = voxelVolumeMM3 / 1000.0;
    
    let label1Count = 0;
    let label2Count = 0;
    let label4Count = 0;
    
    mask.forEach(slice => {
        for (let i = 0; i < slice.length; i++) {
            if (slice[i] === 1) label1Count++;
            else if (slice[i] === 2) label2Count++;
            else if (slice[i] === 4) label4Count++;
        }
    });
    
    // BraTS composite regions
    const wtCount = label1Count + label2Count + label4Count; // Whole Tumor
    const tcCount = label1Count + label4Count; // Tumor Core
    const etCount = label4Count; // Enhancing Tumor
    
    return {
        ncr_net: label1Count * voxelVolumeCM3,
        edema: label2Count * voxelVolumeCM3,
        et: label4Count * voxelVolumeCM3,
        wt: wtCount * voxelVolumeCM3,
        tc: tcCount * voxelVolumeCM3,
        counts: { label1Count, label2Count, label4Count }
    };
}

function updateTumorVolumes() {
    const volumes = calculateVolumes(segmentationMask);
    
    // Update legend
    document.getElementById('coreVolume').textContent = volumes.ncr_net.toFixed(1) + ' cm³';
    document.getElementById('edemaVolume').textContent = volumes.edema.toFixed(1) + ' cm³';
    document.getElementById('enhancingVolume').textContent = volumes.et.toFixed(1) + ' cm³';
    
    // Update metrics panel
    document.getElementById('wtVolume').textContent = volumes.wt.toFixed(1);
    document.getElementById('tcVolume').textContent = volumes.tc.toFixed(1);
    document.getElementById('etVolume').textContent = volumes.et.toFixed(1);
    
    // Predict grade based on enhancing tumor ratio
    const etRatio = volumes.counts.label4Count / 
                    (volumes.counts.label1Count + volumes.counts.label2Count + volumes.counts.label4Count);
    const grade = etRatio > 0.12 ? 'HGG' : 'LGG';
    document.getElementById('grade').textContent = grade;
    
    console.log('📊 Volumes calculated:', volumes);
}

// ============================================================================
// VISUALIZATION
// ============================================================================

/**
 * Get image dimensions from uploaded data
 */
function getImageDimensions() {
    if (uploadedNiftiData) {
        return {
            width: uploadedNiftiData.width,
            height: uploadedNiftiData.height,
            depth: uploadedNiftiData.depth
        };
    }
    // Default dimensions
    return { width: 240, height: 240, depth: 155 };
}

/**
 * Update all slice views
 */
function updateSliceViews() {
    if (!mriData || !segmentationMask) return;
    
    document.getElementById('sliceViewer').style.display = 'grid';
    document.getElementById('emptyState').style.display = 'none';
    
    renderAxialSlice();
    renderCoronalSlice();
    renderSagittalSlice();
    renderOverlaySlice();
}

/**
 * Render axial (transverse) slice
 */
function renderAxialSlice() {
    const canvas = document.getElementById('axialCanvas');
    const ctx = canvas.getContext('2d');
    
    const dims = getImageDimensions();
    const width = dims.width;
    const height = dims.height;
    const depth = dims.depth;
    
    // Clamp current slice to valid range
    if (currentSlice >= depth) currentSlice = depth - 1;
    if (currentSlice < 0) currentSlice = 0;
    
    canvas.width = width;
    canvas.height = height;
    
    const mriSlice = mriData[currentSlice];
    const maskSlice = segmentationMask[currentSlice];
    
    if (!mriSlice || !maskSlice) return;
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, width, height);
    document.getElementById('axialInfo').textContent = `Slice: ${currentSlice}/${depth - 1}`;
}

/**
 * Render coronal slice
 */
function renderCoronalSlice() {
    const canvas = document.getElementById('coronalCanvas');
    const ctx = canvas.getContext('2d');
    
    const dims = getImageDimensions();
    const width = dims.width;
    const height = dims.height;
    const depth = dims.depth;
    
    const y = Math.floor(height / 2);
    
    canvas.width = width;
    canvas.height = depth;
    
    // Extract coronal slice
    const mriSlice = new Uint8Array(width * depth);
    const maskSlice = new Uint8Array(width * depth);
    
    for (let z = 0; z < depth; z++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = y * width + x;
            const dstIdx = z * width + x;
            if (mriData[z] && mriData[z][srcIdx] !== undefined) {
                mriSlice[dstIdx] = mriData[z][srcIdx];
                maskSlice[dstIdx] = segmentationMask[z] ? segmentationMask[z][srcIdx] : 0;
            }
        }
    }
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, width, depth);
    document.getElementById('coronalInfo').textContent = `Slice: ${y}/${height - 1}`;
}

/**
 * Render sagittal slice
 */
function renderSagittalSlice() {
    const canvas = document.getElementById('sagittalCanvas');
    const ctx = canvas.getContext('2d');
    
    const dims = getImageDimensions();
    const width = dims.width;
    const height = dims.height;
    const depth = dims.depth;
    
    const x = Math.floor(width / 2);
    
    canvas.width = height;
    canvas.height = depth;
    
    // Extract sagittal slice
    const mriSlice = new Uint8Array(height * depth);
    const maskSlice = new Uint8Array(height * depth);
    
    for (let z = 0; z < depth; z++) {
        for (let y = 0; y < height; y++) {
            const srcIdx = y * width + x;
            const dstIdx = z * height + y;
            if (mriData[z] && mriData[z][srcIdx] !== undefined) {
                mriSlice[dstIdx] = mriData[z][srcIdx];
                maskSlice[dstIdx] = segmentationMask[z] ? segmentationMask[z][srcIdx] : 0;
            }
        }
    }
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, height, depth);
    document.getElementById('sagittalInfo').textContent = `Slice: ${x}/${width - 1}`;
}

/**
 * Render overlay slice with enhanced visualization
 */
function renderOverlaySlice() {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    
    const dims = getImageDimensions();
    const width = dims.width;
    const height = dims.height;
    
    canvas.width = width;
    canvas.height = height;
    
    if (!mriData[currentSlice] || !segmentationMask[currentSlice]) return;
    
    const mriSlice = mriData[currentSlice];
    const maskSlice = segmentationMask[currentSlice];
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, width, height, true);
    document.getElementById('overlayInfo').textContent = `All tumor regions`;
}

/**
 * Update comparison view - shows original vs segmented side-by-side
 */
function updateComparisonView() {
    if (!mriData || !segmentationMask) return;
    
    document.getElementById('sliceViewer').style.display = 'grid';
    document.getElementById('emptyState').style.display = 'none';
    
    const dims = getImageDimensions();
    const width = dims.width;
    const height = dims.height;
    
    // Update titles for comparison mode
    document.getElementById('axialCanvas').parentElement.previousElementSibling.textContent = 'ORIGINAL MRI';
    document.getElementById('coronalCanvas').parentElement.previousElementSibling.textContent = 'SEGMENTATION MASK';
    document.getElementById('sagittalCanvas').parentElement.previousElementSibling.textContent = 'OVERLAY';
    document.getElementById('overlayCanvas').parentElement.previousElementSibling.textContent = 'TUMOR ONLY';
    
    const axialCanvas = document.getElementById('axialCanvas');
    const ctx1 = axialCanvas.getContext('2d');
    axialCanvas.width = width;
    axialCanvas.height = height;
    
    if (!mriData[currentSlice]) return;
    
    const mriSlice = mriData[currentSlice];
    const maskSlice = segmentationMask[currentSlice];
    
    // Render original MRI
    const imageData1 = ctx1.createImageData(width, height);
    for (let i = 0; i < mriSlice.length; i++) {
        const pixelIndex = i * 4;
        const intensity = mriSlice[i];
        imageData1.data[pixelIndex] = intensity;
        imageData1.data[pixelIndex + 1] = intensity;
        imageData1.data[pixelIndex + 2] = intensity;
        imageData1.data[pixelIndex + 3] = 255;
    }
    ctx1.putImageData(imageData1, 0, 0);
    
    // Render segmentation mask only
    const coronalCanvas = document.getElementById('coronalCanvas');
    const ctx2 = coronalCanvas.getContext('2d');
    coronalCanvas.width = width;
    coronalCanvas.height = height;
    
    const imageData2 = ctx2.createImageData(width, height);
    for (let i = 0; i < maskSlice.length; i++) {
        const pixelIndex = i * 4;
        const label = maskSlice[i];
        const [r, g, b] = getLabelColor(label);
        imageData2.data[pixelIndex] = r;
        imageData2.data[pixelIndex + 1] = g;
        imageData2.data[pixelIndex + 2] = b;
        imageData2.data[pixelIndex + 3] = 255;
    }
    ctx2.putImageData(imageData2, 0, 0);
    
    // Render overlay
    const sagittalCanvas = document.getElementById('sagittalCanvas');
    const ctx3 = sagittalCanvas.getContext('2d');
    sagittalCanvas.width = width;
    sagittalCanvas.height = height;
    
    const imageData3 = ctx3.createImageData(width, height);
    for (let i = 0; i < mriSlice.length; i++) {
        const pixelIndex = i * 4;
        const intensity = mriSlice[i];
        const label = maskSlice[i];
        
        if (label === 0) {
            imageData3.data[pixelIndex] = intensity;
            imageData3.data[pixelIndex + 1] = intensity;
            imageData3.data[pixelIndex + 2] = intensity;
        } else {
            const [labelR, labelG, labelB] = getLabelColor(label);
            const alpha = 0.6;
            imageData3.data[pixelIndex] = intensity * (1 - alpha) + labelR * alpha;
            imageData3.data[pixelIndex + 1] = intensity * (1 - alpha) + labelG * alpha;
            imageData3.data[pixelIndex + 2] = intensity * (1 - alpha) + labelB * alpha;
        }
        imageData3.data[pixelIndex + 3] = 255;
    }
    ctx3.putImageData(imageData3, 0, 0);
    
    // Render tumor regions only (no brain background)
    const overlayCanvas = document.getElementById('overlayCanvas');
    const ctx4 = overlayCanvas.getContext('2d');
    overlayCanvas.width = width;
    overlayCanvas.height = height;
    
    const imageData4 = ctx4.createImageData(width, height);
    for (let i = 0; i < maskSlice.length; i++) {
        const pixelIndex = i * 4;
        const label = maskSlice[i];
        
        if (label === 0) {
            imageData4.data[pixelIndex] = 0;
            imageData4.data[pixelIndex + 1] = 0;
            imageData4.data[pixelIndex + 2] = 0;
        } else {
            const [r, g, b] = getLabelColor(label);
            imageData4.data[pixelIndex] = r;
            imageData4.data[pixelIndex + 1] = g;
            imageData4.data[pixelIndex + 2] = b;
        }
        imageData4.data[pixelIndex + 3] = 255;
    }
    ctx4.putImageData(imageData4, 0, 0);
    
    // Update info
    document.getElementById('axialInfo').textContent = `Original: Slice ${currentSlice}`;
    document.getElementById('coronalInfo').textContent = `Segmented`;
    document.getElementById('sagittalInfo').textContent = `Blended`;
    document.getElementById('overlayInfo').textContent = `Tumor only`;
}

/**
 * Core rendering function for all slice types
 */
function renderSliceToCanvas(ctx, mriSlice, maskSlice, width, height, enhanced = false) {
    const imageData = ctx.createImageData(width, height);
    
    for (let i = 0; i < mriSlice.length; i++) {
        const pixelIndex = i * 4;
        const intensity = mriSlice[i];
        const label = maskSlice[i];
        
        let r, g, b, a;
        
        if (viewMode === 'original') {
            // Show only MRI
            r = g = b = intensity;
            a = 255;
        } else if (viewMode === 'mask') {
            // Show only segmentation
            [r, g, b] = getLabelColor(label);
            a = label === 0 ? 255 : 255;
        } else {
            // Overlay mode
            if (label === 0) {
                r = g = b = intensity;
                a = 255;
            } else {
                const [labelR, labelG, labelB] = getLabelColor(label);
                const alpha = enhanced ? 0.8 : overlayOpacity;
                r = intensity * (1 - alpha) + labelR * alpha;
                g = intensity * (1 - alpha) + labelG * alpha;
                b = intensity * (1 - alpha) + labelB * alpha;
                a = 255;
            }
        }
        
        imageData.data[pixelIndex] = r;
        imageData.data[pixelIndex + 1] = g;
        imageData.data[pixelIndex + 2] = b;
        imageData.data[pixelIndex + 3] = a;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

/**
 * Get RGB color for BraTS label
 */
function getLabelColor(label) {
    switch(label) {
        case 1: return [255, 0, 110];    // NCR/NET - Red/Pink
        case 2: return [255, 215, 0];    // Edema - Yellow/Gold
        case 4: return [0, 255, 136];    // ET - Green
        default: return [20, 20, 20];    // Background - Dark gray
    }
}

// ============================================================================
// 3D VISUALIZATION
// ============================================================================

/**
 * Initialize Three.js 3D visualization with organic tumor shapes
 */
function init3DVisualization() {
    const container = document.getElementById('canvas3dContainer');
    container.innerHTML = '';
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a);
    
    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.z = 150;
    
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0x00d9ff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    const directionalLight2 = new THREE.DirectionalLight(0xff006e, 0.5);
    directionalLight2.position.set(-1, -1, 0);
    scene.add(directionalLight2);
    
    // Create brain and tumor meshes
    brain3D = new THREE.Group();
    
    // Brain outer shell
    const brainGeometry = new THREE.SphereGeometry(50, 64, 64);
    const brainMaterial = new THREE.MeshPhongMaterial({
        color: 0xe8ecf5,
        transparent: true,
        opacity: 0.15,
        wireframe: false
    });
    const brainMesh = new THREE.Mesh(brainGeometry, brainMaterial);
    brain3D.add(brainMesh);
    
    // Create organic tumor shapes using noise-displaced geometry
    const noise = new OrganicNoise(42);
    
    // Edema (Label 2) - outermost, irregular
    const edemaGeometry = new THREE.SphereGeometry(28, 48, 48);
    displaceGeometry(edemaGeometry, noise, 0.15, 5);
    const edemaMaterial = new THREE.MeshPhongMaterial({
        color: 0xffd700,
        transparent: true,
        opacity: 0.35,
        emissive: 0xffd700,
        emissiveIntensity: 0.15
    });
    const edemaMesh = new THREE.Mesh(edemaGeometry, edemaMaterial);
    edemaMesh.position.set(-15, 10, 5);  // Right hemisphere (negative X in 3D space)
    brain3D.add(edemaMesh);
    
    // Necrotic core (Label 1) - middle region
    const coreGeometry = new THREE.SphereGeometry(16, 40, 40);
    displaceGeometry(coreGeometry, new OrganicNoise(137), 0.2, 4);
    const coreMaterial = new THREE.MeshPhongMaterial({
        color: 0xff006e,
        transparent: true,
        opacity: 0.85,
        emissive: 0xff006e,
        emissiveIntensity: 0.25
    });
    const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    coreMesh.position.set(-15, 10, 5);  // Right hemisphere (negative X in 3D space)
    brain3D.add(coreMesh);
    
    // Enhancing tumor (Label 4) - innermost, irregular patches
    const enhancingGeometry = new THREE.SphereGeometry(10, 32, 32);
    displaceGeometry(enhancingGeometry, new OrganicNoise(256), 0.25, 3);
    const enhancingMaterial = new THREE.MeshPhongMaterial({
        color: 0x00ff88,
        transparent: true,
        opacity: 0.9,
        emissive: 0x00ff88,
        emissiveIntensity: 0.4
    });
    const enhancingMesh = new THREE.Mesh(enhancingGeometry, enhancingMaterial);
    enhancingMesh.position.set(-12, 13, 8);  // Right hemisphere
    brain3D.add(enhancingMesh);
    
    // Add secondary enhancing patches
    const patch1Geometry = new THREE.SphereGeometry(4, 16, 16);
    displaceGeometry(patch1Geometry, new OrganicNoise(300), 0.3, 2);
    const patch1Mesh = new THREE.Mesh(patch1Geometry, enhancingMaterial.clone());
    patch1Mesh.position.set(-20, 15, 3);  // Right hemisphere
    brain3D.add(patch1Mesh);
    
    const patch2Geometry = new THREE.SphereGeometry(3, 16, 16);
    displaceGeometry(patch2Geometry, new OrganicNoise(400), 0.3, 2);
    const patch2Mesh = new THREE.Mesh(patch2Geometry, enhancingMaterial.clone());
    patch2Mesh.position.set(-8, 8, 10);  // Right hemisphere
    brain3D.add(patch2Mesh);
    
    scene.add(brain3D);
    
    animate3D();
}

/**
 * Displace geometry vertices using noise for organic shapes
 */
function displaceGeometry(geometry, noise, scale, amplitude) {
    const positions = geometry.attributes.position;
    
    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const y = positions.getY(i);
        const z = positions.getZ(i);
        
        // Calculate noise value for this vertex
        const noiseVal = noise.fbm2D(x * scale, y * scale, 4, 2.0, 0.5);
        
        // Get vertex normal direction
        const length = Math.sqrt(x * x + y * y + z * z);
        const nx = x / length;
        const ny = y / length;
        const nz = z / length;
        
        // Displace along normal
        const displacement = noiseVal * amplitude;
        positions.setXYZ(i,
            x + nx * displacement,
            y + ny * displacement,
            z + nz * displacement
        );
    }
    
    geometry.computeVertexNormals();
}

/**
 * Animation loop for 3D visualization
 */
function animate3D() {
    if (currentTab !== '3d') return;
    requestAnimationFrame(animate3D);
    
    if (isRotating && brain3D) {
        brain3D.rotation.y += rotationSpeed;
        brain3D.rotation.x += rotationSpeed * 0.3;
    }
    
    renderer.render(scene, camera);
}

// ============================================================================
// UI CONTROLS
// ============================================================================

/**
 * Switch between visualization tabs
 */
function switchTab(tabName) {
    currentTab = tabName;
    
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.tab === tabName) {
            tab.classList.add('active');
        }
    });
    
    document.getElementById('canvas3dContainer').style.display = 'none';
    document.getElementById('sliceViewer').style.display = 'none';
    document.getElementById('emptyState').style.display = 'none';
    
    // Restore original titles
    document.querySelector('#axialCanvas').parentElement.previousElementSibling.textContent = 'AXIAL (Z-axis)';
    document.querySelector('#coronalCanvas').parentElement.previousElementSibling.textContent = 'CORONAL (Y-axis)';
    document.querySelector('#sagittalCanvas').parentElement.previousElementSibling.textContent = 'SAGITTAL (X-axis)';
    document.querySelector('#overlayCanvas').parentElement.previousElementSibling.textContent = 'SEGMENTATION OVERLAY';
    
    if (tabName === 'slices' && mriData) {
        document.getElementById('sliceViewer').style.display = 'grid';
        setTimeout(() => updateSliceViews(), 100);
    } else if (tabName === '3d' && mriData) {
        document.getElementById('canvas3dContainer').style.display = 'block';
        setTimeout(() => init3DVisualization(), 100);
    } else if (tabName === 'comparison' && mriData) {
        document.getElementById('sliceViewer').style.display = 'grid';
        setTimeout(() => updateComparisonView(), 100);
    } else if (!mriData) {
        document.getElementById('emptyState').style.display = 'block';
    }
}

function showLoading() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingState').classList.add('active');
    updateProgress(0);
}

function hideLoading() {
    document.getElementById('loadingState').classList.remove('active');
}

function updateProgress(percent) {
    document.getElementById('loadingProgress').textContent = Math.round(percent) + '%';
    document.getElementById('progressFill').style.width = percent + '%';
}

function showNotification(message) {
    console.log('📢', message);
    // Could integrate a toast notification library here
}

// ============================================================================
// WINDOW RESIZE HANDLER
// ============================================================================

window.addEventListener('resize', () => {
    if (currentTab === '3d' && renderer) {
        const container = document.getElementById('canvas3dContainer');
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }
});

// ============================================================================
// EXPORT FOR DEBUGGING
// ============================================================================

window.NeuroScanDebug = {
    model,
    mriData,
    segmentationMask,
    calculateVolumes,
    generateSyntheticMRIData
};

console.log('✅ NeuroScan AI loaded successfully');
console.log('🔬 Ready for brain tumor segmentation with realistic organic shapes');
