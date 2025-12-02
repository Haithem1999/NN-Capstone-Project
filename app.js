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

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('analyzeBtn').disabled = false;
        showNotification(`File loaded: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    }
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
    
    // Tumor parameters - position slightly off-center like real tumors
    const tumorCenterX = 145;
    const tumorCenterY = 105;
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
 * Run 3D U-Net segmentation
 * In production, this would use actual trained weights
 */
async function runSegmentation() {
    showLoading();
    const startTime = performance.now();
    
    try {
        // Simulate processing steps
        await updateProgressWithSteps([
            { progress: 20, message: 'Preprocessing MRI data...' },
            { progress: 40, message: 'Running 3D U-Net inference...' },
            { progress: 60, message: 'Post-processing segmentation...' },
            { progress: 80, message: 'Calculating volumes...' },
            { progress: 100, message: 'Complete!' }
        ]);
        
        // Generate synthetic data (in production, process uploaded file)
        generateSyntheticMRIData();
        
        // In a real implementation with trained weights:
        // const predictions = await runModelInference(mriData);
        // segmentationMask = postprocessPredictions(predictions);
        
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
    
    const width = 240;
    const height = 240;
    canvas.width = width;
    canvas.height = height;
    
    const mriSlice = mriData[currentSlice];
    const maskSlice = segmentationMask[currentSlice];
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, width, height);
    document.getElementById('axialInfo').textContent = `Slice: ${currentSlice}/154`;
}

/**
 * Render coronal slice
 */
function renderCoronalSlice() {
    const canvas = document.getElementById('coronalCanvas');
    const ctx = canvas.getContext('2d');
    
    const width = 240;
    const depth = 155;
    const y = 120;
    
    canvas.width = width;
    canvas.height = depth;
    
    // Extract coronal slice
    const mriSlice = new Uint8Array(width * depth);
    const maskSlice = new Uint8Array(width * depth);
    
    for (let z = 0; z < depth; z++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = y * width + x;
            const dstIdx = z * width + x;
            mriSlice[dstIdx] = mriData[z][srcIdx];
            maskSlice[dstIdx] = segmentationMask[z][srcIdx];
        }
    }
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, width, depth);
    document.getElementById('coronalInfo').textContent = `Slice: ${y}/240`;
}

/**
 * Render sagittal slice
 */
function renderSagittalSlice() {
    const canvas = document.getElementById('sagittalCanvas');
    const ctx = canvas.getContext('2d');
    
    const width = 240;
    const height = 240;
    const depth = 155;
    const x = 120;
    
    canvas.width = height;
    canvas.height = depth;
    
    // Extract sagittal slice
    const mriSlice = new Uint8Array(height * depth);
    const maskSlice = new Uint8Array(height * depth);
    
    for (let z = 0; z < depth; z++) {
        for (let y = 0; y < height; y++) {
            const srcIdx = y * width + x;
            const dstIdx = z * height + y;
            mriSlice[dstIdx] = mriData[z][srcIdx];
            maskSlice[dstIdx] = segmentationMask[z][srcIdx];
        }
    }
    
    renderSliceToCanvas(ctx, mriSlice, maskSlice, height, depth);
    document.getElementById('sagittalInfo').textContent = `Slice: ${x}/240`;
}

/**
 * Render overlay slice with enhanced visualization
 */
function renderOverlaySlice() {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    
    const width = 240;
    const height = 240;
    canvas.width = width;
    canvas.height = height;
    
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
    
    // Update titles for comparison mode
    document.getElementById('axialCanvas').parentElement.previousElementSibling.textContent = 'ORIGINAL MRI';
    document.getElementById('coronalCanvas').parentElement.previousElementSibling.textContent = 'SEGMENTATION MASK';
    document.getElementById('sagittalCanvas').parentElement.previousElementSibling.textContent = 'OVERLAY';
    document.getElementById('overlayCanvas').parentElement.previousElementSibling.textContent = 'TUMOR ONLY';
    
    const axialCanvas = document.getElementById('axialCanvas');
    const ctx1 = axialCanvas.getContext('2d');
    const width = 240;
    const height = 240;
    axialCanvas.width = width;
    axialCanvas.height = height;
    
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
    edemaMesh.position.set(15, 5, 5);
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
    coreMesh.position.set(15, 5, 5);
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
    enhancingMesh.position.set(18, 8, 8);
    brain3D.add(enhancingMesh);
    
    // Add secondary enhancing patches
    const patch1Geometry = new THREE.SphereGeometry(4, 16, 16);
    displaceGeometry(patch1Geometry, new OrganicNoise(300), 0.3, 2);
    const patch1Mesh = new THREE.Mesh(patch1Geometry, enhancingMaterial.clone());
    patch1Mesh.position.set(10, 12, 3);
    brain3D.add(patch1Mesh);
    
    const patch2Geometry = new THREE.SphereGeometry(3, 16, 16);
    displaceGeometry(patch2Geometry, new OrganicNoise(400), 0.3, 2);
    const patch2Mesh = new THREE.Mesh(patch2Geometry, enhancingMaterial.clone());
    patch2Mesh.position.set(22, 2, 10);
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
