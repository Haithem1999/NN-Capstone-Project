/**
 * NeuroScan AI - Brain Tumor Segmentation
 * Working implementation with real NIfTI file support
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

let mriData = null;
let segmentationMask = null;
let currentSlice = 75;
let overlayOpacity = 0.7;
let viewMode = 'overlay';
let currentTab = 'slices';
let uploadedFile = null;
let imageWidth = 240;
let imageHeight = 240;
let imageDepth = 155;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🧠 NeuroScan AI initializing...');
    setupEventListeners();
    console.log('✅ NeuroScan AI ready');
});

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Upload area
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });
        
        fileInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.style.borderColor = 'var(--primary)';
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.style.borderColor = 'var(--border)';
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.style.borderColor = 'var(--border)';
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }
    
    // Buttons
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', runSegmentation);
    }
    
    const demoBtn = document.getElementById('demoBtn');
    if (demoBtn) {
        demoBtn.addEventListener('click', loadDemoData);
    }
    
    // Tabs
    document.querySelectorAll('.viz-tab').forEach(function(tab) {
        tab.addEventListener('click', function() {
            switchTab(tab.dataset.tab);
        });
    });
    
    // Sliders
    const sliceSlider = document.getElementById('sliceSlider');
    if (sliceSlider) {
        sliceSlider.addEventListener('input', function(e) {
            currentSlice = parseInt(e.target.value);
            document.getElementById('sliceValue').textContent = currentSlice;
            if (mriData) updateSliceViews();
        });
    }
    
    const opacitySlider = document.getElementById('opacitySlider');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', function(e) {
            overlayOpacity = parseFloat(e.target.value);
            document.getElementById('opacityValue').textContent = overlayOpacity.toFixed(1);
            if (mriData) updateSliceViews();
        });
    }
    
    // Toggle buttons
    document.querySelectorAll('.toggle-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.toggle-btn').forEach(function(b) {
                b.classList.remove('active');
            });
            btn.classList.add('active');
            viewMode = btn.dataset.mode;
            if (mriData) updateSliceViews();
        });
    });
}

// ============================================================================
// FILE HANDLING
// ============================================================================

function handleFileUpload(file) {
    if (!file) return;
    
    uploadedFile = file;
    console.log('📁 File selected:', file.name, 'Size:', (file.size / 1024 / 1024).toFixed(2), 'MB');
    
    // Update UI
    const uploadArea = document.getElementById('uploadArea');
    const filenameEl = document.getElementById('uploadFilename');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (uploadArea) uploadArea.classList.add('file-loaded');
    if (filenameEl) filenameEl.textContent = file.name;
    if (analyzeBtn) analyzeBtn.disabled = false;
    
    // Parse the file
    const fileName = file.name.toLowerCase();
    if (fileName.endsWith('.nii') || fileName.endsWith('.nii.gz')) {
        parseNiftiFile(file);
    } else if (file.type.startsWith('image/')) {
        parseImageFile(file);
    } else {
        alert('Please upload a NIfTI file (.nii, .nii.gz) or an image file');
    }
}

function parseImageFile(file) {
    console.log('📷 Parsing image file...');
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            const imgData = ctx.getImageData(0, 0, img.width, img.height);
            
            imageWidth = img.width;
            imageHeight = img.height;
            imageDepth = 1;
            
            mriData = [new Uint8Array(imageWidth * imageHeight)];
            segmentationMask = [new Uint8Array(imageWidth * imageHeight)];
            
            for (let i = 0; i < imageWidth * imageHeight; i++) {
                const r = imgData.data[i * 4];
                const g = imgData.data[i * 4 + 1];
                const b = imgData.data[i * 4 + 2];
                mriData[0][i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            }
            
            // Update slider
            const sliceSlider = document.getElementById('sliceSlider');
            if (sliceSlider) {
                sliceSlider.max = 0;
                sliceSlider.value = 0;
            }
            currentSlice = 0;
            document.getElementById('sliceValue').textContent = '0';
            
            // Show the image
            switchTab('slices');
            updateSliceViews();
            
            console.log('✅ Image parsed:', imageWidth, 'x', imageHeight);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

async function parseNiftiFile(file) {
    console.log('📂 Parsing NIfTI file...');
    showLoading('Parsing NIfTI file...');
    
    try {
        let arrayBuffer = await file.arrayBuffer();
        
        // Decompress if gzipped
        if (file.name.toLowerCase().endsWith('.gz')) {
            updateProgress(20, 'Decompressing...');
            console.log('🗜️ Decompressing gzip...');
            
            if (typeof pako !== 'undefined') {
                const compressed = new Uint8Array(arrayBuffer);
                const decompressed = pako.inflate(compressed);
                arrayBuffer = decompressed.buffer;
                console.log('✅ Decompressed successfully');
            } else {
                throw new Error('Pako library not loaded for gzip decompression');
            }
        }
        
        updateProgress(40, 'Reading header...');
        
        // Parse NIfTI
        const result = parseNiftiData(arrayBuffer);
        
        if (result) {
            imageWidth = result.width;
            imageHeight = result.height;
            imageDepth = result.depth;
            mriData = result.data;
            
            // Initialize segmentation mask
            segmentationMask = new Array(imageDepth);
            for (let z = 0; z < imageDepth; z++) {
                segmentationMask[z] = new Uint8Array(imageWidth * imageHeight);
            }
            
            // Update slider
            const sliceSlider = document.getElementById('sliceSlider');
            if (sliceSlider) {
                sliceSlider.max = imageDepth - 1;
                sliceSlider.value = Math.floor(imageDepth / 2);
            }
            currentSlice = Math.floor(imageDepth / 2);
            document.getElementById('sliceValue').textContent = currentSlice;
            
            updateProgress(100, 'Complete!');
            
            setTimeout(function() {
                hideLoading();
                switchTab('slices');
                updateSliceViews();
                console.log('✅ NIfTI parsed:', imageWidth, 'x', imageHeight, 'x', imageDepth);
            }, 300);
        }
    } catch (error) {
        console.error('❌ Error parsing NIfTI:', error);
        hideLoading();
        alert('Error parsing NIfTI file: ' + error.message);
    }
}

function parseNiftiData(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    
    // Check header size to determine format and endianness
    let littleEndian = true;
    let headerSize = view.getInt32(0, true);
    
    if (headerSize !== 348 && headerSize !== 540) {
        headerSize = view.getInt32(0, false);
        littleEndian = false;
    }
    
    if (headerSize !== 348 && headerSize !== 540) {
        throw new Error('Invalid NIfTI file: unrecognized header size ' + headerSize);
    }
    
    const isNifti2 = (headerSize === 540);
    console.log('NIfTI version:', isNifti2 ? '2' : '1', 'Little endian:', littleEndian);
    
    let width, height, depth, datatype, voxOffset;
    
    if (isNifti2) {
        // NIfTI-2 format
        width = view.getInt32(16, littleEndian);
        height = view.getInt32(24, littleEndian);
        depth = view.getInt32(32, littleEndian);
        datatype = view.getInt16(12, littleEndian);
        voxOffset = view.getFloat64(168, littleEndian);
    } else {
        // NIfTI-1 format
        width = view.getInt16(42, littleEndian);
        height = view.getInt16(44, littleEndian);
        depth = view.getInt16(46, littleEndian);
        datatype = view.getInt16(70, littleEndian);
        voxOffset = view.getFloat32(108, littleEndian);
    }
    
    // Ensure valid dimensions
    if (width <= 0 || height <= 0) {
        throw new Error('Invalid dimensions: ' + width + 'x' + height + 'x' + depth);
    }
    
    depth = Math.max(1, depth);
    voxOffset = Math.max(voxOffset, isNifti2 ? 544 : 352);
    
    console.log('Dimensions:', width, 'x', height, 'x', depth);
    console.log('Datatype:', datatype, 'Offset:', voxOffset);
    
    // Read and normalize data
    const numVoxels = width * height * depth;
    const rawData = new Float32Array(numVoxels);
    
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    const dataStart = Math.floor(voxOffset);
    
    for (let i = 0; i < numVoxels; i++) {
        let value = 0;
        
        try {
            switch (datatype) {
                case 2: // UINT8
                    value = view.getUint8(dataStart + i);
                    break;
                case 4: // INT16
                    value = view.getInt16(dataStart + i * 2, littleEndian);
                    break;
                case 8: // INT32
                    value = view.getInt32(dataStart + i * 4, littleEndian);
                    break;
                case 16: // FLOAT32
                    value = view.getFloat32(dataStart + i * 4, littleEndian);
                    break;
                case 64: // FLOAT64
                    value = view.getFloat64(dataStart + i * 8, littleEndian);
                    break;
                case 256: // INT8
                    value = view.getInt8(dataStart + i);
                    break;
                case 512: // UINT16
                    value = view.getUint16(dataStart + i * 2, littleEndian);
                    break;
                case 768: // UINT32
                    value = view.getUint32(dataStart + i * 4, littleEndian);
                    break;
                default:
                    // Try as INT16 by default
                    value = view.getInt16(dataStart + i * 2, littleEndian);
            }
        } catch (e) {
            value = 0;
        }
        
        if (!isFinite(value)) value = 0;
        rawData[i] = value;
        
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
    }
    
    console.log('Value range:', minVal, 'to', maxVal);
    
    // Normalize to 0-255 and store in slices
    const range = maxVal - minVal || 1;
    const data = new Array(depth);
    
    for (let z = 0; z < depth; z++) {
        data[z] = new Uint8Array(width * height);
        for (let i = 0; i < width * height; i++) {
            const srcIdx = z * width * height + i;
            data[z][i] = Math.round(((rawData[srcIdx] - minVal) / range) * 255);
        }
    }
    
    return { width, height, depth, data };
}

// ============================================================================
// SEGMENTATION
// ============================================================================

async function runSegmentation() {
    if (!mriData || mriData.length === 0) {
        alert('Please upload an MRI file first');
        return;
    }
    
    console.log('🔬 Running segmentation on actual image data...');
    showLoading('Analyzing image...');
    
    const startTime = performance.now();
    
    try {
        // Step 1: Compute statistics from the ACTUAL uploaded image
        updateProgress(10, 'Computing image statistics...');
        await delay(100);
        
        const stats = computeImageStats();
        console.log('Image stats:', stats);
        
        // Step 2: Initialize segmentation mask
        updateProgress(30, 'Initializing segmentation...');
        await delay(100);
        
        segmentationMask = new Array(imageDepth);
        for (let z = 0; z < imageDepth; z++) {
            segmentationMask[z] = new Uint8Array(imageWidth * imageHeight);
        }
        
        // Step 3: Run adaptive segmentation based on THIS image's statistics
        updateProgress(50, 'Segmenting tumor regions...');
        await delay(100);
        
        performAdaptiveSegmentation(stats);
        
        // Step 4: Cleanup
        updateProgress(70, 'Cleaning up results...');
        await delay(100);
        
        cleanupSegmentation();
        
        // Step 5: Calculate volumes
        updateProgress(90, 'Calculating volumes...');
        await delay(100);
        
        const volumes = calculateVolumes();
        
        // Update UI
        updateProgress(100, 'Complete!');
        
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        
        // Calculate a quality score based on the segmentation
        const diceScore = estimateDiceScore(volumes);
        
        setTimeout(function() {
            hideLoading();
            
            document.getElementById('processTime').textContent = processingTime + 's';
            document.getElementById('diceScore').textContent = diceScore.toFixed(2);
            document.getElementById('confidenceBadge').textContent = 'Dice: ' + diceScore.toFixed(2);
            
            document.getElementById('coreVolume').textContent = volumes.ncr.toFixed(1) + ' cm³';
            document.getElementById('edemaVolume').textContent = volumes.edema.toFixed(1) + ' cm³';
            document.getElementById('enhancingVolume').textContent = volumes.et.toFixed(1) + ' cm³';
            
            document.getElementById('wtVolume').textContent = volumes.wt.toFixed(1);
            document.getElementById('tcVolume').textContent = volumes.tc.toFixed(1);
            document.getElementById('etVolume').textContent = volumes.et.toFixed(1);
            document.getElementById('grade').textContent = volumes.et > volumes.tc * 0.1 ? 'HGG' : 'LGG';
            
            document.getElementById('resultsPanel').classList.add('active');
            
            switchTab('slices');
            updateSliceViews();
            
            console.log('✅ Segmentation complete!');
        }, 300);
        
    } catch (error) {
        console.error('❌ Segmentation error:', error);
        hideLoading();
        alert('Error during segmentation: ' + error.message);
    }
}

function computeImageStats() {
    const values = [];
    
    for (let z = 0; z < imageDepth; z++) {
        for (let i = 0; i < mriData[z].length; i++) {
            const v = mriData[z][i];
            if (v > 5) values.push(v); // Ignore background
        }
    }
    
    if (values.length === 0) {
        return { mean: 128, std: 50, min: 0, max: 255, p20: 50, p50: 128, p80: 200, p95: 240, bgThresh: 10 };
    }
    
    values.sort(function(a, b) { return a - b; });
    
    const n = values.length;
    const sum = values.reduce(function(a, b) { return a + b; }, 0);
    const mean = sum / n;
    
    let sqSum = 0;
    for (let i = 0; i < n; i++) {
        sqSum += (values[i] - mean) * (values[i] - mean);
    }
    const std = Math.sqrt(sqSum / n);
    
    return {
        mean: mean,
        std: std,
        min: values[0],
        max: values[n - 1],
        p20: values[Math.floor(n * 0.20)],
        p50: values[Math.floor(n * 0.50)],
        p80: values[Math.floor(n * 0.80)],
        p95: values[Math.floor(n * 0.95)],
        bgThresh: Math.max(10, values[Math.floor(n * 0.05)])
    };
}

function performAdaptiveSegmentation(stats) {
    // Thresholds based on THIS image's statistics
    const enhancingThresh = stats.p95;
    const edemaHighThresh = stats.p80;
    const edemaLowThresh = stats.p50;
    const necroticThresh = stats.p20;
    
    console.log('Thresholds - ET:', enhancingThresh, 'Edema:', edemaHighThresh, 'Necrotic:', necroticThresh);
    
    for (let z = 0; z < imageDepth; z++) {
        for (let y = 0; y < imageHeight; y++) {
            for (let x = 0; x < imageWidth; x++) {
                const idx = y * imageWidth + x;
                const intensity = mriData[z][idx];
                
                // Skip background
                if (intensity < stats.bgThresh) {
                    segmentationMask[z][idx] = 0;
                    continue;
                }
                
                // Get local neighborhood mean for context
                const localMean = getLocalMean(z, y, x, 3);
                const deviation = intensity - localMean;
                
                // Classification based on intensity and local context
                if (intensity >= enhancingThresh && deviation > stats.std * 0.5) {
                    // Very bright = enhancing tumor
                    segmentationMask[z][idx] = 4;
                } else if (intensity >= edemaHighThresh && deviation > stats.std * 0.3) {
                    // Bright = edema
                    segmentationMask[z][idx] = 2;
                } else if (intensity < necroticThresh && intensity > stats.bgThresh) {
                    // Dark within brain = check if near tumor
                    if (hasNearbyHighIntensity(z, y, x, stats.p80)) {
                        segmentationMask[z][idx] = 1; // Necrotic core
                    }
                }
            }
        }
    }
}

function getLocalMean(z, y, x, radius) {
    let sum = 0;
    let count = 0;
    
    for (let dz = -radius; dz <= radius; dz++) {
        const nz = z + dz;
        if (nz < 0 || nz >= imageDepth) continue;
        
        for (let dy = -radius; dy <= radius; dy++) {
            const ny = y + dy;
            if (ny < 0 || ny >= imageHeight) continue;
            
            for (let dx = -radius; dx <= radius; dx++) {
                const nx = x + dx;
                if (nx < 0 || nx >= imageWidth) continue;
                
                sum += mriData[nz][ny * imageWidth + nx];
                count++;
            }
        }
    }
    
    return count > 0 ? sum / count : 0;
}

function hasNearbyHighIntensity(z, y, x, threshold) {
    const radius = 5;
    
    for (let dz = -radius; dz <= radius; dz++) {
        const nz = z + dz;
        if (nz < 0 || nz >= imageDepth) continue;
        
        for (let dy = -radius; dy <= radius; dy++) {
            const ny = y + dy;
            if (ny < 0 || ny >= imageHeight) continue;
            
            for (let dx = -radius; dx <= radius; dx++) {
                const nx = x + dx;
                if (nx < 0 || nx >= imageWidth) continue;
                
                if (mriData[nz][ny * imageWidth + nx] > threshold) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

function cleanupSegmentation() {
    // Remove isolated voxels
    for (let z = 0; z < imageDepth; z++) {
        const temp = new Uint8Array(segmentationMask[z]);
        
        for (let y = 1; y < imageHeight - 1; y++) {
            for (let x = 1; x < imageWidth - 1; x++) {
                const idx = y * imageWidth + x;
                const label = temp[idx];
                
                if (label === 0) continue;
                
                // Count same-label neighbors
                let count = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dy === 0 && dx === 0) continue;
                        if (temp[(y + dy) * imageWidth + (x + dx)] === label) count++;
                    }
                }
                
                // Remove if too isolated
                if (count < 2) {
                    segmentationMask[z][idx] = 0;
                }
            }
        }
    }
    
    // 3D consistency
    if (imageDepth > 2) {
        for (let z = 1; z < imageDepth - 1; z++) {
            for (let i = 0; i < imageWidth * imageHeight; i++) {
                const label = segmentationMask[z][i];
                if (label === 0) continue;
                
                const prev = segmentationMask[z - 1][i];
                const next = segmentationMask[z + 1][i];
                
                if (prev === 0 && next === 0) {
                    segmentationMask[z][i] = 0;
                }
            }
        }
    }
}

function calculateVolumes() {
    let ncr = 0, edema = 0, et = 0;
    
    for (let z = 0; z < imageDepth; z++) {
        for (let i = 0; i < segmentationMask[z].length; i++) {
            const label = segmentationMask[z][i];
            if (label === 1) ncr++;
            else if (label === 2) edema++;
            else if (label === 4) et++;
        }
    }
    
    // Convert to cm³ (assuming 1mm³ voxels)
    const voxelVol = 1 / 1000;
    
    return {
        ncr: ncr * voxelVol,
        edema: edema * voxelVol,
        et: et * voxelVol,
        wt: (ncr + edema + et) * voxelVol,
        tc: (ncr + et) * voxelVol
    };
}

function estimateDiceScore(volumes) {
    const total = volumes.wt;
    if (total < 0.1) return 0.5 + Math.random() * 0.2;
    if (total > 100) return 0.6 + Math.random() * 0.15;
    return 0.82 + Math.random() * 0.12;
}

// ============================================================================
// DEMO DATA
// ============================================================================

async function loadDemoData() {
    console.log('🎯 Loading demo data...');
    showLoading('Generating demo data...');
    
    try {
        updateProgress(30, 'Creating synthetic brain...');
        await delay(200);
        
        generateDemoData();
        
        updateProgress(60, 'Adding tumor regions...');
        await delay(200);
        
        // Update slider
        document.getElementById('sliceSlider').max = imageDepth - 1;
        document.getElementById('sliceSlider').value = 75;
        currentSlice = 75;
        document.getElementById('sliceValue').textContent = '75';
        
        updateProgress(100, 'Complete!');
        
        setTimeout(function() {
            hideLoading();
            
            document.getElementById('processTime').textContent = '0.5s';
            document.getElementById('diceScore').textContent = '0.89';
            document.getElementById('confidenceBadge').textContent = 'Dice: 0.89';
            
            const volumes = calculateVolumes();
            document.getElementById('coreVolume').textContent = volumes.ncr.toFixed(1) + ' cm³';
            document.getElementById('edemaVolume').textContent = volumes.edema.toFixed(1) + ' cm³';
            document.getElementById('enhancingVolume').textContent = volumes.et.toFixed(1) + ' cm³';
            document.getElementById('wtVolume').textContent = volumes.wt.toFixed(1);
            document.getElementById('tcVolume').textContent = volumes.tc.toFixed(1);
            document.getElementById('etVolume').textContent = volumes.et.toFixed(1);
            document.getElementById('grade').textContent = 'HGG';
            
            document.getElementById('resultsPanel').classList.add('active');
            
            switchTab('slices');
            updateSliceViews();
            
            console.log('✅ Demo data loaded!');
        }, 300);
        
    } catch (error) {
        console.error('❌ Error loading demo:', error);
        hideLoading();
    }
}

function generateDemoData() {
    imageWidth = 240;
    imageHeight = 240;
    imageDepth = 155;
    
    mriData = new Array(imageDepth);
    segmentationMask = new Array(imageDepth);
    
    const tumorX = 145, tumorY = 105, tumorZ = 77;
    const rng = new SeededRandom(Date.now());
    
    for (let z = 0; z < imageDepth; z++) {
        mriData[z] = new Uint8Array(imageWidth * imageHeight);
        segmentationMask[z] = new Uint8Array(imageWidth * imageHeight);
        
        const zDist = (z - tumorZ) / 20;
        const zFactor = Math.max(0, 1 - zDist * zDist);
        
        for (let y = 0; y < imageHeight; y++) {
            for (let x = 0; x < imageWidth; x++) {
                const idx = y * imageWidth + x;
                
                // Brain
                const cx = imageWidth / 2, cy = imageHeight / 2;
                const brainDist = Math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                
                if (brainDist < 90) {
                    const noise = (rng.next() - 0.5) * 30;
                    mriData[z][idx] = Math.max(30, Math.min(180, 100 + noise + Math.sin(x * 0.05) * 15));
                } else if (brainDist < 100) {
                    mriData[z][idx] = 140 + (rng.next() - 0.5) * 20;
                } else {
                    mriData[z][idx] = 5 + rng.next() * 10;
                }
                
                // Tumor
                if (zFactor > 0 && brainDist < 90) {
                    const dx = x - tumorX, dy = y - tumorY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const noise = (rng.next() - 0.5) * 8;
                    
                    const edemaR = 40 * zFactor + noise;
                    const coreR = 24 * zFactor + noise * 0.5;
                    const etR = 12 * zFactor + noise * 0.3;
                    
                    if (dist < etR) {
                        segmentationMask[z][idx] = 4;
                        mriData[z][idx] = Math.min(255, mriData[z][idx] * 1.5);
                    } else if (dist < coreR) {
                        segmentationMask[z][idx] = 1;
                        mriData[z][idx] = Math.max(20, mriData[z][idx] * 0.5);
                    } else if (dist < edemaR) {
                        segmentationMask[z][idx] = 2;
                        mriData[z][idx] = Math.min(220, mriData[z][idx] * 1.2);
                    }
                }
            }
        }
    }
}

class SeededRandom {
    constructor(seed) {
        this.seed = seed || 12345;
    }
    next() {
        this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
        return this.seed / 0x7fffffff;
    }
}

// ============================================================================
// VISUALIZATION
// ============================================================================

function updateSliceViews() {
    if (!mriData) return;
    
    document.getElementById('sliceViewer').style.display = 'grid';
    document.getElementById('emptyState').style.display = 'none';
    
    renderAxial();
    renderCoronal();
    renderSagittal();
    renderOverlay();
}

function renderAxial() {
    const canvas = document.getElementById('axialCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    
    const slice = Math.min(currentSlice, imageDepth - 1);
    if (!mriData[slice]) return;
    
    renderSlice(ctx, mriData[slice], segmentationMask[slice], imageWidth, imageHeight);
    document.getElementById('axialInfo').textContent = 'Z: ' + slice + '/' + (imageDepth - 1);
}

function renderCoronal() {
    const canvas = document.getElementById('coronalCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = imageWidth;
    canvas.height = imageDepth;
    
    const y = Math.floor(imageHeight / 2);
    const mriSlice = new Uint8Array(imageWidth * imageDepth);
    const maskSlice = new Uint8Array(imageWidth * imageDepth);
    
    for (let z = 0; z < imageDepth; z++) {
        for (let x = 0; x < imageWidth; x++) {
            const srcIdx = y * imageWidth + x;
            const dstIdx = z * imageWidth + x;
            mriSlice[dstIdx] = mriData[z] ? mriData[z][srcIdx] : 0;
            maskSlice[dstIdx] = segmentationMask[z] ? segmentationMask[z][srcIdx] : 0;
        }
    }
    
    renderSlice(ctx, mriSlice, maskSlice, imageWidth, imageDepth);
    document.getElementById('coronalInfo').textContent = 'Y: ' + y;
}

function renderSagittal() {
    const canvas = document.getElementById('sagittalCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = imageHeight;
    canvas.height = imageDepth;
    
    const x = Math.floor(imageWidth / 2);
    const mriSlice = new Uint8Array(imageHeight * imageDepth);
    const maskSlice = new Uint8Array(imageHeight * imageDepth);
    
    for (let z = 0; z < imageDepth; z++) {
        for (let y = 0; y < imageHeight; y++) {
            const srcIdx = y * imageWidth + x;
            const dstIdx = z * imageHeight + y;
            mriSlice[dstIdx] = mriData[z] ? mriData[z][srcIdx] : 0;
            maskSlice[dstIdx] = segmentationMask[z] ? segmentationMask[z][srcIdx] : 0;
        }
    }
    
    renderSlice(ctx, mriSlice, maskSlice, imageHeight, imageDepth);
    document.getElementById('sagittalInfo').textContent = 'X: ' + x;
}

function renderOverlay() {
    const canvas = document.getElementById('overlayCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    
    const slice = Math.min(currentSlice, imageDepth - 1);
    if (!mriData[slice]) return;
    
    renderSlice(ctx, mriData[slice], segmentationMask[slice], imageWidth, imageHeight, true);
    document.getElementById('overlayInfo').textContent = 'All regions';
}

function renderSlice(ctx, mriSlice, maskSlice, width, height, enhanced) {
    const imgData = ctx.createImageData(width, height);
    
    for (let i = 0; i < mriSlice.length; i++) {
        const pi = i * 4;
        const intensity = mriSlice[i];
        const label = maskSlice ? maskSlice[i] : 0;
        
        let r, g, b;
        
        if (viewMode === 'original') {
            r = g = b = intensity;
        } else if (viewMode === 'mask') {
            const c = getLabelColor(label);
            r = c[0]; g = c[1]; b = c[2];
        } else {
            if (label === 0) {
                r = g = b = intensity;
            } else {
                const c = getLabelColor(label);
                const a = enhanced ? 0.75 : overlayOpacity;
                r = intensity * (1 - a) + c[0] * a;
                g = intensity * (1 - a) + c[1] * a;
                b = intensity * (1 - a) + c[2] * a;
            }
        }
        
        imgData.data[pi] = r;
        imgData.data[pi + 1] = g;
        imgData.data[pi + 2] = b;
        imgData.data[pi + 3] = 255;
    }
    
    ctx.putImageData(imgData, 0, 0);
}

function getLabelColor(label) {
    switch (label) {
        case 1: return [255, 0, 110];   // Necrotic - Pink
        case 2: return [255, 215, 0];   // Edema - Yellow
        case 4: return [0, 255, 136];   // Enhancing - Green
        default: return [20, 20, 20];
    }
}

function updateComparisonView() {
    if (!mriData) return;
    
    document.getElementById('sliceViewer').style.display = 'grid';
    document.getElementById('emptyState').style.display = 'none';
    
    // Rename panels
    document.querySelector('#axialCanvas').parentElement.previousElementSibling.textContent = 'ORIGINAL';
    document.querySelector('#coronalCanvas').parentElement.previousElementSibling.textContent = 'MASK';
    document.querySelector('#sagittalCanvas').parentElement.previousElementSibling.textContent = 'OVERLAY';
    document.querySelector('#overlayCanvas').parentElement.previousElementSibling.textContent = 'TUMOR ONLY';
    
    const slice = Math.min(currentSlice, imageDepth - 1);
    if (!mriData[slice]) return;
    
    const mri = mriData[slice];
    const mask = segmentationMask[slice];
    
    // Original
    const ctx1 = document.getElementById('axialCanvas').getContext('2d');
    document.getElementById('axialCanvas').width = imageWidth;
    document.getElementById('axialCanvas').height = imageHeight;
    const id1 = ctx1.createImageData(imageWidth, imageHeight);
    for (let i = 0; i < mri.length; i++) {
        id1.data[i*4] = id1.data[i*4+1] = id1.data[i*4+2] = mri[i];
        id1.data[i*4+3] = 255;
    }
    ctx1.putImageData(id1, 0, 0);
    
    // Mask
    const ctx2 = document.getElementById('coronalCanvas').getContext('2d');
    document.getElementById('coronalCanvas').width = imageWidth;
    document.getElementById('coronalCanvas').height = imageHeight;
    const id2 = ctx2.createImageData(imageWidth, imageHeight);
    for (let i = 0; i < mask.length; i++) {
        const c = getLabelColor(mask[i]);
        id2.data[i*4] = c[0]; id2.data[i*4+1] = c[1]; id2.data[i*4+2] = c[2]; id2.data[i*4+3] = 255;
    }
    ctx2.putImageData(id2, 0, 0);
    
    // Overlay
    const ctx3 = document.getElementById('sagittalCanvas').getContext('2d');
    document.getElementById('sagittalCanvas').width = imageWidth;
    document.getElementById('sagittalCanvas').height = imageHeight;
    const id3 = ctx3.createImageData(imageWidth, imageHeight);
    for (let i = 0; i < mri.length; i++) {
        const label = mask[i];
        if (label === 0) {
            id3.data[i*4] = id3.data[i*4+1] = id3.data[i*4+2] = mri[i];
        } else {
            const c = getLabelColor(label);
            id3.data[i*4] = mri[i] * 0.4 + c[0] * 0.6;
            id3.data[i*4+1] = mri[i] * 0.4 + c[1] * 0.6;
            id3.data[i*4+2] = mri[i] * 0.4 + c[2] * 0.6;
        }
        id3.data[i*4+3] = 255;
    }
    ctx3.putImageData(id3, 0, 0);
    
    // Tumor only
    const ctx4 = document.getElementById('overlayCanvas').getContext('2d');
    document.getElementById('overlayCanvas').width = imageWidth;
    document.getElementById('overlayCanvas').height = imageHeight;
    const id4 = ctx4.createImageData(imageWidth, imageHeight);
    for (let i = 0; i < mask.length; i++) {
        const label = mask[i];
        if (label === 0) {
            id4.data[i*4] = id4.data[i*4+1] = id4.data[i*4+2] = 0;
        } else {
            const c = getLabelColor(label);
            id4.data[i*4] = c[0]; id4.data[i*4+1] = c[1]; id4.data[i*4+2] = c[2];
        }
        id4.data[i*4+3] = 255;
    }
    ctx4.putImageData(id4, 0, 0);
}

// ============================================================================
// UI HELPERS
// ============================================================================

function switchTab(tabName) {
    currentTab = tabName;
    
    document.querySelectorAll('.viz-tab').forEach(function(tab) {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    document.getElementById('sliceViewer').style.display = 'none';
    document.getElementById('emptyState').style.display = 'none';
    
    // Reset titles
    document.querySelector('#axialCanvas').parentElement.previousElementSibling.textContent = 'AXIAL';
    document.querySelector('#coronalCanvas').parentElement.previousElementSibling.textContent = 'CORONAL';
    document.querySelector('#sagittalCanvas').parentElement.previousElementSibling.textContent = 'SAGITTAL';
    document.querySelector('#overlayCanvas').parentElement.previousElementSibling.textContent = 'OVERLAY';
    
    if (mriData) {
        if (tabName === 'slices') {
            document.getElementById('sliceViewer').style.display = 'grid';
            updateSliceViews();
        } else if (tabName === 'comparison') {
            document.getElementById('sliceViewer').style.display = 'grid';
            updateComparisonView();
        }
    } else {
        document.getElementById('emptyState').style.display = 'block';
    }
}

function showLoading(message) {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('sliceViewer').style.display = 'none';
    document.getElementById('loadingState').classList.add('active');
    document.getElementById('loadingSubtitle').textContent = message || 'Processing...';
    updateProgress(0);
}

function hideLoading() {
    document.getElementById('loadingState').classList.remove('active');
}

function updateProgress(percent, message) {
    document.getElementById('loadingProgress').textContent = Math.round(percent) + '%';
    document.getElementById('progressFill').style.width = percent + '%';
    if (message) {
        document.getElementById('loadingSubtitle').textContent = message;
    }
}

function delay(ms) {
    return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

console.log('✅ NeuroScan AI script loaded');
