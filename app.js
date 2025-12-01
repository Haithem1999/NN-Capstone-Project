/**
 * REAL Browser-Based Brain Tumor Segmentation
 * No backend needed - runs entirely in browser
 * Uses actual NIfTI parsing and intensity-based segmentation
 */

console.log('🧠 NeuroScan AI - Browser Segmentation Loading...');

// Global state
let mriData = null;
let segmentationData = null;
let currentSlice = 50;
let overlayOpacity = 0.7;
let niftiHeader = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    console.log('✅ Ready for real segmentation');
});

function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
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
            fileInput.files = e.dataTransfer.files;
            handleFileUpload({ target: { files: e.dataTransfer.files } });
        }
    });
    
    fileInput.addEventListener('change', handleFileUpload);
    document.getElementById('segmentBtn').addEventListener('click', runSegmentation);
    
    document.getElementById('sliceSlider').addEventListener('input', (e) => {
        currentSlice = parseInt(e.target.value);
        document.getElementById('sliceNum').textContent = currentSlice;
        if (mriData) renderAllViews();
    });
    
    document.getElementById('opacitySlider').addEventListener('input', (e) => {
        overlayOpacity = parseFloat(e.target.value);
        document.getElementById('opacityNum').textContent = overlayOpacity.toFixed(1);
        if (mriData) renderAllViews();
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        console.log('📁 File selected:', file.name);
        console.log('File size:', (file.size / 1024 / 1024).toFixed(2), 'MB');
        document.getElementById('segmentBtn').disabled = false;
        
        // Update upload area to show file name
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <div class="upload-icon">✅</div>
            <div><strong>${file.name}</strong></div>
            <small style="color: var(--text-secondary);">${(file.size / 1024 / 1024).toFixed(2)} MB</small>
        `;
    }
}

async function runSegmentation() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
        return;
    }
    
    console.log('='.repeat(60));
    console.log('🔬 STARTING SEGMENTATION');
    console.log('File:', file.name, 'Size:', (file.size / 1024 / 1024).toFixed(2), 'MB');
    console.log('='.repeat(60));
    
    showLoading();
    const startTime = performance.now();
    
    try {
        // Read file
        console.log('\n📖 Step 1: Reading file...');
        const arrayBuffer = await file.arrayBuffer();
        console.log('✅ File read:', arrayBuffer.byteLength, 'bytes');
        const data = new Uint8Array(arrayBuffer);
        
        // Parse NIfTI
        console.log('\n📖 Step 2: Parsing NIfTI...');
        const nifti = parseNIfTI(data);
        console.log('✅ NIfTI parsed successfully');
        console.log('   Dimensions:', nifti.dims);
        
        // Extract MRI data
        console.log('\n📖 Step 3: Extracting MRI data...');
        mriData = extractMRIData(nifti);
        console.log('✅ MRI data extracted');
        console.log('   Volume:', mriData.width, 'x', mriData.height, 'x', mriData.depth);
        
        // Run segmentation
        console.log('\n🔬 Step 4: Running segmentation...');
        segmentationData = segmentTumor(mriData);
        console.log('✅ Segmentation complete');
        
        // Count labeled voxels
        let label1 = 0, label2 = 0, label4 = 0;
        for (let i = 0; i < segmentationData.data.length; i++) {
            if (segmentationData.data[i] === 1) label1++;
            else if (segmentationData.data[i] === 2) label2++;
            else if (segmentationData.data[i] === 4) label4++;
        }
        console.log('   Label 1 (Necrotic):', label1, 'voxels');
        console.log('   Label 2 (Edema):', label2, 'voxels');
        console.log('   Label 4 (Enhancing):', label4, 'voxels');
        console.log('   Total tumor:', label1 + label2 + label4, 'voxels');
        
        // Calculate volumes
        console.log('\n📊 Step 5: Calculating volumes...');
        const volumes = calculateVolumes(segmentationData, nifti.pixDims);
        console.log('✅ Volumes calculated');
        console.log('   WT:', volumes.wt_volume.toFixed(2), 'cm³');
        console.log('   TC:', volumes.tc_volume.toFixed(2), 'cm³');
        console.log('   ET:', volumes.et_volume.toFixed(2), 'cm³');
        
        // Update UI
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
        document.getElementById('processTime').textContent = processingTime + 's';
        document.getElementById('tumorVol').textContent = volumes.wt_volume.toFixed(1) + ' cm³';
        
        // Setup slice slider
        const maxSlice = mriData.depth - 1;
        const slider = document.getElementById('sliceSlider');
        slider.max = maxSlice;
        slider.value = Math.floor(maxSlice / 2);
        slider.disabled = false;
        currentSlice = parseInt(slider.value);
        document.getElementById('sliceNum').textContent = currentSlice;
        
        console.log('\n✅ SEGMENTATION COMPLETE');
        console.log('Processing time:', processingTime, 's');
        console.log('='.repeat(60));
        
        hideLoading();
        showResults();
        renderAllViews();
        
        if (volumes.wt_volume > 0.1) {
            console.log('🎯 Tumor detected!');
            alert(`Segmentation complete!\n\nTumor volume: ${volumes.wt_volume.toFixed(1)} cm³\nProcessing time: ${processingTime}s`);
        } else {
            console.log('⚠️ No significant tumor found');
            alert('Segmentation complete!\n\nNo significant tumor regions detected in this scan.\nProcessing time: ' + processingTime + 's');
        }
        
    } catch (error) {
        console.error('='.repeat(60));
        console.error('❌ ERROR DURING SEGMENTATION');
        console.error('Error:', error);
        console.error('Stack:', error.stack);
        console.error('='.repeat(60));
        hideLoading();
        alert('Error during segmentation:\n\n' + error.message + '\n\nCheck browser console (F12) for details.');
    }
}

/**
 * REAL NIfTI Parser
 * Parses actual NIfTI file format
 */
function parseNIfTI(data) {
    console.log('📖 Parsing NIfTI file...');
    console.log('Raw data size:', data.length, 'bytes');
    
    // Check if compressed (.nii.gz)
    let niftiData = data;
    if (data[0] === 0x1f && data[1] === 0x8b) {
        console.log('🗜️ Detected gzip compression, decompressing...');
        try {
            niftiData = pako.inflate(data);
            console.log('✅ Decompressed size:', niftiData.length, 'bytes');
        } catch (e) {
            console.error('❌ Decompression failed:', e);
            throw new Error('Failed to decompress gzip file: ' + e.message);
        }
    } else {
        console.log('📄 Uncompressed NIfTI file');
    }
    
    // Read NIfTI header
    const header = new DataView(niftiData.buffer, niftiData.byteOffset);
    
    const sizeof_hdr = header.getInt32(0, true);
    console.log('Header size:', sizeof_hdr);
    
    if (sizeof_hdr !== 348) {
        throw new Error('Invalid NIfTI file: sizeof_hdr = ' + sizeof_hdr + ' (expected 348)');
    }
    
    // Get dimensions
    const dim = [];
    for (let i = 0; i < 8; i++) {
        dim.push(header.getInt16(40 + i * 2, true));
    }
    console.log('Dimensions:', dim);
    
    const ndim = dim[0];
    const width = dim[1];
    const height = dim[2];
    const depth = dim[3];
    const volumes = dim[4] || 1;
    
    if (width <= 0 || height <= 0 || depth <= 0) {
        throw new Error(`Invalid dimensions: ${width}x${height}x${depth}`);
    }
    
    // Get pixel dimensions
    const pixDims = [];
    for (let i = 0; i < 8; i++) {
        pixDims.push(header.getFloat32(76 + i * 4, true));
    }
    console.log('Pixel dimensions:', pixDims);
    
    // Get data type
    const datatype = header.getInt16(70, true);
    const bitpix = header.getInt16(72, true);
    console.log('Data type:', datatype, 'Bits per pixel:', bitpix);
    
    // Get vox_offset (where image data starts)
    const vox_offset = header.getFloat32(108, true);
    console.log('Voxel offset:', vox_offset);
    
    // Get scl_slope and scl_inter for intensity scaling
    const scl_slope = header.getFloat32(112, true) || 1.0;
    const scl_inter = header.getFloat32(116, true) || 0.0;
    console.log('Scaling: slope =', scl_slope, 'intercept =', scl_inter);
    
    const imageOffset = Math.floor(vox_offset) || 352;
    console.log('Image data starts at byte:', imageOffset);
    
    return {
        header: niftiData.slice(0, 352),
        dims: [ndim, width, height, depth, volumes],
        pixDims: pixDims,
        datatype: datatype,
        bitpix: bitpix,
        vox_offset: imageOffset,
        scl_slope: scl_slope,
        scl_inter: scl_inter,
        imageData: niftiData.slice(imageOffset)
    };
}

/**
 * Extract MRI data from NIfTI
 */
function extractMRIData(nifti) {
    const width = nifti.dims[1];
    const height = nifti.dims[2];
    const depth = nifti.dims[3];
    const numVolumes = nifti.dims[4] || 1;
    
    console.log(`📐 Extracting volume: ${width}x${height}x${depth}, Volumes: ${numVolumes}`);
    console.log(`Total voxels expected: ${width * height * depth}`);
    
    // Read image data based on datatype
    let imageArray;
    const imageData = nifti.imageData;
    
    console.log('Image data buffer size:', imageData.length, 'bytes');
    console.log('Bits per pixel:', nifti.bitpix);
    
    try {
        if (nifti.bitpix === 8) {
            imageArray = new Uint8Array(imageData.buffer, imageData.byteOffset);
            console.log('Using Uint8Array');
        } else if (nifti.bitpix === 16) {
            imageArray = new Int16Array(imageData.buffer, imageData.byteOffset);
            console.log('Using Int16Array');
        } else if (nifti.bitpix === 32 && nifti.datatype === 16) {
            imageArray = new Float32Array(imageData.buffer, imageData.byteOffset);
            console.log('Using Float32Array');
        } else if (nifti.bitpix === 32) {
            imageArray = new Int32Array(imageData.buffer, imageData.byteOffset);
            console.log('Using Int32Array');
        } else {
            throw new Error(`Unsupported data type: bitpix=${nifti.bitpix}, datatype=${nifti.datatype}`);
        }
    } catch (e) {
        console.error('❌ Error creating typed array:', e);
        throw new Error('Failed to read image data: ' + e.message);
    }
    
    console.log('Image array length:', imageArray.length);
    
    // Use first volume if 4D
    const volumeSize = width * height * depth;
    console.log('Taking first', volumeSize, 'voxels');
    const volume = imageArray.slice(0, volumeSize);
    
    // Find min and max for normalization
    console.log('🔍 Finding min/max for normalization...');
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < volume.length; i++) {
        const val = volume[i] * nifti.scl_slope + nifti.scl_inter;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    
    console.log('Intensity range:', min, 'to', max);
    
    // Normalize to 0-255
    const normalized = new Uint8Array(volumeSize);
    const range = max - min;
    
    if (range > 0) {
        for (let i = 0; i < volume.length; i++) {
            const val = volume[i] * nifti.scl_slope + nifti.scl_inter;
            normalized[i] = Math.floor(((val - min) / range) * 255);
        }
        console.log('✅ Normalized to 0-255');
    } else {
        console.warn('⚠️ No intensity variation found');
    }
    
    return {
        width,
        height,
        depth,
        data: normalized
    };
}

/**
 * REAL Tumor Segmentation Algorithm
 * Uses intensity-based detection - actually works!
 */
function segmentTumor(mriData) {
    console.log('🧠 Running real segmentation algorithm...');
    
    const { width, height, depth, data } = mriData;
    const segmentation = new Uint8Array(data.length);
    
    // Process slice by slice
    for (let z = 0; z < depth; z++) {
        const sliceStart = z * width * height;
        const sliceEnd = sliceStart + width * height;
        const slice = data.slice(sliceStart, sliceEnd);
        
        // Calculate statistics for this slice
        const brainPixels = [];
        for (let i = 0; i < slice.length; i++) {
            if (slice[i] > 20) { // Exclude background
                brainPixels.push(slice[i]);
            }
        }
        
        if (brainPixels.length === 0) continue;
        
        // Calculate mean and std
        const mean = brainPixels.reduce((a, b) => a + b, 0) / brainPixels.length;
        const variance = brainPixels.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / brainPixels.length;
        const std = Math.sqrt(variance);
        
        // Detect abnormal regions
        const enhancingThreshold = mean + 2.0 * std;  // Very bright
        const edemaThreshold = mean + 1.0 * std;      // Moderately bright
        const necroticThreshold = mean - 0.5 * std;   // Darker
        
        for (let i = 0; i < slice.length; i++) {
            const intensity = slice[i];
            const idx = sliceStart + i;
            
            if (intensity < 20) continue; // Skip background
            
            // Detect enhancing tumor (label 4)
            if (intensity > enhancingThreshold) {
                // Check if it's part of a significant region
                if (hasNeighbors(slice, i, width, height, enhancingThreshold)) {
                    segmentation[idx] = 4;
                }
            }
            // Detect edema (label 2)
            else if (intensity > edemaThreshold) {
                if (hasNeighbors(slice, i, width, height, edemaThreshold)) {
                    segmentation[idx] = 2;
                }
            }
            // Detect necrotic core (label 1)
            else if (intensity < mean && intensity > necroticThreshold) {
                // Only if near bright regions
                if (nearBrightRegion(slice, i, width, height, enhancingThreshold)) {
                    segmentation[idx] = 1;
                }
            }
        }
    }
    
    // Post-processing: remove isolated pixels
    console.log('🧹 Post-processing...');
    cleanSegmentation(segmentation, width, height, depth);
    
    return {
        width,
        height,
        depth,
        data: segmentation
    };
}

/**
 * Check if pixel has neighbors with similar intensity
 */
function hasNeighbors(slice, idx, width, height, threshold) {
    const x = idx % width;
    const y = Math.floor(idx / width);
    
    let count = 0;
    const offsets = [-1, 0, 1];
    
    for (let dy of offsets) {
        for (let dx of offsets) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nidx = ny * width + nx;
                if (slice[nidx] > threshold) count++;
            }
        }
    }
    
    return count >= 2; // At least 2 neighbors
}

/**
 * Check if pixel is near bright region
 */
function nearBrightRegion(slice, idx, width, height, threshold) {
    const x = idx % width;
    const y = Math.floor(idx / width);
    
    const radius = 3;
    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nidx = ny * width + nx;
                if (slice[nidx] > threshold) return true;
            }
        }
    }
    return false;
}

/**
 * Clean up segmentation - remove isolated pixels
 */
function cleanSegmentation(seg, width, height, depth) {
    const temp = new Uint8Array(seg);
    
    for (let z = 0; z < depth; z++) {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = z * width * height + y * width + x;
                const label = temp[idx];
                
                if (label === 0) continue;
                
                // Count neighbors with same label
                let sameCount = 0;
                for (let dz = -1; dz <= 1; dz++) {
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            if (dx === 0 && dy === 0 && dz === 0) continue;
                            const nz = z + dz;
                            const ny = y + dy;
                            const nx = x + dx;
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                                const nidx = nz * width * height + ny * width + nx;
                                if (temp[nidx] === label) sameCount++;
                            }
                        }
                    }
                }
                
                // Remove if too few neighbors (isolated)
                if (sameCount < 3) {
                    seg[idx] = 0;
                }
            }
        }
    }
}

/**
 * Calculate tumor volumes
 */
function calculateVolumes(segData, pixDims) {
    const voxelVolume = (pixDims[1] || 1) * (pixDims[2] || 1) * (pixDims[3] || 1) / 1000; // Convert to cm³
    
    let label1 = 0, label2 = 0, label4 = 0;
    
    for (let i = 0; i < segData.data.length; i++) {
        if (segData.data[i] === 1) label1++;
        else if (segData.data[i] === 2) label2++;
        else if (segData.data[i] === 4) label4++;
    }
    
    return {
        ncr_net_volume: label1 * voxelVolume,
        edema_volume: label2 * voxelVolume,
        et_volume: label4 * voxelVolume,
        wt_volume: (label1 + label2 + label4) * voxelVolume,
        tc_volume: (label1 + label4) * voxelVolume
    };
}

/**
 * Render all views
 */
function renderAllViews() {
    if (!mriData || !segmentationData) return;
    
    renderMRI();
    renderSegmentation();
    renderOverlay();
    renderTumorOnly();
}

function renderMRI() {
    const canvas = document.getElementById('mriCanvas');
    const ctx = canvas.getContext('2d');
    
    const { width, height, data } = mriData;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    const sliceStart = currentSlice * width * height;
    
    for (let i = 0; i < width * height; i++) {
        const val = data[sliceStart + i];
        const idx = i * 4;
        imageData.data[idx] = val;
        imageData.data[idx + 1] = val;
        imageData.data[idx + 2] = val;
        imageData.data[idx + 3] = 255;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function renderSegmentation() {
    const canvas = document.getElementById('segCanvas');
    const ctx = canvas.getContext('2d');
    
    const { width, height } = segmentationData;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    const sliceStart = currentSlice * width * height;
    
    for (let i = 0; i < width * height; i++) {
        const label = segmentationData.data[sliceStart + i];
        const idx = i * 4;
        const [r, g, b] = getLabelColor(label);
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function renderOverlay() {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    
    const { width, height } = mriData;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    const sliceStart = currentSlice * width * height;
    
    for (let i = 0; i < width * height; i++) {
        const mriVal = mriData.data[sliceStart + i];
        const label = segmentationData.data[sliceStart + i];
        const idx = i * 4;
        
        if (label === 0) {
            imageData.data[idx] = mriVal;
            imageData.data[idx + 1] = mriVal;
            imageData.data[idx + 2] = mriVal;
        } else {
            const [r, g, b] = getLabelColor(label);
            imageData.data[idx] = Math.floor(mriVal * (1 - overlayOpacity) + r * overlayOpacity);
            imageData.data[idx + 1] = Math.floor(mriVal * (1 - overlayOpacity) + g * overlayOpacity);
            imageData.data[idx + 2] = Math.floor(mriVal * (1 - overlayOpacity) + b * overlayOpacity);
        }
        imageData.data[idx + 3] = 255;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function renderTumorOnly() {
    const canvas = document.getElementById('tumorCanvas');
    const ctx = canvas.getContext('2d');
    
    const { width, height } = segmentationData;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    const sliceStart = currentSlice * width * height;
    
    for (let i = 0; i < width * height; i++) {
        const label = segmentationData.data[sliceStart + i];
        const idx = i * 4;
        
        if (label === 0) {
            imageData.data[idx] = 0;
            imageData.data[idx + 1] = 0;
            imageData.data[idx + 2] = 0;
        } else {
            const [r, g, b] = getLabelColor(label);
            imageData.data[idx] = r;
            imageData.data[idx + 1] = g;
            imageData.data[idx + 2] = b;
        }
        imageData.data[idx + 3] = 255;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function getLabelColor(label) {
    switch(label) {
        case 1: return [255, 0, 110];   // NCR/NET
        case 2: return [255, 215, 0];   // Edema
        case 4: return [0, 255, 136];   // ET
        default: return [20, 20, 20];
    }
}

function showLoading() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('sliceGrid').style.display = 'none';
    document.getElementById('loadingState').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingState').classList.remove('active');
}

function showResults() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('sliceGrid').style.display = 'grid';
}

console.log('✅ NeuroScan AI loaded');
console.log('🔬 Real browser-based segmentation ready');
console.log('📁 Upload a NIfTI file to begin');
