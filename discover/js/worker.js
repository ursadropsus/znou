// eve/js/worker.js

// --- CONFIGURATION MIRRORED FROM MAIN.JS (Iteration 16.1) ---
const distributionSize = 70;
const SYZYGY_LINE_COUNT = 10;
const MAX_HITS_FOR_GRAVITY = 10;
const MAX_GRAVITY_FACTOR = 0.2;

// --- STATE (Managed by the worker) ---
let basePositions = []; // The default, randomly scattered positions of all stars

// ==============================================================================
// --- CORE PHYSICS LOGIC (100% 1:1 RESTORATION from Iteration 16.1) ---
// This function is a direct, literal port of the positioning logic from
// the tick() loop in main.js (Iteration 16.1). No improvisations have been made.
// ==============================================================================

function calculateAllTargetPositions(hitCounts, filters) {
    let { isGravityViewActive, isNormalizedDensityActive, isOrreryViewActive } = filters;
    
    const targetPositions = new Float32Array(basePositions.length * 3);

    // --- SETUP LOGIC (LITERAL PORT) ---
    let normMinHits = 1;
    let normMaxHits = 1;
    let sortedSystems = [];
    const orrerySelection = {};

    // This block exactly mimics the original file's normalization setup,
    // including the self-deactivation logic if the data is invalid.
    if (isNormalizedDensityActive) {
        let discoveredSystems = [];
        for (let i = 0; i < hitCounts.length; i++) {
            if (hitCounts[i] > 0) discoveredSystems.push({ id: i, hits: hitCounts[i] });
        }
        
        let shouldDeactivate = false;
        if (discoveredSystems.length < 2) {
            shouldDeactivate = true;
        } else {
            const hits = discoveredSystems.map(s => s.hits);
            const min = Math.min(...hits);
            const max = Math.max(...hits);
            if (min === max) { // This check prevents division by zero
                shouldDeactivate = true;
            } else {
                normMinHits = min;
                normMaxHits = max;
            }
        }
        if (shouldDeactivate) {
            isNormalizedDensityActive = false; // Self-deactivates for this run
        }
    }
    
    // This block exactly mimics the original file's Syzygy/Orrery setup
    if (isOrreryViewActive) {
        let allDiscovered = [];
        for (let i = 0; i < hitCounts.length; i++) {
            if (hitCounts[i] > 0) allDiscovered.push({ id: i, hits: hitCounts[i] });
        }
        sortedSystems = allDiscovered.sort((a, b) => b.hits - a.hits);
        sortedSystems.slice(0, SYZYGY_LINE_COUNT + 1).forEach((entry, index) => {
            orrerySelection[entry.id] = { rank: index, hits: entry.hits };
        });
    }
    // --- END SETUP LOGIC ---


    // --- Main Calculation Loop (LITERAL PORT) ---
    for (let i = 0; i < basePositions.length; i++) {
        const basePos = basePositions[i];
        const hits = hitCounts[i] || 0;
        let targetX = basePos.x, targetY = basePos.y, targetZ = basePos.z;

        // The following if/else if chain is structured identically to Iteration 16.1
        if (isOrreryViewActive && orrerySelection[i]) {
            const selectionData = orrerySelection[i];
            const maxDistance = distributionSize / 2;
            if (selectionData.rank === 0) {
                targetX = 0; targetY = 0; targetZ = 0;
            } else {
                // This logic requires the min/max hits from the Normalization block,
                // which is why that block runs even if only Orrery is active.
                const proximityScore = (normMaxHits - selectionData.hits) / (normMaxHits - normMinHits);
                const radius = proximityScore * maxDistance;
                const angle = (selectionData.rank - 1) * (2 * Math.PI / SYZYGY_LINE_COUNT);
                targetX = radius * Math.cos(angle);
                targetY = radius * Math.sin(angle);
                targetZ = 0;
            }

        } else if (isNormalizedDensityActive) { // This will be false if shouldDeactivate was true
            const maxDistance = distributionSize / 2;
            const mag = Math.sqrt(basePos.x * basePos.x + basePos.y * basePos.y + basePos.z * basePos.z) || 1;
            const dirX = basePos.x / mag, dirY = basePos.y / mag, dirZ = basePos.z / mag;
            
            if (hits > 0) {
                const proximityScore = (normMaxHits - hits) / (normMaxHits - normHits);
                const newDistance = proximityScore * maxDistance;
                targetX = dirX * newDistance; 
                targetY = dirY * newDistance; 
                targetZ = dirZ * newDistance;
            } else {
                targetX = dirX * maxDistance;
                targetY = dirY * maxDistance;
                targetZ = dirZ * maxDistance;
            }

        } else if (isGravityViewActive) {
            const pullFactor = Math.min(hits / MAX_HITS_FOR_GRAVITY, 1.0);
            const gravityFactor = 1.0 - (1.0 - MAX_GRAVITY_FACTOR) * pullFactor;
            targetX *= gravityFactor; 
            targetY *= gravityFactor; 
            targetZ *= gravityFactor;
        }

        const offset = i * 3;
        targetPositions[offset] = targetX;
        targetPositions[offset + 1] = targetY;
        targetPositions[offset + 2] = targetZ;
    }
    return targetPositions;
}

// ==============================================================================
// --- MESSAGE HANDLING ---
// The worker listens for messages from main.js
// ==============================================================================
self.onmessage = function(e) {
    const { type, payload } = e.data;

    if (type === 'INIT') {
        basePositions = payload.basePositions;
    } else if (type === 'PROCESS') {
        const hitCounts = new Uint32Array(payload.hitCountsBuffer);
        
        // Pass a copy of filters to the calculation function, so we can
        // safely modify the `isNormalizedDensityActive` flag inside it
        // without affecting the main thread's state.
        const filtersCopy = { ...payload.filters };
        
        const targetPositions = calculateAllTargetPositions(hitCounts, filtersCopy);
        
        self.postMessage({
            type: 'COMPLETE',
            payload: {
                targetPositions: targetPositions.buffer
            }
        }, [targetPositions.buffer]);
    }
};