// js/stateManager.js

const SAVE_KEY = 'chimeraDirectiveState_v6';
const starCount = 3072;
const distributionSize = 70;
let gameState = {};

function createDefaultState() {
    const neurons = [];
    for (let i = 0; i < starCount; i++) {
        const x = (Math.random() - 0.5) * distributionSize;
        const y = (Math.random() - 0.5) * distributionSize;
        const z = (Math.random() - 0.5) * distributionSize;
        neurons.push({ id: i, basePosition: { x, y, z }, currentPosition: { x, y, z } });
    }
    return {
        neurons: neurons,
        atlas: {
            explicit_resonance: {},
            explicit_inference: {},
            implicit_resonance: {},
            implicit_inference: {}
        },
        scannedBooks: {},
        operatorRank: 0,
        unlockedCodexContent: { 'welcome': 0 },
        viewedCodexContent: {},
        newCodexLayers: [],
        uiState: {
            atlasButton: 'disabled',
            protocolSwitch: 'locked',
            filters: { discovery: 'locked', gravity: 'locked', normalized: 'locked', syzygy: 'locked', orrery: 'locked', tags: 'locked' }
        }
    };
}

function loadState() {
    try {
        const savedState = localStorage.getItem(SAVE_KEY);
        if (savedState) {
            let parsedState = JSON.parse(savedState);
            // Migration for users coming from older save versions
            if (parsedState.autoScanner) {
                delete parsedState.autoScanner; // Remove obsolete scanner state
            }
            if (!parsedState.scannedBooks) {
                parsedState.scannedBooks = {}; // Add new state property if it doesn't exist
            }
            // Merge with a default state to ensure all necessary keys are present
            return { ...createDefaultState(), ...parsedState };
        } else {
            return createDefaultState();
        }
    } catch (error) {
        console.error("Failed to load or parse state, creating new state:", error);
        return createDefaultState();
    }
}

export function initState() {
    gameState = loadState();
}

export function getGameState() {
    return gameState;
}

export function saveState() {
    try {
        localStorage.setItem(SAVE_KEY, JSON.stringify(gameState));
    } catch (error) {
        console.error("Failed to save state:", error);
    }
}

export function calculateTotalDiscoveredSystems() {
    const state = getGameState();
    if (!state.atlas) return 0;
    const allKeys = Object.values(state.atlas).flatMap(quadrant => Object.keys(quadrant));
    return new Set(allKeys).size;
}

// --- NEW FUNCTION ---
export function purgeState() {
    localStorage.removeItem(SAVE_KEY);
}