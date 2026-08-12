/* --------------------------------------------------- */
/* The Chimera Directive - Main Script                 */
/* Iteration 20.0: Final Polish & Pacing Update        */
/* --------------------------------------------------- */

// --- VIDEO MANAGER FRAMEWORK ---
const videoManager = {
    player: null,
    init: function() {
        this.player = document.getElementById('splash-video-player');
        if (!this.player) console.error("Video player element not found!");
    },
    play: function(src, options = {}) {
        if (!this.player) return;
        const { blendMode = 'screen', volume = 1.0, onEnded = () => {} } = options;
        this.player.src = src;
        this.player.style.mixBlendMode = blendMode;
        this.player.volume = volume;
        const onEndedHandler = () => {
            this.player.classList.add('hidden');
            this.player.removeEventListener('ended', onEndedHandler);
            onEnded();
        };
        this.player.addEventListener('ended', onEndedHandler);
        this.player.classList.remove('hidden');
        const playPromise = this.player.play();
        if (playPromise !== undefined) {
            playPromise.catch(error => {
                console.error("Video autoplay failed:", error);
                onEndedHandler();
            });
        }
    }
};

// --- CONFIGURATION ---
const API_URL = '/scan';
const CACHE_BASE_URL = '/data/';
const SAVE_KEY = 'chimeraDirectiveState_v5';

// DEV-MODE PACING: Lower ranks for easy testing. Raise for release.
const PROTOCOL_UNLOCK_RANK = 5;
const CACHE_UNLOCK_RANK = 3;
const FILTERS_UNLOCK_RANK = 2; // Groups all minor filters together

const LOGIN_USERNAMES = ['*z̸̡̡̢̨̛̯̼̭͎͚͉̠̜̘̰̜̠̾͌̿͂́̂̀̌͒͘͜*********','*****i̸̛̦̇̌̂̎͐͂̇͑̄̿̋̚͝͠*****','k̵̨̧̛̛͎̠̝̞̗̖͉̣͒̅̏̾̆̀͗̉̌̚̕**********','*******ȏ̵̢͕̥̤̄***','*********n̶̘̰̺̳͖̰͎͚͊͂͜*',];
const MAX_STREAM_LINES = 7;
const MAX_CALIBRATION_LOG_LINES = 4;
const MAX_INPUT_LOG_LINES = 10;
const MAX_LIVE_LOG_LINES = 15;
const SCANNER_SPEED_TIERS = [ { name: '1x', multiplier: 1, unlockReq: 0, animates: true }, { name: '5x', multiplier: 5, unlockReq: 10, animates: false }, { name: '25x', multiplier: 25, unlockReq: 20, animates: false }, { name: 'MAX', multiplier: 100, unlockReq: 40, animates: false }];
const SYZYGY_LINE_COUNT = 10;
const ORBIT_HOT_COLOR = new THREE.Color('#ffee00');
const ORBIT_COLD_COLOR = new THREE.Color('#00ffff');

// --- SETUP (SCENE, ETC.) ---
const scene = new THREE.Scene();
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 100;
const camera = new THREE.OrthographicCamera(frustumSize * aspect / -2, frustumSize * aspect / 2, frustumSize / 2, frustumSize / -2, 1, 1000);
const canvas = document.querySelector('canvas.webgl');
const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
const composer = new THREE.EffectComposer(renderer);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
function createPixelTexture() { const canvas = document.createElement('canvas'); canvas.width = 16; canvas.height = 16; const context = canvas.getContext('2d'); context.fillStyle = 'white'; context.fillRect(0, 0, 16, 16); return new THREE.CanvasTexture(canvas); }
const pixelTexture = createPixelTexture();
pixelTexture.minFilter = THREE.NearestFilter;
pixelTexture.magFilter = THREE.NearestFilter;
const starCount = 3072;
const starGeometry = new THREE.BufferGeometry();
const positions = new Float32Array(starCount * 3);
const colors = new Float32Array(starCount * 3);
const baseColor = new THREE.Color('#3a7a9f'); const activeColor = new THREE.Color('#ffffff'); const discoveredColor = new THREE.Color('#00ffff'); const undiscoveredColor = new THREE.Color('#ff4100'); const distributionSize = 70;
const starMaterial = new THREE.PointsMaterial({ size: 0.5, sizeAttenuation: true, vertexColors: true, map: pixelTexture, alphaTest: 0.5, transparent: true });
const starfield = new THREE.Points(starGeometry, starMaterial);
const boxGeometry = new THREE.BoxGeometry(distributionSize, distributionSize, distributionSize); const boxEdges = new THREE.EdgesGeometry(boxGeometry); const boxMaterial = new THREE.LineBasicMaterial({ color: 0x1a3a4f }); const boxLines = new THREE.LineSegments(boxEdges, boxMaterial);
const tracerGroup = new THREE.Group(); const tracerHistory = []; const tracerLength = 10; for (let i = 0; i < tracerLength; i++) { const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3)); const material = new THREE.PointsMaterial({ color: activeColor, size: Math.max(0.1, 1.5 - (i * 0.2)), sizeAttenuation: true, map: pixelTexture, alphaTest: 0.5, transparent: true, opacity: 1.0 - (i / tracerLength), blending: THREE.AdditiveBlending, }); const particle = new THREE.Points(geometry, material); tracerGroup.add(particle); }
tracerGroup.visible = false;
const systemLabelsGroup = new THREE.Group();
const syzygyLineGroup = new THREE.Group();
const orreryRingGroup = new THREE.Group();

// --- HTML ELEMENT REFERENCES ---
const uiContainer = document.querySelector('.ui-container');
const viewportBorder = document.querySelector('.viewport-border');
const neuronLabelElement = document.getElementById('neuron-label'); const terminalElement = document.querySelector('.terminal'); const terminalInput = document.querySelector('.terminal-input'); const terminalOutput = document.querySelector('.terminal-output');
const operatorRankValue = document.getElementById('operator-rank-value');
const protocolSelector = document.getElementById('protocol-selector');
const btnProtoExplicit = document.getElementById('btn-proto-explicit');
const btnProtoImplicit = document.getElementById('btn-proto-implicit');
const modeSelector = document.getElementById('mode-selector'); 
const btnModeResonance = document.getElementById('btn-mode-resonance');
const btnModeInference = document.getElementById('btn-mode-inference');
const filterStatusToastElement = document.getElementById('filter-status-toast');
const filterMenuTrigger = document.getElementById('filter-menu-trigger');
const btnFilterDiscovery = document.getElementById('btn-filter-discovery');
const btnFilterDensity = document.getElementById('btn-filter-density');
const btnFilterNormalizedDensity = document.getElementById('btn-filter-normalized-density');
const btnFilterSyzygy = document.getElementById('btn-filter-syzygy');
const btnFilterOrrery = document.getElementById('btn-filter-orrery');
const btnToggleSystemTags = document.getElementById('btn-toggle-system-tags');
const filterMenuContainer = document.getElementById('filter-menu-container');
const filterMenuMain = document.getElementById('filter-menu-main');
const btnOpenAtlas = document.getElementById('btn-open-atlas');
const atlasOverlay = document.getElementById('atlas-overlay'); const atlasList = document.getElementById('atlas-list'); const btnAtlasClose = document.getElementById('btn-atlas-close'); const btnPurgeData = document.getElementById('btn-purge-data');
const atlasDiscoveryCounter = document.getElementById('atlas-discovery-counter');
const atlasQuadrantCounter = document.getElementById('atlas-quadrant-counter');
const btnAtlasPrev = document.getElementById('btn-atlas-prev');
const btnAtlasNext = document.getElementById('btn-atlas-next');
const operatorRankDisplay = document.getElementById('operator-rank-display');
const atlasSearchInput = document.getElementById('atlas-search-input');
const atlasSortSelect = document.getElementById('atlas-sort-select');
const dataStreamVisualizer = document.getElementById('data-stream-visualizer');
const dataStreamContent = document.getElementById('data-stream-content');
const cacheAnalysisUnit = document.getElementById('cache-analysis-unit'); const cacheSelect = document.getElementById('cache-select'); const btnScannerToggle = document.getElementById('btn-scanner-toggle');
const scannerStatusText = document.getElementById('scanner-status-text'); const scannerProgressText = document.getElementById('scanner-progress-text'); const scannerProgressBar = document.getElementById('scanner-progress-bar');
const scannerSpeedControls = document.getElementById('scanner-speed-controls');
const calibrationBridge = document.getElementById('calibration-bridge');
const calibrationLog = document.getElementById('calibration-log');
const inputHistoryLog = document.getElementById('input-history-log');
const liveLogFeed = document.getElementById('live-log-feed');
const calibrationVideoPlayerA = document.getElementById('calibration-video-a');
const calibrationVideoPlayerB = document.getElementById('calibration-video-b');
const btnToggleCodex = document.getElementById('btn-toggle-codex');
const codexContainer = document.getElementById('codex-container');
const codexTabs = document.getElementById('codex-tabs');
const codexList = document.getElementById('codex-list');
const codexVideoPlayer = document.getElementById('codex-video-player');
const codexVideoHeader = document.getElementById('codex-video-header');
const codexDescription = document.getElementById('codex-description');
// --- New: Exchange and Data Parcels
const parcelCountElement = document.getElementById('parcel-count');
const parcelProgressDotsElement = document.getElementById('parcel-progress-dots');
const gotoExchangeBtn = document.getElementById('goto-exchange-btn');

// --- STATE MANAGEMENT ---
let lastKnownParcelCount = 0;
let gameState = {};
let currentProtocol = 'implicit';
let currentMapMode = 'resonance';
const quadrantOrder = ['implicit_resonance', 'implicit_inference', 'explicit_resonance', 'explicit_inference'];
let isAnimating = false; let animationProgress = 0; const animationDuration = 0.35; const startPosition = new THREE.Vector3(); const endPosition = new THREE.Vector3(); let lastActiveStarIndex = -1;
let activeLabelStarIndex = -1; let labelVisibilityTimeout = null; let toastVisibilityTimeout = null;
let isFilterViewActive = false; let isGravityViewActive = false; let isTerminalFocused = true;
let isNormalizedDensityActive = false;
let isSyzygyFilterActive = false;
let isOrreryViewActive = false;
let normMinHits = 1;
let normMaxHits = 1;
let scannerTimeout = null; let loadedCaches = {};
let hideMenuTimeout = null;
let dataStreamLines = [];
let cameraTargetPosition = null;
let highlightedSystem = null;
let areSystemTagsVisible = false;
let systemLabels = {};
let currentPlaneQuaternion = new THREE.Quaternion();
let targetPlaneQuaternion = new THREE.Quaternion();
let orreryRotationTimer = null;
let isCodexOpen = false;
let activeCodexLayer = 'eve';
let lastSelectedEntryId = null;
const clock = new THREE.Clock();
let flashColorIndex = 0;
let lastFlashTime = 0;

// --- CALIBRATION SYSTEM ---
let calibrationProgress = 0;
let activeVideoPlayer = 'a';
const CALIBRATION_COMPLETE_COUNT = 5;
const CALIBRATION_VIDEO_FILES = [ 'assets/video/calibration_0.mp4', 'assets/video/calibration_1.mp4', 'assets/video/calibration_2.mp4', 'assets/video/calibration_3.mp4', 'assets/video/calibration_4.mp4', 'assets/video/calibration_5.mp4' ];
const calibrationVideoCache = {};
async function preloadAllCalibrationVideos() {
    const promises = CALIBRATION_VIDEO_FILES.map(async (path) => {
        if (!calibrationVideoCache[path]) {
            try {
                const response = await fetch(path);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const videoBlob = await response.blob();
                calibrationVideoCache[path] = URL.createObjectURL(videoBlob);
            } catch (error) {
                console.error(`Failed to preload video ${path}:`, error);
                calibrationVideoCache[path] = path;
            }
        }
    });
    await Promise.all(promises);
}
function updateCalibrationLog(text) {
    const currentLatest = calibrationLog.querySelector('.latest');
    if (currentLatest) currentLatest.classList.remove('latest');
    const newLogEntry = document.createElement('p');
    newLogEntry.textContent = text;
    newLogEntry.classList.add('latest');
    calibrationLog.appendChild(newLogEntry);
    while (calibrationLog.children.length > MAX_CALIBRATION_LOG_LINES) {
        calibrationLog.firstChild.remove();
    }
}
function updateInputHistoryLog(prompt, neuronId) {
    const currentLatest = inputHistoryLog.querySelector('.latest');
    if (currentLatest) {
        currentLatest.classList.remove('latest');
    }
    const newLogText = `> ${prompt}\n  [J5-${neuronId}]`;
    const newLogEntry = document.createElement('p');
    newLogEntry.textContent = newLogText;
    newLogEntry.classList.add('latest');
    inputHistoryLog.appendChild(newLogEntry);
    while (inputHistoryLog.children.length > MAX_INPUT_LOG_LINES) {
        inputHistoryLog.firstChild.remove();
    }
}
function handleCalibrationVideoEnd(event) {
    if (!event.target.loop) {
        setTimeout(() => {
            calibrationBridge.classList.add('is-fading-out');
        }, 1000);
    }
}
function triggerCalibrationSequence(newProgress) {
    const currentPlayer = activeVideoPlayer === 'a' ? calibrationVideoPlayerA : calibrationVideoPlayerB;
    const nextPlayer = activeVideoPlayer === 'a' ? calibrationVideoPlayerB : calibrationVideoPlayerA;

    // Fade out the old player and activate the new one
    currentPlayer.classList.remove('is-active-player');
    nextPlayer.classList.add('is-active-player');
    nextPlayer.play().catch(e => console.error("Calibration video play failed:", e));

    // Toggle the state for the *next* call
    activeVideoPlayer = activeVideoPlayer === 'a' ? 'b' : 'a';

    // Preload the next video into the player that just finished
    const nextVideoIndex = newProgress + 1;
    if (nextVideoIndex < CALIBRATION_VIDEO_FILES.length) {
        const nextVideoPath = CALIBRATION_VIDEO_FILES[nextVideoIndex];
        currentPlayer.src = calibrationVideoCache[nextVideoPath] || nextVideoPath;
        currentPlayer.loop = nextVideoIndex < CALIBRATION_COMPLETE_COUNT;
        currentPlayer.load();
    }
}
async function initializeCalibrationUI() {
    const discoveredCount = calculateTotalDiscoveredSystems();
    calibrationProgress = discoveredCount;
    if (discoveredCount < CALIBRATION_COMPLETE_COUNT) {
        calibrationBridge.classList.remove('hidden');
        await preloadAllCalibrationVideos();
        calibrationVideoPlayerA.addEventListener('ended', handleCalibrationVideoEnd);
        calibrationVideoPlayerB.addEventListener('ended', handleCalibrationVideoEnd);
        calibrationVideoPlayerA.src = calibrationVideoCache[CALIBRATION_VIDEO_FILES[calibrationProgress]];
        calibrationVideoPlayerA.loop = calibrationProgress < CALIBRATION_COMPLETE_COUNT;
        calibrationVideoPlayerA.load();
        calibrationVideoPlayerA.play();
        calibrationVideoPlayerA.classList.add('is-active-player');
        if (calibrationProgress + 1 < CALIBRATION_VIDEO_FILES.length) {
            const nextVideoPath = CALIBRATION_VIDEO_FILES[calibrationProgress + 1];
            calibrationVideoPlayerB.src = calibrationVideoCache[nextVideoPath];
            calibrationVideoPlayerB.loop = (calibrationProgress + 1) < CALIBRATION_COMPLETE_COUNT;
            calibrationVideoPlayerB.load();
        }
        updateCalibrationLog("Calibration channel open.\nProvide initial signal.");
    }
}

// --- 3D LABEL SYSTEM ---
function createSystemLabel(systemId, position) {
    if (systemLabels[systemId]) return;
    const text = `J5-${systemId}`;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const fontSize = 24;
    context.font = `${fontSize}px 'Roboto Mono', monospace`;
    const textWidth = context.measureText(text).width;
    canvas.width = textWidth + 10;
    canvas.height = fontSize + 10;
    context.font = `${fontSize}px 'Roboto Mono', monospace`;
    context.fillStyle = 'rgba(167, 216, 255, 0.8)';
    context.fillText(text, 5, fontSize);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.8 });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(position.x, position.y, position.z);
    sprite.visible = areSystemTagsVisible;
    systemLabels[systemId] = sprite;
    systemLabelsGroup.add(sprite);
}

// --- CORE STATE & SAVE/LOAD FUNCTIONS ---
function calculateTotalDiscoveredSystems() {
    if (!gameState.atlas) return 0;
    const allKeys = [
        ...Object.keys(gameState.atlas.explicit_resonance || {}),
        ...Object.keys(gameState.atlas.explicit_inference || {}),
        ...Object.keys(gameState.atlas.implicit_resonance || {}),
        ...Object.keys(gameState.atlas.implicit_inference || {})
    ];
    const allDiscovered = new Set(allKeys);
    return allDiscovered.size;
}

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
        autoScanner: {
            unlocked: false,
            isRunning: false,
            activeCache: 'the_sea_implicit_resonance.json',
            activeSpeed: '1x',
            caches: [
                { filename: "the_sea_implicit_resonance.json", displayName: "Cache: Implicit Resonance", lineProcessed: 0, totalLines: 0, protocol: 'implicit', mode: 'resonance' },
                { filename: "the_sea_implicit_inference.json", displayName: "Cache: Implicit Inference", lineProcessed: 0, totalLines: 0, protocol: 'implicit', mode: 'inference' },
                { filename: "the_sea_explicit_resonance.json", displayName: "Cache: Explicit Resonance", lineProcessed: 0, totalLines: 0, protocol: 'explicit', mode: 'resonance' },
                { filename: "the_sea_explicit_inference.json", displayName: "Cache: Explicit Inference", lineProcessed: 0, totalLines: 0, protocol: 'explicit', mode: 'inference' }
            ]
        },
        operatorRank: 0,
        unclaimedEvents: [],
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

function saveState() { try { localStorage.setItem(SAVE_KEY, JSON.stringify(gameState)); } catch (error) { console.error("Failed to save state:", error); } }

function loadState() {
    try {
        const savedState = localStorage.getItem(SAVE_KEY);
        if (savedState) {
            let parsedState = JSON.parse(savedState);
            if (parsedState.atlas && !parsedState.atlas.explicit_resonance) {
                console.log("Old save file detected. Migrating to 4-quadrant structure...");
                const oldAtlasResonance = parsedState.atlas.resonance || parsedState.atlas || {};
                const oldAtlasInference = parsedState.atlas.inference || {};
                parsedState.atlas = {
                    explicit_resonance: oldAtlasResonance,
                    explicit_inference: oldAtlasInference,
                    implicit_resonance: {},
                    implicit_inference: {}
                };
            }
            gameState = parsedState;
            if (!gameState.autoScanner || !Array.isArray(gameState.autoScanner.caches) || gameState.autoScanner.caches.length < 4) gameState.autoScanner = createDefaultState().autoScanner;
            if (!gameState.autoScanner.activeSpeed) gameState.autoScanner.activeSpeed = '1x';
            if (typeof gameState.operatorRank !== 'number') gameState.operatorRank = calculateTotalDiscoveredSystems();
            if (!gameState.unlockedCodexContent) gameState.unlockedCodexContent = { 'welcome': 0 };
            if (!gameState.viewedCodexContent) gameState.viewedCodexContent = {};
            if (!gameState.newCodexLayers) gameState.newCodexLayers = [];
            if (!gameState.uiState || !gameState.uiState.filters) gameState.uiState = createDefaultState().uiState;
            if (!gameState.uiState.protocolSwitch) gameState.uiState.protocolSwitch = 'locked';
        } else {
            gameState = createDefaultState();
        }
    } catch (error) {
        console.error("Failed to load or parse state, creating new state:", error);
        gameState = createDefaultState();
    }
    for (let i = 0; i < starCount; i++) { const neuron = gameState.neurons[i]; positions[i * 3] = neuron.basePosition.x; positions[i * 3 + 1] = neuron.basePosition.y; positions[i * 3 + 2] = neuron.basePosition.z; }
}

// --- API COMMUNICATION ---
async function pingApiForNeuron(prompt) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, protocol: currentProtocol, mode: currentMapMode })
        });
        if (!response.ok) {
            console.error(`API Error: Server responded with status ${response.status}`);
            return null;
        }
        const data = await response.json();
        if (typeof data.neuron_id !== 'number') {
            console.error('API Error: Invalid response format from server.', data);
            return null;
        }
        return data;
    } catch (error) {
        console.error('API Call Failed:', error);
        return null;
    }
}

// --- UI & THEME LOGIC ---
function getActiveAtlasKey() { return `${currentProtocol}_${currentMapMode}`; }

function updateGlobalTheme() {
    uiContainer.classList.remove('theme-imp-res', 'theme-imp-inf', 'theme-exp-res', 'theme-exp-inf');
    uiContainer.classList.add(`theme-${currentProtocol.slice(0,3)}-${currentMapMode.slice(0,3)}`);
}

function updateStarColors() {
    const activeAtlas = gameState.atlas[getActiveAtlasKey()];
    for (let i = 0; i < starCount; i++) {
        const isDiscovered = activeAtlas && activeAtlas[i] !== undefined;
        const targetColor = isFilterViewActive ? (isDiscovered ? discoveredColor : undiscoveredColor) : baseColor;
        colors[i * 3] = targetColor.r;
        colors[i * 3 + 1] = targetColor.g;
        colors[i * 3 + 2] = targetColor.b;
    }
    if (highlightedSystem && highlightedSystem.id !== null) {
        colors[highlightedSystem.id * 3] = activeColor.r;
        colors[highlightedSystem.id * 3 + 1] = activeColor.g;
        colors[highlightedSystem.id * 3 + 2] = activeColor.b;
    }
    starGeometry.attributes.color.needsUpdate = true;
}
function showFilterToast(message) { if (toastVisibilityTimeout) clearTimeout(toastVisibilityTimeout); filterStatusToastElement.innerText = message; filterStatusToastElement.classList.add('visible'); toastVisibilityTimeout = setTimeout(() => { filterStatusToastElement.classList.remove('visible'); }, 2000); }

// --- ATLAS LOGIC ---
function renderAtlas() {
    atlasList.innerHTML = '';
    const activeAtlas = gameState.atlas[getActiveAtlasKey()];
    const searchTerm = atlasSearchInput.value.toLowerCase();
    const sortMethod = atlasSortSelect.value;
    let systemIds = activeAtlas ? Object.keys(activeAtlas) : [];

    if (atlasQuadrantCounter) {
        atlasQuadrantCounter.innerText = `VIEW DISCOVERED: ${systemIds.length}`;
    }
    
    if (searchTerm) {
        systemIds = systemIds.filter(id => {
            const data = activeAtlas[id];
            if (!data) return false;
            return data.discoveryPrompt.toLowerCase().includes(searchTerm) || data.resonances.some(r => r.toLowerCase().includes(searchTerm));
        });
    }

    systemIds.sort((a, b) => {
        const dataA = activeAtlas[a];
        const dataB = activeAtlas[b];
        switch (sortMethod) {
            case 'id-desc': return parseInt(b) - parseInt(a);
            case 'hits-asc': return (dataA?.hits || 0) - (dataB?.hits || 0);
            case 'hits-desc': return (dataB?.hits || 0) - (dataA?.hits || 0);
            default: return parseInt(a) - parseInt(b);
        }
    });

    const totalDiscovered = calculateTotalDiscoveredSystems();
    gameState.operatorRank = totalDiscovered;
    atlasDiscoveryCounter.innerText = `TOTAL SYSTEMS DISCOVERED: ${totalDiscovered} / ${starCount}`;
    operatorRankDisplay.innerText = `OPERATOR RANK: ${gameState.operatorRank}`;
    const atlasHeaderTitle = atlasOverlay.querySelector('h1');
    atlasHeaderTitle.innerText = `${currentProtocol.toUpperCase()} // ${currentMapMode.toUpperCase()} ATLAS`;
    
    if (systemIds.length === 0) {
        atlasList.innerHTML = `<p>No systems discovered in this atlas view.</p>`;
        return;
    }
    let maxHits = 0;
    systemIds.forEach(id => { if ((activeAtlas[id]?.hits || 0) > maxHits) maxHits = activeAtlas[id].hits; });
    if (maxHits === 0) maxHits = 1;

    for (const id of systemIds) {
        const data = activeAtlas[id];
        if (!data) continue;
        const heatLevel = Math.min(5, Math.ceil((data.hits / maxHits) * 5) || 1);
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.className = `heatmap-tier-${heatLevel}`;
        summary.innerHTML = `<span>J5-${id} (${data.hits} Hits)</span><button class="btn-atlas-goto" data-system-id="${id}">Focus</button>`;
        const content = document.createElement('div');
        content.className = 'atlas-entry-content';
        content.innerHTML = `<p><strong>Discovery Prompt:</strong> "${data.discoveryPrompt}"</p><p><strong>All Inputs:</strong></p><ul>${data.resonances.map(r => `<li>"${r}"</li>`).join('')}</ul>`;
        details.append(summary, content);
        atlasList.appendChild(details);
    }
}

// --- CODEX LOGIC ---
function openCodex() {
    isCodexOpen = true;
    codexContainer.classList.add('active');
    btnToggleCodex.classList.add('active');
    gameState.newCodexLayers = [];
    btnToggleCodex.classList.remove('is-flashing-green', 'is-flashing-purple', 'is-flashing-orange');
    populateCodexList(activeCodexLayer);
    if (!lastSelectedEntryId) {
        const firstButton = codexList.querySelector('.codex-list-btn:not(.is-locked)');
        if (firstButton) {
            handleCodexSelection(firstButton.dataset.id, activeCodexLayer);
        }
    }
}
function closeCodex() {
    isCodexOpen = false;
    codexContainer.classList.remove('active');
    btnToggleCodex.classList.remove('active');
    codexVideoPlayer.pause();
}
function populateCodexList(layer) {
    codexList.innerHTML = '';
    if (typeof CODEX_DATA === 'undefined' || !CODEX_DATA[layer]) {
        codexList.innerHTML = '<p>Codex data not found.</p>';
        return;
    }
    CODEX_DATA[layer].forEach(entry => {
        const button = document.createElement('button');
        button.className = 'codex-list-btn';
        button.dataset.id = entry.id;
        const highestUnlockedBlock = gameState.unlockedCodexContent[entry.id];
        if (highestUnlockedBlock !== undefined) {
            button.innerText = entry.title;
            if ((gameState.viewedCodexContent[entry.id] ?? -1) < highestUnlockedBlock) {
                button.classList.add('is-new');
            }
        } else {
            button.innerText = entry.title.replace(/[a-zA-Z0-9]/g, '█');
            button.classList.add('is-locked');
            button.disabled = true;
        }
        codexList.appendChild(button);
    });
}
function handleCodexSelection(codexId, layer) {
    const entryData = CODEX_DATA[layer]?.find(e => e.id === codexId);
    if (!entryData) return;
    lastSelectedEntryId = codexId;
    codexVideoHeader.innerText = entryData.title;
    codexVideoPlayer.src = entryData.videoSrc;
    codexVideoPlayer.loop = true;
    codexVideoPlayer.play().catch(e => console.error("Codex video playback failed.", e));
    const highestUnlockedBlock = gameState.unlockedCodexContent[codexId];
    let fullDescriptionHTML = '';
    if (highestUnlockedBlock !== undefined) {
        for (let i = 0; i <= highestUnlockedBlock; i++) {
            fullDescriptionHTML += `<p>${entryData.contentBlocks[i].description}</p>`;
        }
    }
    codexDescription.innerHTML = fullDescriptionHTML;
    gameState.viewedCodexContent[codexId] = highestUnlockedBlock;
    codexList.querySelectorAll('.codex-list-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.id === codexId);
        if (btn.dataset.id === codexId) {
            btn.classList.remove('is-new');
        }
    });
    saveState();
}

// --- AUTO-SCANNER & BURST SPEED LOGIC ---
function populateSpeedButtons() { scannerSpeedControls.innerHTML = ''; SCANNER_SPEED_TIERS.forEach(tier => { const button = document.createElement('button'); button.innerText = tier.name; button.dataset.speed = tier.name; button.className = 'speed-control-btn'; scannerSpeedControls.appendChild(button); }); }
function updateScannerSpeedUI() { const totalDiscovered = calculateTotalDiscoveredSystems(); scannerSpeedControls.querySelectorAll('.speed-control-btn').forEach(button => { const speed = button.dataset.speed; const tier = SCANNER_SPEED_TIERS.find(t => t.name === speed); button.disabled = totalDiscovered < tier.unlockReq; button.classList.toggle('active', speed === gameState.autoScanner.activeSpeed); }); }
function updateDataStreamVisualizer() { dataStreamContent.innerHTML = dataStreamLines.map(line => `<p>${line}</p>`).join(''); }
function checkUnlock() {
    const totalDiscovered = calculateTotalDiscoveredSystems();
    if (gameState.uiState.atlasButton === 'disabled' && totalDiscovered >= 1) {
        gameState.uiState.atlasButton = 'enabled';
    }
    if (gameState.autoScanner.unlocked) {
        cacheAnalysisUnit.classList.remove('hidden');
    } else if (totalDiscovered >= CACHE_UNLOCK_RANK) {
        gameState.autoScanner.unlocked = true;
        cacheAnalysisUnit.classList.remove('hidden');
        showFilterToast("CACHE ANALYSIS UNIT: [ONLINE]");
    }
}
async function runScannerStep() {
    if (!gameState.autoScanner.isRunning) {
        scannerStatusText.innerText = 'PAUSED';
        btnScannerToggle.innerText = 'RESUME ANALYSIS';
        return;
    }
    const speedTier = SCANNER_SPEED_TIERS.find(t => t.name === gameState.autoScanner.activeSpeed);
    const batchSize = speedTier.multiplier;
    const cacheName = gameState.autoScanner.activeCache;
    if (!loadedCaches[cacheName]) {
        try {
            const response = await fetch(`${CACHE_BASE_URL}${cacheName}`);
            loadedCaches[cacheName] = await response.json();
            const cacheState = gameState.autoScanner.caches.find(c => c.filename === cacheName);
            if (cacheState) cacheState.totalLines = loadedCaches[cacheName].length;
        } catch (error) {
            console.error(`Failed to load cache ${cacheName}:`, error);
            scannerStatusText.innerText = 'LOAD ERROR';
            gameState.autoScanner.isRunning = false;
            return;
        }
    }
    const cache = loadedCaches[cacheName];
    const cacheState = gameState.autoScanner.caches.find(c => c.filename === cacheName);
    if (!cacheState) {
        console.error("Could not find cache state for", cacheName);
        gameState.autoScanner.isRunning = false;
        return;
    }
    for (let i = 0; i < batchSize; i++) {
        if (cacheState.lineProcessed >= cacheState.totalLines) {
            scannerStatusText.innerText = 'EXPENDED';
            btnScannerToggle.innerText = 'ANALYSIS COMPLETE';
            btnScannerToggle.disabled = true;
            gameState.autoScanner.isRunning = false;
            dataStreamVisualizer.classList.add('hidden');
            updateScannerUI();
            saveState();
            return;
        }
        const entry = cache[cacheState.lineProcessed];
        dataStreamLines.push(entry.sentence);
        if (dataStreamLines.length > MAX_STREAM_LINES) dataStreamLines.shift();
        processResonanceSilent(entry.neuron_id, entry.sentence, cacheState.protocol, cacheState.mode);
        cacheState.lineProcessed++;
    }
    updateDataStreamVisualizer();
    updateScannerUI();
    updateScannerSpeedUI();
    saveState();
    scannerTimeout = setTimeout(runScannerStep, 0);
}
function updateScannerUI() {
    const cacheState = gameState.autoScanner.caches.find(c => c.filename === gameState.autoScanner.activeCache);
    if (!cacheState) return;
    const progress = (cacheState.totalLines > 0) ? (cacheState.lineProcessed / cacheState.totalLines) * 100 : 0;
    scannerProgressText.innerText = `${cacheState.lineProcessed} / ${cacheState.totalLines}`;
    scannerProgressBar.style.width = `${progress}%`;
}

// --- INPUT HANDLING & BUTTONS ---
function switchProtocol(newProtocol) {
    if (newProtocol === currentProtocol) return;
    currentProtocol = newProtocol;
    updateSelectorsAndCache();
    showFilterToast(`${newProtocol.toUpperCase()} PROTOCOL ACTIVE`);
}
function switchMapMode(newMode) {
    if (newMode === currentMapMode) return;
    currentMapMode = newMode;
    updateSelectorsAndCache();
    showFilterToast(`${newMode.toUpperCase()} MODE ACTIVE`);
}

function updateSelectorsAndCache() {
    btnProtoExplicit.classList.toggle('active', currentProtocol === 'explicit');
    btnProtoImplicit.classList.toggle('active', currentProtocol === 'implicit');
    btnModeResonance.classList.toggle('active', currentMapMode === 'resonance');
    btnModeInference.classList.toggle('active', currentMapMode === 'inference');
    
    const targetCache = gameState.autoScanner.caches.find(c => c.protocol === currentProtocol && c.mode === currentMapMode);
    if (targetCache) {
        cacheSelect.value = targetCache.filename;
        // Trigger change event to update scanner UI
        cacheSelect.dispatchEvent(new Event('change'));
    }
    
    updateStarColors();
    updateGlobalTheme();
}

btnProtoExplicit?.addEventListener('click', () => switchProtocol('explicit'));
btnProtoImplicit?.addEventListener('click', () => switchProtocol('implicit'));
btnModeResonance?.addEventListener('click', () => switchMapMode('resonance'));
btnModeInference?.addEventListener('click', () => switchMapMode('inference'));
btnFilterDiscovery.addEventListener('click', () => { isFilterViewActive = !isFilterViewActive; btnFilterDiscovery.classList.toggle('active', isFilterViewActive); updateStarColors(); showFilterToast(`DISCOVERY FILTER: [ ${isFilterViewActive ? 'ACTIVE' : 'INACTIVE'} ]`); });
btnFilterDensity.addEventListener('click', () => { isGravityViewActive = !isGravityViewActive; if (isGravityViewActive) { isNormalizedDensityActive = false; btnFilterNormalizedDensity.classList.remove('active'); } btnFilterDensity.classList.toggle('active', isGravityViewActive); showFilterToast(`GRAVITY DENSITY: [ ${isGravityViewActive ? 'ACTIVE' : 'INACTIVE'} ]`); });
btnFilterNormalizedDensity.addEventListener('click', () => { isNormalizedDensityActive = !isNormalizedDensityActive; if (isNormalizedDensityActive) { isGravityViewActive = false; btnFilterDensity.classList.remove('active'); } btnFilterNormalizedDensity.classList.toggle('active', isNormalizedDensityActive); showFilterToast(`NORMALIZED DENSITY: [ ${isNormalizedDensityActive ? 'ACTIVE' : 'INACTIVE'} ]`); });
btnFilterSyzygy.addEventListener('click', () => { isSyzygyFilterActive = !isSyzygyFilterActive; if (isSyzygyFilterActive) { isOrreryViewActive = false; btnFilterOrrery.classList.remove('active'); if (!isNormalizedDensityActive) { isNormalizedDensityActive = true; btnFilterNormalizedDensity.classList.add('active'); if (isGravityViewActive) { isGravityViewActive = false; btnFilterDensity.classList.remove('active'); } } } btnFilterSyzygy.classList.toggle('active', isSyzygyFilterActive); showFilterToast(`SYZYGY VIEW: [ ${isSyzygyFilterActive ? 'ACTIVE' : 'INACTIVE'} ]`); });
btnFilterOrrery.addEventListener('click', () => { isOrreryViewActive = !isOrreryViewActive; btnFilterOrrery.classList.toggle('active', isOrreryViewActive); if (isOrreryViewActive) { isSyzygyFilterActive = false; btnFilterSyzygy.classList.remove('active'); if (!isNormalizedDensityActive) { isNormalizedDensityActive = true; btnFilterNormalizedDensity.classList.add('active'); if (isGravityViewActive) { isGravityViewActive = false; btnFilterDensity.classList.remove('active'); } } setNewOrreryTargetRotation(); orreryRotationTimer = setInterval(setNewOrreryTargetRotation, 11000); } else { if (orreryRotationTimer) clearInterval(orreryRotationTimer); } showFilterToast(`ORRERY VIEW: [ ${isOrreryViewActive ? 'ACTIVE' : 'INACTIVE'} ]`); });
btnToggleSystemTags.addEventListener('click', () => { areSystemTagsVisible = !areSystemTagsVisible; btnToggleSystemTags.classList.toggle('active', areSystemTagsVisible); showFilterToast(`SYSTEM ID TAGS: [ ${areSystemTagsVisible ? 'ACTIVE' : 'INACTIVE'} ]`); if (areSystemTagsVisible) { const allKnownSystems = new Set(Object.keys(gameState.atlas).flatMap(key => Object.keys(gameState.atlas[key]))); allKnownSystems.forEach(id => { if (!systemLabels[id]) createSystemLabel(id, gameState.neurons[id].basePosition); }); } for (const id in systemLabels) { systemLabels[id].visible = areSystemTagsVisible; } });
filterMenuContainer.addEventListener('mouseenter', () => { if (hideMenuTimeout) clearTimeout(hideMenuTimeout); filterMenuMain.classList.remove('hidden'); });
filterMenuContainer.addEventListener('mouseleave', () => { hideMenuTimeout = setTimeout(() => { filterMenuMain.classList.add('hidden'); }, 300); });
btnOpenAtlas.addEventListener('click', () => { renderAtlas(); atlasOverlay.classList.remove('hidden'); });
btnAtlasClose.addEventListener('click', () => { atlasOverlay.classList.add('hidden'); });
btnPurgeData.addEventListener('click', () => { if (confirm("Are you sure you want to permanently erase all discovery data? This cannot be undone.")) { localStorage.removeItem(SAVE_KEY); location.reload(); } });
atlasSearchInput.addEventListener('input', renderAtlas);
atlasSortSelect.addEventListener('change', renderAtlas);
atlasList.addEventListener('click', (event) => { if (event.target.matches('.btn-atlas-goto')) { focusOnSystem(parseInt(event.target.dataset.systemId)); } });
btnAtlasPrev?.addEventListener('click', () => { const currentIndex = quadrantOrder.indexOf(getActiveAtlasKey()); const nextIndex = (currentIndex - 1 + quadrantOrder.length) % quadrantOrder.length; const [protocol, mode] = quadrantOrder[nextIndex].split('_'); switchProtocol(protocol); switchMapMode(mode); renderAtlas(); });
btnAtlasNext?.addEventListener('click', () => { const currentIndex = quadrantOrder.indexOf(getActiveAtlasKey()); const nextIndex = (currentIndex + 1) % quadrantOrder.length; const [protocol, mode] = quadrantOrder[nextIndex].split('_'); switchProtocol(protocol); switchMapMode(mode); renderAtlas(); });
scannerSpeedControls.addEventListener('click', (event) => { if (event.target.matches('.speed-control-btn') && !event.target.disabled) { gameState.autoScanner.activeSpeed = event.target.dataset.speed; updateScannerSpeedUI(); saveState(); } });
btnScannerToggle.addEventListener('click', () => {
    gameState.autoScanner.isRunning = !gameState.autoScanner.isRunning;
    if (gameState.autoScanner.isRunning) {
        scannerStatusText.innerText = 'RUNNING';
        btnScannerToggle.innerText = 'PAUSE ANALYSIS';
        dataStreamVisualizer.classList.remove('hidden');
        runScannerStep();
    } else {
        scannerStatusText.innerText = 'PAUSED';
        btnScannerToggle.innerText = 'RESUME ANALYSIS';
        dataStreamVisualizer.classList.add('hidden');
        if (scannerTimeout) clearTimeout(scannerTimeout);
    }
    saveState();
});
cacheSelect.addEventListener('change', () => {
    const selectedCache = gameState.autoScanner.caches.find(c => c.filename === cacheSelect.value);
    if (selectedCache) {
        currentProtocol = selectedCache.protocol;
        currentMapMode = selectedCache.mode;
        updateSelectorsAndCache(); // Syncs top-left buttons
    }
    // BUG FIX: Re-enable the button if the newly selected cache is not exhausted.
    if (selectedCache && selectedCache.lineProcessed < selectedCache.totalLines) {
        btnScannerToggle.disabled = false;
        btnScannerToggle.innerText = 'START ANALYSIS';
        scannerStatusText.innerText = 'IDLE';
    }
    gameState.autoScanner.activeCache = cacheSelect.value;
    updateScannerUI();
    saveState();
});
btnToggleCodex.addEventListener('click', () => { if (isCodexOpen) closeCodex(); else openCodex(); });
codexList.addEventListener('click', (event) => { const targetButton = event.target.closest('.codex-list-btn'); if (targetButton && !targetButton.disabled) handleCodexSelection(targetButton.dataset.id, activeCodexLayer); });
codexContainer.addEventListener('click', (event) => { if (event.target === codexContainer) closeCodex(); });
codexTabs.addEventListener('click', (event) => { const targetButton = event.target.closest('.codex-tab-btn'); if (targetButton && targetButton.dataset.layer) { const newLayer = targetButton.dataset.layer; if (newLayer === activeCodexLayer) return; codexContainer.classList.remove('theme-ursa', 'theme-interpretability'); if (newLayer !== 'eve') codexContainer.classList.add(`theme-${newLayer}`); activeCodexLayer = newLayer; codexTabs.querySelectorAll('.codex-tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.layer === newLayer)); populateCodexList(newLayer); if (lastSelectedEntryId) handleCodexSelection(lastSelectedEntryId, newLayer); } });

// --- TERMINAL INTERACTION ---
terminalInput.addEventListener('keydown', async (event) => { if (event.key === 'Enter' && !isAnimating) { const command = terminalInput.value; if (command) { terminalInput.value = ''; await triggerResonanceEffect(command); } } });
terminalElement.addEventListener('animationend', () => { terminalElement.classList.remove('is-flashing'); });

// --- CORE LOGIC & RESONANCE ---
function checkForUnlocks(neuronId) {
    const totalDiscovered = calculateTotalDiscoveredSystems();
    gameState.operatorRank = totalDiscovered;
    const activeAtlasKey = getActiveAtlasKey();
    const neuronHits = gameState.atlas[activeAtlasKey]?.[neuronId]?.hits || 0;
    CODEX_DATA.eve.forEach(entry => {
        const currentUnlockedBlock = gameState.unlockedCodexContent[entry.id] ?? -1;
        const nextBlockIndex = currentUnlockedBlock + 1;
        if (entry.contentBlocks[nextBlockIndex]) {
            const conditions = entry.contentBlocks[nextBlockIndex].unlockConditions;
            let conditionMet = false;
            switch (conditions.type) {
                case 'initial': conditionMet = true; break;
                case 'rank': if (gameState.operatorRank >= conditions.value) conditionMet = true; break;
                case 'repeatHit': if (neuronHits >= conditions.count) conditionMet = true; break;
            }
            if (conditionMet) {
                gameState.unlockedCodexContent[entry.id] = nextBlockIndex;
                ['eve', 'ursa', 'interpretability'].forEach(layer => {
                    if (!gameState.newCodexLayers.includes(layer)) gameState.newCodexLayers.push(layer);
                });
            }
        }
    });

    if (totalDiscovered >= FILTERS_UNLOCK_RANK) {
        if (gameState.uiState.filters.discovery === 'locked') { gameState.uiState.filters.discovery = 'unlocked'; showFilterToast("FILTER UNLOCKED: DISCOVERY"); }
        if (gameState.uiState.filters.gravity === 'locked') { gameState.uiState.filters.gravity = 'unlocked'; showFilterToast("FILTER UNLOCKED: GRAVITY DENSITY"); }
        if (gameState.uiState.filters.tags === 'locked') { gameState.uiState.filters.tags = 'unlocked'; showFilterToast("FILTER UNLOCKED: SYSTEM ID TAGS"); }
        if (gameState.uiState.filters.normalized === 'locked') { gameState.uiState.filters.normalized = 'unlocked'; showFilterToast("FILTER UNLOCKED: NORMALIZED DENSITY"); }
        if (gameState.uiState.filters.syzygy === 'locked') { gameState.uiState.filters.syzygy = 'unlocked'; showFilterToast("FILTER UNLOCKED: SYZYGY VIEW"); }
        if (gameState.uiState.filters.orrery === 'locked') { gameState.uiState.filters.orrery = 'unlocked'; showFilterToast("FILTER UNLOCKED: ORRERY VIEW"); }
    }
    
    if (totalDiscovered >= PROTOCOL_UNLOCK_RANK) {
        if (gameState.uiState.protocolSwitch === 'locked') {
            gameState.uiState.protocolSwitch = 'unlocked';
            showFilterToast("NEW PROTOCOL AVAILABLE: EXPLICIT");
        }
    }
    updateUIStates();
}
function updateUIStates() {
    btnOpenAtlas.disabled = (gameState.uiState.atlasButton === 'disabled' && calculateTotalDiscoveredSystems() < 1);

    const isProtocolLocked = gameState.uiState.protocolSwitch === 'locked';
    if(protocolSelector) protocolSelector.style.display = isProtocolLocked ? 'none' : 'flex';
    if(btnProtoExplicit) {
        btnProtoExplicit.disabled = isProtocolLocked;
        btnProtoExplicit.title = isProtocolLocked ? `Increase Operator Rank to ${PROTOCOL_UNLOCK_RANK} to unlock` : '';
    }

    Object.entries({ discovery: btnFilterDiscovery, gravity: btnFilterDensity, normalized: btnFilterNormalizedDensity, syzygy: btnFilterSyzygy, orrery: btnFilterOrrery, tags: btnToggleSystemTags }).forEach(([filterName, buttonElement]) => {
        const isLocked = gameState.operatorRank < FILTERS_UNLOCK_RANK;
        buttonElement.classList.toggle('is-locked', isLocked);
        buttonElement.disabled = isLocked;
    });
}
function processResonanceSilent(targetStarIndex, prompt, protocol, mode) {
    if (targetStarIndex === null || targetStarIndex >= starCount) return;
    const atlasKey = `${protocol}_${mode}`;
    if (!gameState.atlas[atlasKey]) return;
    const wasAlreadyDiscoveredOnAnyMap = Object.values(gameState.atlas).some(atlas => atlas[targetStarIndex]);
    if (!wasAlreadyDiscoveredOnAnyMap) {
        gameState.operatorRank++;
        operatorRankValue.innerText = gameState.operatorRank;
        operatorRankValue.classList.add('is-flashing');
        operatorRankValue.addEventListener('animationend', () => operatorRankValue.classList.remove('is-flashing'), { once: true });
        if (calibrationProgress < CALIBRATION_COMPLETE_COUNT) {
            calibrationProgress++;
            triggerCalibrationSequence(calibrationProgress);
            switch (calibrationProgress) {
                case 1: updateCalibrationLog("Anchor acquired. \nCalibration progressing: 1/5."); break;
                case 2: updateCalibrationLog("Trajectory widening. \nCalibration: 2/5."); break;
                case 3: updateCalibrationLog("Lattice responding. \nCalibration: 3/5."); break;
                case 4: updateCalibrationLog("Alignment stabilizing. \nCalibration: 4/5."); break;
                case 5: updateCalibrationLog("Calibration complete. \nAccess level expanded."); break;
            }
        }
    }
    const activeAtlas = gameState.atlas[atlasKey];
    const wasDiscoveredOnThisMap = activeAtlas[targetStarIndex] !== undefined;
    if (!activeAtlas[targetStarIndex]) {
        activeAtlas[targetStarIndex] = { hits: 0, discoveryPrompt: prompt, resonances: [] };
        if (areSystemTagsVisible) createSystemLabel(targetStarIndex, gameState.neurons[targetStarIndex].basePosition);
    }
    activeAtlas[targetStarIndex].hits++;
    if (!activeAtlas[targetStarIndex].resonances.includes(prompt)) activeAtlas[targetStarIndex].resonances.push(prompt);
    if (isFilterViewActive && !wasDiscoveredOnThisMap && atlasKey === getActiveAtlasKey()) updateStarColors();
    checkForUnlocks(targetStarIndex);
    checkUnlock();
    updateParcelUI();
}
async function processResonanceVisual(targetStarIndex, prompt) {
    terminalOutput.textContent = `...Connection established. Firing tracer...`;
    processResonanceSilent(targetStarIndex, prompt, currentProtocol, currentMapMode);
    const targetNeuron = gameState.neurons[targetStarIndex];
    animationProgress = 0;
    tracerHistory.length = 0;
    lastActiveStarIndex = targetStarIndex;
    startPosition.set(0, 0, 0);
    const endPosVec = new THREE.Vector3(targetNeuron.basePosition.x, targetNeuron.basePosition.y, targetNeuron.basePosition.z);
    endPosition.copy(endPosVec);
    tracerGroup.children.forEach(child => child.position.copy(startPosition));
    tracerGroup.visible = true;
    await new Promise(resolve => setTimeout(resolve, 50));
    saveState();
}
async function triggerResonanceEffect(prompt) {
    isAnimating = true; terminalInput.disabled = true; terminalElement.classList.add('is-flashing');
    terminalOutput.textContent = `PROBING [${currentProtocol.slice(0,3).toUpperCase()}/${currentMapMode.slice(0,3).toUpperCase()}] WITH: "${prompt}"...`;

    const responseData = await pingApiForNeuron(prompt); // <-- RENAME VARIABLE

    if (responseData === null) { // <-- UPDATE CHECK
        terminalOutput.textContent = `[ERROR] Connection failed.`;
        terminalInput.disabled = false; isAnimating = false; terminalInput.focus(); return;
    }

    const targetStarIndex = responseData.neuron_id; // <-- GET NEURON_ID FROM RESPONSE
    const eventId = responseData.event_id; // <-- GET EVENT_ID FROM RESPONSE

    // --- NEW LOGIC TO SAVE THE EVENT ID ---
    if (eventId) {
    if (!gameState.unclaimedEvents) {
        gameState.unclaimedEvents = [];
    }
    gameState.unclaimedEvents.push(eventId);
    // We will save state inside processResonanceVisual as before
}
    updateInputHistoryLog(prompt, targetStarIndex);
    await processResonanceVisual(targetStarIndex, prompt);
}

// --- "GO TO" FUNCTIONALITY ---
function focusOnSystem(systemId) {
    if (systemId === null || !gameState.neurons[systemId]) return;
    atlasOverlay.classList.add('hidden');
    const targetNeuron = gameState.neurons[systemId];
    cameraTargetPosition = new THREE.Vector3(targetNeuron.currentPosition.x, targetNeuron.currentPosition.y, targetNeuron.currentPosition.z);
    if (highlightedSystem?.timeout) {
        clearTimeout(highlightedSystem.timeout);
    }
    highlightedSystem = {
        id: systemId,
        timeout: setTimeout(() => {
            highlightedSystem = null;
            updateStarColors();
        }, 2500)
    };
    updateStarColors();
}

// --- HANDLE WINDOW RESIZING ---
window.addEventListener('resize', () => {
    const aspect = window.innerWidth / window.innerHeight;
    camera.left = frustumSize * aspect / -2;
    camera.right = frustumSize * aspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = frustumSize / -2;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// --- GLOBAL KEYBOARD LISTENER ---
window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        if (isCodexOpen) {
            closeCodex();
        } else if (!atlasOverlay.classList.contains('hidden')) {
            atlasOverlay.classList.add('hidden');
        }
    }
});

// --- ANIMATION LOOP ---
const tick = () => {
    const deltaTime = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();
    controls.update();

    if (gameState.newCodexLayers.length > 0 && !isCodexOpen) {
        if (elapsedTime - lastFlashTime > 0.75) {
            lastFlashTime = elapsedTime;
            flashColorIndex = (flashColorIndex + 1) % gameState.newCodexLayers.length;
            btnToggleCodex.classList.remove('is-flashing-green', 'is-flashing-purple', 'is-flashing-orange');
            const layerToFlash = gameState.newCodexLayers[flashColorIndex];
            if (layerToFlash === 'eve') btnToggleCodex.classList.add('is-flashing-green');
            else if (layerToFlash === 'interpretability') btnToggleCodex.classList.add('is-flashing-purple');
            else if (layerToFlash === 'ursa') btnToggleCodex.classList.add('is-flashing-orange');
        }
    } else {
        btnToggleCodex.classList.remove('is-flashing-green', 'is-flashing-purple', 'is-flashing-orange');
    }

    const activeAtlas = gameState.atlas[getActiveAtlasKey()];
    if (isNormalizedDensityActive) {
        const discoveredSystems = activeAtlas ? Object.values(activeAtlas) : [];
        let shouldDeactivate = false;
        if (discoveredSystems.length < 2) {
            shouldDeactivate = true;
        } else {
            const hits = discoveredSystems.map(s => s.hits);
            const min = Math.min(...hits);
            const max = Math.max(...hits);
            if (min === max) {
                shouldDeactivate = true;
            } else {
                normMinHits = min;
                normMaxHits = max;
            }
        }
        if (shouldDeactivate) {
            isNormalizedDensityActive = false;
            btnFilterNormalizedDensity.classList.remove('active');
        }
    }
    if (isOrreryViewActive) {
        currentPlaneQuaternion.slerp(targetPlaneQuaternion, 0.02);
        orreryRingGroup.quaternion.copy(currentPlaneQuaternion);
    }
    let sortedSystems = [];
    const orrerySelection = {};
    if ((isSyzygyFilterActive || isOrreryViewActive) && activeAtlas) {
        sortedSystems = Object.entries(activeAtlas).sort((a, b) => b[1].hits - a[1].hits);
        if (isOrreryViewActive) {
            sortedSystems.slice(0, SYZYGY_LINE_COUNT + 1).forEach((entry, index) => {
                orrerySelection[entry[0]] = { rank: index, hits: entry[1].hits };
            });
        }
    }
    if (isSyzygyFilterActive) {
        syzygyLineGroup.visible = sortedSystems.length >= 2;
        if (syzygyLineGroup.visible) {
            const sunPosition = gameState.neurons[sortedSystems[0][0]].currentPosition;
            const opacity = (Math.sin(elapsedTime * 3) + 1) / 2 * 0.6 + 0.4;
            for (let i = 0; i < SYZYGY_LINE_COUNT; i++) {
                const line = syzygyLineGroup.children[i];
                line.material.opacity = opacity;
                if (sortedSystems[i + 1]) {
                    line.visible = true;
                    const planetPosition = gameState.neurons[sortedSystems[i + 1][0]].currentPosition;
                    const positions = line.geometry.attributes.position;
                    positions.setXYZ(0, sunPosition.x, sunPosition.y, sunPosition.z);
                    positions.setXYZ(1, planetPosition.x, planetPosition.y, planetPosition.z);
                    positions.needsUpdate = true;
                } else {
                    line.visible = false;
                }
            }
        }
    } else {
        syzygyLineGroup.visible = false;
    }
    for (let i = 0; i < starCount; i++) {
        const neuron = gameState.neurons[i];
        const atlasEntry = activeAtlas ? activeAtlas[i] : null;
        const targetPosition = new THREE.Vector3();
        if (isOrreryViewActive && orrerySelection[i]) {
            const selectionData = orrerySelection[i];
            const maxDistance = distributionSize / 2;
            if (selectionData.rank === 0) {
                targetPosition.set(0, 0, 0);
            } else {
                const proximityScore = (normMaxHits - selectionData.hits) / (normMaxHits - normMinHits);
                const radius = proximityScore * maxDistance;
                const angle = (selectionData.rank - 1) * (2 * Math.PI / SYZYGY_LINE_COUNT);
                const tempVec = new THREE.Vector3(radius * Math.cos(angle), radius * Math.sin(angle), 0);
                tempVec.applyQuaternion(currentPlaneQuaternion);
                targetPosition.copy(tempVec);
            }
        } else if (isNormalizedDensityActive) {
            const maxDistance = distributionSize / 2;
            const direction = new THREE.Vector3(neuron.basePosition.x, neuron.basePosition.y, neuron.basePosition.z).normalize();
            if (atlasEntry) {
                const proximityScore = (normMaxHits - atlasEntry.hits) / (normMaxHits - normMinHits);
                targetPosition.copy(direction.multiplyScalar(proximityScore * maxDistance));
            } else {
                targetPosition.copy(direction.multiplyScalar(maxDistance));
            }
        } else if (isGravityViewActive) {
            const pullFactor = Math.min((atlasEntry?.hits || 0) / 10, 1.0);
            const gravityFactor = 1.0 - (1.0 - 0.2) * pullFactor;
            targetPosition.set(neuron.basePosition.x, neuron.basePosition.y, neuron.basePosition.z).multiplyScalar(gravityFactor);
        } else {
            targetPosition.set(neuron.basePosition.x, neuron.basePosition.y, neuron.basePosition.z);
        }
        const currentPosVec = new THREE.Vector3().copy(neuron.currentPosition);
        currentPosVec.lerp(targetPosition, 0.05);
        neuron.currentPosition = { x: currentPosVec.x, y: currentPosVec.y, z: currentPosVec.z };
        positions[i * 3] = neuron.currentPosition.x;
        positions[i * 3 + 1] = neuron.currentPosition.y;
        positions[i * 3 + 2] = neuron.currentPosition.z;
    }
    starGeometry.attributes.position.needsUpdate = true;
    if(isOrreryViewActive && sortedSystems.length > 1) {
        orreryRingGroup.visible = true;
        const maxDistance = distributionSize / 2;
        for (let i = 0; i < SYZYGY_LINE_COUNT; i++) {
            const ring = orreryRingGroup.children[i];
            if (sortedSystems[i + 1]) {
                ring.visible = true;
                const hits = sortedSystems[i + 1][1].hits;
                const proximityScore = (normMaxHits - hits) / (normMaxHits - normMinHits);
                const radius = proximityScore * maxDistance;
                ring.scale.set(radius, radius, 1);
                ring.material.color.copy(ORBIT_HOT_COLOR).lerp(ORBIT_COLD_COLOR, proximityScore);
            } else {
                ring.visible = false;
            }
        }
    } else {
        orreryRingGroup.visible = false;
    }
    if (cameraTargetPosition) {
        controls.target.lerp(cameraTargetPosition, 0.08);
        if (controls.target.distanceTo(cameraTargetPosition) < 0.1) {
            cameraTargetPosition = null;
        }
    }
    if (areSystemTagsVisible) {
        const scale = Math.pow(controls.object.zoom, 1.2) * 0.3;
        for (const id in systemLabels) {
            const label = systemLabels[id];
            label.position.copy(gameState.neurons[id].currentPosition);
            label.scale.set(scale, scale, 1);
        }
    }
    if (tracerGroup.visible && animationProgress < 1) {
        animationProgress += deltaTime / animationDuration;
        const tempPosition = new THREE.Vector3().lerpVectors(startPosition, endPosition, animationProgress);
        tracerHistory.push(tempPosition.clone());
        if (tracerHistory.length > tracerLength) tracerHistory.shift();
        tracerGroup.children.forEach((child, index) => {
            const historyIndex = Math.max(0, tracerHistory.length - 1 - index);
            child.position.copy(tracerHistory[historyIndex]);
        });
    } else if (tracerGroup.visible && animationProgress >= 1) {
        tracerGroup.visible = false;
        starGeometry.attributes.color.setXYZ(lastActiveStarIndex, activeColor.r, activeColor.g, activeColor.b);
        starGeometry.attributes.color.needsUpdate = true;
        terminalOutput.textContent = `...Signal Lock Acquired. System [J5-${lastActiveStarIndex}] confirmed.`;
        activeLabelStarIndex = lastActiveStarIndex;
        neuronLabelElement.innerText = `J5-${lastActiveStarIndex}`;
        neuronLabelElement.classList.add('visible');
        terminalInput.disabled = false;
        isAnimating = false;
        terminalInput.focus();
        if (labelVisibilityTimeout) clearTimeout(labelVisibilityTimeout);
        labelVisibilityTimeout = setTimeout(() => {
            neuronLabelElement.classList.remove('visible');
            activeLabelStarIndex = -1;
            updateStarColors();
        }, 1500);
    }
    if (activeLabelStarIndex !== -1) {
        const screenPosition = new THREE.Vector3().copy(gameState.neurons[activeLabelStarIndex].currentPosition).project(camera);
        const x = (screenPosition.x * 0.5 + 0.5) * window.innerWidth;
        const y = (-screenPosition.y * 0.5 + 0.5) * window.innerHeight;
        neuronLabelElement.style.transform = `translate(-50%, -150%) translate(${x}px, ${y}px)`;
    }
    composer.render();
    window.requestAnimationFrame(tick);
};

// --- START THE APP ---
function initializeScene() {
    loadState();
    camera.position.z = 100;
    scene.add(camera);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    composer.setSize(window.innerWidth, window.innerHeight);
    const renderPass = new THREE.RenderPass(scene, camera);
    composer.addPass(renderPass);
    const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.2, 0.5, 0.9);
    composer.addPass(bloomPass);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.minZoom = 0.5;
    controls.maxZoom = 4.0;
    starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    scene.add(starfield, boxLines, tracerGroup, systemLabelsGroup, syzygyLineGroup, orreryRingGroup);
    for (let i = 0; i < SYZYGY_LINE_COUNT; i++) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2 * 3), 3));
        const material = new THREE.LineBasicMaterial({ color: discoveredColor, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        syzygyLineGroup.add(new THREE.Line(geometry, material));
    }
    for (let i = 0; i < SYZYGY_LINE_COUNT; i++) {
        const geometry = new THREE.RingGeometry(0.99, 1, 64);
        const material = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.4, side: THREE.DoubleSide });
        orreryRingGroup.add(new THREE.Mesh(geometry, material));
    }
    tick();
}

function activateUI() {
    terminalElement.classList.add('is-active');
    populateSpeedButtons();
    updateScannerSpeedUI();
    const allKnownSystems = new Set(Object.keys(gameState.atlas).flatMap(key => Object.keys(gameState.atlas[key])));
    allKnownSystems.forEach(id => createSystemLabel(id, gameState.neurons[id].basePosition));
    cacheSelect.innerHTML = '';
    gameState.autoScanner.caches.forEach(cache => {
        const option = document.createElement('option');
        option.value = cache.filename;
        option.innerText = cache.displayName;
        cacheSelect.appendChild(option);
    });
    cacheSelect.value = gameState.autoScanner.activeCache;
    updateStarColors();
    checkUnlock();
    updateScannerUI();
    initializeCalibrationUI();
    gameState.operatorRank = calculateTotalDiscoveredSystems();
    operatorRankValue.innerText = gameState.operatorRank;
    checkForUnlocks(0); 
    updateUIStates();
    populateCodexList(activeCodexLayer);
    updateGlobalTheme();
    if (gameState.autoScanner.isRunning) {
        scannerStatusText.innerText = 'RUNNING';
        btnScannerToggle.innerText = 'PAUSE ANALYSIS';
        dataStreamVisualizer.classList.remove('hidden');
        runScannerStep();
    }
    uiContainer.classList.add('active');
    viewportBorder.classList.add('active');
    if (btnProtoImplicit) btnProtoImplicit.classList.add('active');
    if (btnModeResonance) btnModeResonance.classList.add('active');

    // --- START: New Two-Way Sync Logic ---
    // Listen for storage changes made by other tabs (i.e., the Exchange)
    window.addEventListener('storage', (event) => {
        if (event.key === SAVE_KEY) {
            console.log("Detected gameState change from another tab. Re-syncing...");

            // Re-load the entire gameState from the now-authoritative localStorage
            try {
                const savedState = localStorage.getItem(SAVE_KEY);
                gameState = savedState ? JSON.parse(savedState) : createDefaultState();
                 if (!Array.isArray(gameState.unclaimedEvents)) {
                    gameState.unclaimedEvents = [];
                }
            } catch (error) {
                console.error("Failed to parse updated state from other tab:", error);
                return; 
            }

            // Crucially, re-run the UI update function to reflect the new state
            updateParcelUI();
        }
    });
    // --- END: New Two-Way Sync Logic ---

    updateParcelUI();
}

// --- LIVE LOG & INIT ---
function addLiveLogEntry(prompt) {
    const currentLatest = liveLogFeed.querySelector('.latest');
    if (currentLatest) currentLatest.classList.remove('latest');
    const newLogEntry = document.createElement('p');
    newLogEntry.textContent = `> ${prompt.length > 50 ? prompt.substring(0, 47) + '...' : prompt}`;
    newLogEntry.className = 'latest';
    liveLogFeed.appendChild(newLogEntry);
    while (liveLogFeed.children.length > MAX_LIVE_LOG_LINES) {
        liveLogFeed.firstChild.remove();
    }
}
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket_url = `${protocol}//${window.location.host}/live_feed`;
    const socket = new WebSocket(socket_url);
    socket.onopen = () => console.log("[WebSocket] Connection established");
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'scan_event' && data.prompt) {
                addLiveLogEntry(data.prompt);
            }
        } catch (error) {
            console.error("[WebSocket] Error parsing message:", error);
        }
    };
    socket.onclose = () => {
        console.error('[WebSocket] Connection died. Reconnecting in 3s...');
        setTimeout(initializeWebSocket, 3000);
    };
    socket.onerror = (error) => console.error(`[WebSocket] Error: ${error.message}`);
}
function init() {
    initializeScene();
    initializeWebSocket();
    videoManager.init();
    const loginOverlay = document.getElementById('login-overlay');
    const loginButton = document.getElementById('btn-login');
    const usernameInput = document.getElementById('username');
    const flickerInterval = setInterval(() => {
        usernameInput.value = LOGIN_USERNAMES[Math.floor(Math.random() * LOGIN_USERNAMES.length)];
    }, 75);
    loginButton.addEventListener('click', () => {
        clearInterval(flickerInterval);
        usernameInput.value = 'Operator';
        loginOverlay.style.opacity = '0';
        setTimeout(() => { loginOverlay.classList.add('hidden'); }, 500);
        activateUI();
        videoManager.play('assets/video/intro.mp4', { blendMode: 'screen', volume: 0.7, onEnded: () => console.log("Intro cinematic complete.") });
    }, { once: true });
}

// --- START: Final Parcel UI Function ---
function updateParcelUI() {
    const PARCEL_SIZE = 11;
    const unclaimedCount = gameState.unclaimedEvents?.length || 0;
    const currentParcelCount = Math.floor(unclaimedCount / PARCEL_SIZE);
    const progressCount = unclaimedCount % PARCEL_SIZE;

    if (parcelCountElement) {
        parcelCountElement.textContent = currentParcelCount;
    }

    // Show the button only if there is at least one parcel
    if (gotoExchangeBtn) {
        gotoExchangeBtn.style.display = currentParcelCount > 0 ? 'block' : 'none';
    }

    // Trigger flash animation if a new parcel was just completed
    if (currentParcelCount > lastKnownParcelCount) {
        const trackerElement = document.getElementById('parcel-tracker');
        if (trackerElement) {
            trackerElement.classList.add('is-flashing');
            // Clean up the class after the animation finishes
            trackerElement.addEventListener('animationend', () => {
                trackerElement.classList.remove('is-flashing');
            }, { once: true });
        }
    }
    lastKnownParcelCount = currentParcelCount; // Update the state for the next check

    if (parcelProgressDotsElement) {
        parcelProgressDotsElement.innerHTML = ''; // Clear old dots
        for (let i = 0; i < PARCEL_SIZE; i++) {
            const dot = document.createElement('div');
            dot.className = 'progress-dot';
            if (i < progressCount) {
                dot.classList.add('is-filled');
            }
            parcelProgressDotsElement.appendChild(dot);
        }
    }
}
// --- END: Final Parcel UI Function ---

init();