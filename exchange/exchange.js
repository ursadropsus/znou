// --- CONFIGURATION ---
const SAVE_KEY = 'chimeraDirectiveState_v5'; // Must match main.js
const CLAIM_API_URL = '/api/exchange/claim';
const LEADERBOARD_API_URL = '/api/leaderboard';
const PARCEL_SIZE = 11;

// --- STATE ---
let gameState = {};
let operatorId = null;
let mousePos = { x: 0, y: 0 }; // For parallax effects

// --- DOM ELEMENTS ---
const operatorIdElement = document.getElementById('operator-id');
const kredBalanceElement = document.getElementById('kred-balance');
const parcelListElement = document.getElementById('parcel-list');
const noParcelsMessage = document.getElementById('no-parcels-message');
const leaderboardListElement = document.getElementById('leaderboard-list');
const refreshLeaderboardBtn = document.getElementById('refresh-leaderboard-btn');
const panelTitles = document.querySelectorAll('.scramble-title');

// --- VISUAL EFFECTS (FROM ZNOU.ORG) ---

function hexToRgb(hex) {
    let h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const num = parseInt(h, 16) || 0;
    return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
}

const charset = "0123456789ABCDEF▮▯⧈⧉";
function scrambleText(element, text, duration = 30) {
    let iteration = 0;
    if (element.scrambleInterval) clearInterval(element.scrambleInterval);
    element.scrambleInterval = setInterval(() => {
        element.innerText = text.split("").map((char, index) => {
            if (index < iteration) return text[index];
            return charset[Math.floor(Math.random() * charset.length)];
        }).join("");
        if (iteration >= text.length) clearInterval(element.scrambleInterval);
        iteration += 0.7;
    }, duration);
}

function initNeuronBackground() {
    const canvas = document.getElementById("neuron-background");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const getAccentColor = () => getComputedStyle(document.documentElement).getPropertyValue('--primary-color').trim() || '#00ffff';
    let nodes = [], rafId;

    const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        const width = window.innerWidth, height = window.innerHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        nodes = Array.from({ length: 40 }, () => ({
            x: Math.random() * width, y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.05, vy: (Math.random() - 0.5) * 0.05,
            phase: Math.random() * Math.PI * 2,
        }));
    };
    window.addEventListener("resize", resize);
    resize();

    const render = (time) => {
        const width = window.innerWidth, height = window.innerHeight;
        ctx.clearRect(0, 0, width, height);
        ctx.globalCompositeOperation = "lighter";
        const { r, g, b } = hexToRgb(getAccentColor());
        nodes.forEach(n => {
            n.x += n.vx; n.y += n.vy;
            if (n.x < 0 || n.x > width) n.vx *= -1;
            if (n.y < 0 || n.y > height) n.vy *= -1;
            const pulse = 0.5 + 0.5 * Math.sin(time / 1000 + n.phase);
            ctx.beginPath();
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.1 + pulse * 0.2})`;
            ctx.arc(n.x, n.y, 1.5 + pulse * 1.5, 0, Math.PI * 2);
            ctx.fill();
        });
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dist = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
                if (dist < 150) {
                    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${(1 - dist / 150) * 0.3})`;
                    ctx.lineWidth = 0.8;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                }
            }
        }
        rafId = requestAnimationFrame(render);
    };
    rafId = requestAnimationFrame(render);
}

function initMouseEffects() {
    window.addEventListener("mousemove", (e) => {
        mousePos.x = (e.clientX / window.innerWidth) - 0.5;
        mousePos.y = (e.clientY / window.innerHeight) - 0.5;
    });
}

function initFloatingNumbers() {
    const container = document.getElementById("floating-numbers-container");
    if (!container) return;
    for (let i = 0; i < 24; i++) {
        const cluster = document.createElement('div');
        cluster.className = `number-cluster layer-${(i % 3) + 1}`;
        cluster.style.top = `${Math.random() * 100}%`;
        cluster.style.left = `${Math.random() * 100}%`;
        cluster.innerHTML = `<div class="number-cluster-inner">${Array(4).fill(0).map(() => `<div class="number-line"><div class="number-symbol">${Array(6).fill(0).map(() => `<div class="symbol-pip"></div>`).join('')}</div><span class="random-number">0</span></div>`).join('')}</div>`;
        container.appendChild(cluster);
    }
    const numberSpans = container.querySelectorAll(".random-number");
    setInterval(() => {
        numberSpans.forEach(span => {
            span.innerText = Math.floor(Math.random() * 3072).toString();
        });
    }, 1500);
}

function initParallaxEffects() {
    const dotLayer1 = document.querySelector(".dot-field.layer-1");
    const dotLayer2 = document.querySelector(".dot-field.layer-2");
    const dotLayer3 = document.querySelector(".dot-field.layer-3");
    const numberClusters1 = document.querySelectorAll(".number-cluster.layer-1");
    const numberClusters2 = document.querySelectorAll(".number-cluster.layer-2");
    const numberClusters3 = document.querySelectorAll(".number-cluster.layer-3");

    function onFrame() {
        const scrollY = window.scrollY;
        // Dot Field Parallax
        const dotY1 = scrollY * -0.05, dotY2 = scrollY * -0.12, dotY3 = scrollY * -0.20;
        const mouseX1 = mousePos.x * 140, mouseY1 = mousePos.y * 140;
        const mouseX2 = mousePos.x * 100, mouseY2 = mousePos.y * 100;
        const mouseX3 = mousePos.x * 60, mouseY3 = mousePos.y * 60;
        if(dotLayer1) dotLayer1.style.transform = `translate(${mouseX1}px, ${dotY1 + mouseY1}px)`;
        if(dotLayer2) dotLayer2.style.transform = `translate(${mouseX2}px, ${dotY2 + mouseY2}px)`;
        if(dotLayer3) dotLayer3.style.transform = `translate(${mouseX3}px, ${dotY3 + mouseY3}px)`;

        // Floating Numbers Parallax
        const numY1 = scrollY * -0.013, numY2 = scrollY * -0.03, numY3 = scrollY * -0.05;
        numberClusters1.forEach(el => el.style.transform = `translateY(${numY1}px)`);
        numberClusters2.forEach(el => el.style.transform = `translateY(${numY2}px)`);
        numberClusters3.forEach(el => el.style.transform = `translateY(${numY3}px)`);
        
        requestAnimationFrame(onFrame);
    }
    requestAnimationFrame(onFrame);
}

// --- CORE FUNCTIONS ---

function loadStateAndSetup() {
    try {
        const savedState = localStorage.getItem(SAVE_KEY);
        gameState = savedState ? JSON.parse(savedState) : { unclaimedEvents: [], operatorRank: 0 };
        if (!Array.isArray(gameState.unclaimedEvents)) gameState.unclaimedEvents = [];

        operatorId = localStorage.getItem('operatorId');
        if (!operatorId) {
            operatorId = 'ZNO-' + Math.random().toString(36).substr(2, 8).toUpperCase();
            localStorage.setItem('operatorId', operatorId);
        }

        scrambleText(operatorIdElement, `OPERATOR: [${operatorId}]`);
        scrambleText(kredBalanceElement, `KREDS: [SYNCING...]`);

    } catch (error) {
        console.error("Failed to load state from localStorage:", error);
        document.body.innerHTML = "<h1>Error: Could not load game state.</h1>";
    }
}

function renderParcels() {
    parcelListElement.innerHTML = '';
    const unclaimed = gameState.unclaimedEvents || [];
    const numberOfParcels = Math.floor(unclaimed.length / PARCEL_SIZE);

    noParcelsMessage.classList.toggle('hidden', numberOfParcels > 0);

    for (let i = 0; i < numberOfParcels; i++) {
        const item = document.createElement('div');
        item.className = 'unclaimed-item';
        item.innerHTML = `
            <span>Data Parcel #${i + 1} (${PARCEL_SIZE} Assets)</span>
            <button class="sell-parcel-btn" data-parcel-index="${i}">SELL PARCEL</button>
        `;
        parcelListElement.appendChild(item);
    }
}

async function fetchAndRenderLeaderboard() {
    try {
        const response = await fetch(LEADERBOARD_API_URL);
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        const leaderboardData = await response.json();

        leaderboardListElement.innerHTML = '';
        if (leaderboardData.length === 0) {
            leaderboardListElement.innerHTML = '<p>No operator data available.</p>';
            return;
        }

        const myStats = leaderboardData.find(entry => entry.operator_id === operatorId);
        if (myStats) {
            scrambleText(kredBalanceElement, `KREDS: [${myStats.total_yield}]`);
        } else {
            scrambleText(kredBalanceElement, `KREDS: [UNRANKED]`);
        }

        leaderboardData.forEach(entry => {
            const item = document.createElement('div');
            item.className = 'leaderboard-item';
            item.innerHTML = `
                <span>#${entry.rank} ${entry.operator_id}</span>
                <span>${entry.total_yield} Kreds</span>
            `;
            leaderboardListElement.appendChild(item);
        });

    } catch (error) {
        console.error("Failed to fetch leaderboard:", error);
        leaderboardListElement.innerHTML = '<p>Error fetching yield report.</p>';
    }
}

/**
 * Displays the Yield Settlement Report in a modal window.
 * @param {object} data The full response data from the claim API.
 */
function displaySettlementReport(data) {
    const modal = document.getElementById('settlement-report-modal');
    const summaryEl = document.getElementById('settlement-summary');
    const listEl = document.getElementById('settlement-list');
    const dismissBtn = document.getElementById('settlement-dismiss-btn');
    const titleEl = document.getElementById('settlement-title');

    if (!modal || !listEl || !summaryEl || !dismissBtn || !titleEl) {
        console.error("Settlement report modal elements not found!");
        return;
    }

    const summaryText = `TOTAL YIELD FROM ${data.events_claimed_count} ASSETS: ${data.kreds_awarded} Kreds`;
    scrambleText(summaryEl, summaryText);
    scrambleText(titleEl, "Yield Settlement Report");

    listEl.innerHTML = '';
    data.settlement_report.forEach(item => {
        const gradeClass = 'grade-' + item.yield_grade.toLowerCase().replace(/ /g, '-');
        
        const itemEl = document.createElement('div');
        itemEl.className = `settlement-item`;
        // --- THIS HTML BLOCK IS THE ONLY PART THAT'S CHANGED ---
        itemEl.innerHTML = `
            <div class="settlement-item-summary ${gradeClass}">
                <div>
                    <span class="settlement-toggle">[+]</span>
                    <span>${item.asset_id}</span>
                </div>
                <span>${item.kreds_yielded} Kreds</span>
            </div>
            <div class="settlement-details">
                <p><strong>Method:</strong> ${item.method}</p>
                <p><strong>Rarity Index:</strong> ${item.rarity_index_percent}</p>
                <p><strong>Yield Grade:</strong> ${item.yield_grade}</p>
                <p><strong>Source Prompt:</strong> <span class="prompt-text">"${item.prompt}"</span></p>
            </div>
        `;
        listEl.appendChild(itemEl);
    });

    const dismiss = () => modal.classList.add('hidden');
    dismissBtn.onclick = dismiss;
    modal.onclick = (event) => {
        if (event.target === modal) dismiss();
    };

    listEl.onclick = (event) => {
        const toggle = event.target.closest('.settlement-toggle');
        if (toggle) {
            const item = toggle.closest('.settlement-item');
            item.classList.toggle('expanded');
            toggle.textContent = item.classList.contains('expanded') ? '[-]' : '[+]';
        }
    };

    modal.classList.remove('hidden');
}

async function handleSellParcelClick(event) {
    const sellButton = event.target;
    if (!sellButton.matches('.sell-parcel-btn')) return;

    const parcelIndex = parseInt(sellButton.dataset.parcelIndex, 10);
    const start = parcelIndex * PARCEL_SIZE;
    const eventIdsForParcel = gameState.unclaimedEvents.slice(start, start + PARCEL_SIZE);

    sellButton.disabled = true;
    sellButton.textContent = 'SELLING...';

    try {
        const response = await fetch(CLAIM_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ operator_id: operatorId, event_ids: eventIdsForParcel })
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `HTTP Error: ${response.status}`);
        
        // Update local state and UI first
        scrambleText(kredBalanceElement, `KREDS: [${result.new_balance}]`);
        gameState.unclaimedEvents.splice(start, PARCEL_SIZE);
        localStorage.setItem(SAVE_KEY, JSON.stringify(gameState));
        
        renderParcels();
        fetchAndRenderLeaderboard();

        // THEN, display the full report
        displaySettlementReport(result);

    } catch (error) {
        console.error("Failed to sell parcel:", error);
        sellButton.textContent = 'ERROR';
        alert(`Claim failed: ${error.message}`);
    }
}

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all visual effects first
    initNeuronBackground();
    initMouseEffects();
    initFloatingNumbers();
    initParallaxEffects();

    // Then load core game logic
    loadStateAndSetup();
    renderParcels();
    fetchAndRenderLeaderboard();

    // Scramble panel titles on load
    panelTitles.forEach(title => {
        const originalText = title.textContent;
        scrambleText(title, originalText);
    });

    // Add event listeners
    parcelListElement.addEventListener('click', handleSellParcelClick);
    refreshLeaderboardBtn.addEventListener('click', fetchAndRenderLeaderboard);

    window.addEventListener('storage', (event) => {
        if (event.key === SAVE_KEY) {
            try {
                const savedState = localStorage.getItem(SAVE_KEY);
                gameState = savedState ? JSON.parse(savedState) : { unclaimedEvents: [], operatorRank: 0 };
                if (!Array.isArray(gameState.unclaimedEvents)) gameState.unclaimedEvents = [];
            } catch (error) {
                console.error("Failed to parse updated state:", error);
                return;
            }
            renderParcels();
        }
    });
});