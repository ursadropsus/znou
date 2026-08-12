function lerp(a, b, alpha) {
    return a + alpha * (b - a);
}

/* =========================
   Utility: Hex to RGB
========================= */
function hexToRgb(hex) {
    let h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const num = parseInt(h, 16) || 0;
    return {
        r: (num >> 16) & 255,
        g: (num >> 8) & 255,
        b: num & 255,
    };
}
document.addEventListener("DOMContentLoaded", () => {
    
/* =========================
   Utility: Scramble Text
========================= */
const charset = "0123456789ABCDEF▮▯⧈⧉";

function scrambleText(element, text, duration = 30) {
    let iteration = 0;
    let interval = null;

    // Clear any existing interval to prevent conflicts
    if (element.scrambleInterval) {
        clearInterval(element.scrambleInterval);
    }

    interval = setInterval(() => {
        element.innerText = text
            .split("")
            .map((char, index) => {
                if(index < iteration) {
                    return text[index];
                }
                return charset[Math.floor(Math.random() * charset.length)];
            })
            .join("");
        
        if(iteration >= text.length){
            clearInterval(interval);
        }
        
        iteration += 0.7; 
    }, duration);

    element.scrambleInterval = interval;
}
    

/* =========================
   Neuron Network Background
========================= */
function initNeuronBackground() {
    const canvas = document.getElementById("neuron-background");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const getAccentColor = () => getComputedStyle(document.documentElement).getPropertyValue('--accent-color').trim() || '#29eaff';

    let nodes = [];
    let rafId;

    const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        const width = window.innerWidth;
        const height = window.innerHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // OPTIMIZATION: Reduced node count and velocity
        nodes = Array.from({ length: 35 }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.03,
            vy: (Math.random() - 0.5) * 0.03,
            phase: Math.random() * Math.PI * 2,
        }));
    };

    window.addEventListener("resize", resize);
    resize();

    const maxDist = 140;

    const render = (time) => {
        const t = time / 1000;
        const width = window.innerWidth;
        const height = window.innerHeight;

        ctx.clearRect(0, 0, width, height);
        ctx.globalCompositeOperation = "lighter";

        const accentColor = getAccentColor();
        const { r, g, b } = hexToRgb(accentColor);

        nodes.forEach(n => {
            n.x += n.vx;
            n.y += n.vy;
            if (n.x < 0 || n.x > width) n.vx *= -1;
            if (n.y < 0 || n.y > height) n.vy *= -1;
            const pulse = 0.4 + 0.4 * Math.sin(t * 2 + n.phase);
            ctx.beginPath();
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.12 + pulse * 0.25})`;
            ctx.arc(n.x, n.y, 1.6 + pulse * 1.8, 0, Math.PI * 2);
            ctx.fill();
        });

        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i];
                const b = nodes[j];
                const dist = Math.hypot(a.x - b.x, a.y - b.y);
                if (dist < maxDist) {
                    const alpha = (1 - dist / maxDist) * 0.3;
                    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                    ctx.lineWidth = 0.7;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }
        rafId = requestAnimationFrame(render);
    };
    rafId = requestAnimationFrame(render);
}

/* =========================
   Header HUD Logic
========================= */
function initHeaderHUD() {
    const header = document.getElementById("hud-header");
    const telemetryText = document.getElementById("telemetry-text");
    const diagnosticsText = document.getElementById("diagnostics-text");

    const telemetryLines = [
        "LINK STABILITY 99.34% ▮ QUADRANT METHOD: ACTIVE ▮ PROTOTYPE STATUS: ONLINE",
        "LATENCY 0.83ms ▮ ENERGY DRAIN 1.7% ▮ ACTIVE THREADS 3072",
        "PEAK RESONANCE: 1888 ▮ CACHE STATE: NORMAL ▮ CODEX SYNCED",
        "SIGNAL: PRELIMINARY ▮ FILTERS ACTIVE ▮ ATLAS NOMINAL",
    ];

    const diagnostics = [
        "ACCESS PORT: ARKNODE-11SECURE ▮ MESHNET EXCHANGE: NOT CONFIGURED ▮ PRIVILEGE: OPERATOR-LEVEL",
        "PACKET INTEGRITY 100% ▮ ERROR CORRECTION ACTIVE ▮ MAP TRANSLATION: 1:1",
        "MLPL: 5 ▮ 10:1 RATE LIMIT ESTABLISHED ▮ AWAITING MAPPING INPUTS",
    ];

    let telIndex = 0, diagIndex = 0, diagInterval = null;

    setInterval(() => {
        telIndex = (telIndex + 1) % telemetryLines.length;
        scrambleText(telemetryText, telemetryLines[telIndex], 24);
    }, 4000);

    header.addEventListener("mouseenter", () => {
        if (!diagInterval) {
            scrambleText(diagnosticsText, diagnostics[diagIndex], 20);
            diagInterval = setInterval(() => {
                diagIndex = (diagIndex + 1) % diagnostics.length;
                scrambleText(diagnosticsText, diagnostics[diagIndex], 20);
            }, 3000);
        }
    });

    header.addEventListener("mouseleave", () => {
        clearInterval(diagInterval);
        diagInterval = null;
        diagnosticsText.innerText = "";
    });
    
    scrambleText(telemetryText, telemetryLines[0], 24);
}

/* =========================
   Floating Numbers Field
========================= */
function initFloatingNumbers() {
    const container = document.getElementById("floating-numbers-container");
    if (!container) return;
    const fieldCount = 24;
    
    for (let i = 0; i < fieldCount; i++) {
        const cluster = document.createElement('div');
        cluster.className = `number-cluster layer-${(i % 3) + 1}`;
        cluster.style.top = `${Math.random() * 100}%`;
        cluster.style.left = `${Math.random() * 100}%`;
        cluster.innerHTML = `
            <div class="number-cluster-inner">
                ${Array(4).fill(0).map(() => `
                    <div class="number-line">
                        <div class="number-symbol">${Array(6).fill(0).map(() => `<div class="symbol-pip"></div>`).join('')}</div>
                        <span class="random-number">0</span>
                    </div>`).join('')}
            </div>`;
        container.appendChild(cluster);
    }

    const numberSpans = container.querySelectorAll(".random-number");
    setInterval(() => {
        numberSpans.forEach(span => {
            span.innerText = Math.floor(Math.random() * 3072).toString();
        });
    }, 1500);
}

/* =========================
   Scroll-Based Effects
========================= */
function initScrollEffects() {
    const topColor = hexToRgb("#29eaff");
    const bottomColor = hexToRgb("#46e83a");

    const dotLayer1 = document.querySelector(".dot-field.layer-1");
    const dotLayer2 = document.querySelector(".dot-field.layer-2");
    const dotLayer3 = document.querySelector(".dot-field.layer-3");
    const numberClusters1 = document.querySelectorAll(".number-cluster.layer-1");
    const numberClusters2 = document.querySelectorAll(".number-cluster.layer-2");
    const numberClusters3 = document.querySelectorAll(".number-cluster.layer-3");

    // --- NEW: Elements for the video scape ---
    const scapeContainer = document.getElementById("video-scape-section");
    const scapeBg1 = document.getElementById("scape-bg-1");
    const videoPanels = document.querySelectorAll(".video-panel");

    function onFrame() {
        const scrollY = window.scrollY;
        
        // --- 1. Color Interpolation ---
        const scrollFraction = Math.min(1, scrollY / 2000);
        const r = Math.round(lerp(topColor.r, bottomColor.r, scrollFraction));
        const g = Math.round(lerp(topColor.g, bottomColor.g, scrollFraction));
        const b = Math.round(lerp(topColor.b, bottomColor.b, scrollFraction));
        document.documentElement.style.setProperty('--accent-color', `rgb(${r}, ${g}, ${b})`);

        // --- 2. Dot Field Parallax ---
        const dotY1 = scrollY * -0.05, dotY2 = scrollY * -0.12, dotY3 = scrollY * -0.20;
        const mouseX1 = mousePos.x * 140, mouseY1 = mousePos.y * 140;
        const mouseX2 = mousePos.x * 100, mouseY2 = mousePos.y * 100;
        const mouseX3 = mousePos.x * 60, mouseY3 = mousePos.y * 60;
        dotLayer1.style.transform = `translate(${mouseX1}px, ${dotY1 + mouseY1}px)`;
        dotLayer2.style.transform = `translate(${mouseX2}px, ${dotY2 + mouseY2}px)`;
        dotLayer3.style.transform = `translate(${mouseX3}px, ${dotY3 + mouseY3}px)`;
        
        // --- 3. Floating Numbers Parallax ---
        const scrollFraction3k = Math.min(1, scrollY / 3000);
        const numY1 = scrollY * -0.013, numY2 = scrollY * -0.03, numY3 = scrollY * -0.05;
        numberClusters1.forEach(el => el.style.transform = `translateY(${numY1}px)`);
        numberClusters2.forEach(el => el.style.transform = `translateY(${numY2}px)`);
        numberClusters3.forEach(el => el.style.transform = `translateY(${numY3}px)`);
        const op1 = lerp(0.9, 0.4, scrollFraction3k), op2 = lerp(0.7, 0.3, scrollFraction3k), op3 = lerp(0.5, 0.2, scrollFraction3k);
        numberClusters1.forEach(el => el.style.opacity = op1);
        numberClusters2.forEach(el => el.style.opacity = op2);
        numberClusters3.forEach(el => el.style.opacity = op3);
        
        // --- 4. NEW: Video Scape Parallax ---
        if (scapeContainer) {
            const rect = scapeContainer.getBoundingClientRect();
            const scrollProgress = Math.max(0, Math.min(1, -rect.top / (rect.height - window.innerHeight)));
            
            // Fade out the top background image
            scapeBg1.style.opacity = Math.max(0, 1 - (scrollProgress / 0.4));

            // Move video panels
            videoPanels.forEach(panel => {
                const speed = parseFloat(panel.dataset.parallaxSpeed) || 0;
                const yOffset = scrollProgress * 1200 * speed;
                panel.style.transform = `translateY(-${yOffset}px)`;
            });
        }
        
        requestAnimationFrame(onFrame);
    }
    requestAnimationFrame(onFrame);
}

/* =========================
   Scramble on Scroll Logic
========================= */
function initScrambleOnScroll() {
    const elements = document.querySelectorAll(".scramble-on-scroll");
    elements.forEach(el => {
        el.dataset.originalText = el.innerText;
        el.innerText = "";
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                scrambleText(el, el.dataset.originalText);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.2 });

    elements.forEach(el => observer.observe(el));
}

/* =========================
   Mouse-Based Parallax
========================= */
let mousePos = { x: 0, y: 0 };
function initMouseEffects() {
    window.addEventListener("mousemove", (e) => {
        mousePos.x = (e.clientX / window.innerWidth) - 0.5;
        mousePos.y = (e.clientY / window.innerHeight) - 0.5;
    });
}

/* =========================
   NEW: Lazy Load Videos
========================= */
function initLazyLoadVideos() {
    const videos = document.querySelectorAll(".video-panel video");
    if (videos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Play the video, catching potential browser errors
                video.play().catch(error => console.error("Video play failed:", error));
            } else {
                video.pause();
            }
        });
    }, { threshold: 0.2 }); // Trigger when 20% of the video is visible

    videos.forEach(video => observer.observe(video));
}

/* =========================
   Boot Sequence Logic
========================= */
function runBootSequence() {
    const bootScreen = document.getElementById("boot-sequence");
    const bootTextElement = bootScreen.querySelector(".scramble-text");
    const mainContent = document.getElementById("main-content");
    
    const messages = [
        "CONNECTING TO MESOCOSM LAYER...",
        "PRE-SELECTING OPERATOR ID...",
        "TRANSLATING INTERMEDIUM STREAM...",
        "LINK STABLE / BEGINNING OF SEQUENCE",
    ];

    let currentStage = 0;
    function nextStage() {
        if (currentStage >= messages.length) {
            setTimeout(() => {
                bootScreen.classList.add("finished");
                mainContent.classList.remove("hidden");
            }, 500);
            return;
        }
        scrambleText(bootTextElement, messages[currentStage]);
        currentStage++;
        setTimeout(nextStage, 1100);
    }
    nextStage();
}

    // Run everything
    runBootSequence();
    initNeuronBackground();
    initHeaderHUD();
    initMouseEffects();
    initFloatingNumbers();
    initScrollEffects();
    initScrambleOnScroll();
    initLazyLoadVideos(); // <-- New function call

});