/* --------------------------------------------------- */
/* The Chimera Directive - Codex Data File (v6)        */
/* --------------------------------------------------- */

const CODEX_DATA = {
    // --- LAYER 1: (In-Universe Z-Nou Perspective) ---
    eve: [
        {
            id: 'welcome',
            title: '00: Calibration Protocols',
            videoSrc: 'assets/video/codex_z01_operator.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'initial' },
                    description: `Operator, your connection to the region requires calibration. The system will only stabilize by analyzing novel resonance signals.\n\nProvide unique inputs to proceed.`
                }
            ]
        },
        {
            id: 'resonance_basics',
            title: '01: Resonance & The Region',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 1 },
                    description: `First signal anchor established. A successful Resonance Key fires a tracer to a corresponding system node, confirming its existence.\n\nThe geometry of your dataspace is beginning to resolve.`
                }
            ]
        },
        {
            id: 'atlas_use',
            title: '02: The Resonance Atlas',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 2 },
                    description: `The Atlas is now calibrated to your specific Genesis Point. It serves as your personal log of all discovered systems.`
                }
            ]
        },
        {
            id: 'discovery_filter',
            title: '03: Filtering the Void',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 2 },
                    description: `With sufficient data points, integrated Z-NOU Neuroptics can apply a discovery filter allowing differentiation between confirmed and unknown signal nodes.\n\nFilter interfaces exist to carve signal from noise.`
                }
            ]
        },
        {
            id: 'resonant_hotspots',
            title: '04: Signal Echoes & Hotspots',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'repeatHit', count: 5 },
                    description: `Analysis indicates some nodes are highly resonant, responding to multiple, conceptually similar inputs. These 'hotspots' are of key interest to the Directive.`
                }
            ]
        },
        {
            id: 'gravity_filter',
            title: '05: Resonance Density',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 5 },
                    description: `Z-NOU analytics has unlocked a new visualization mode: Gravity Density. This filter pulls highly resonant systems toward the network's core, revealing clusters of influence.`
                }
            ]
        },
        {
            id: 'map_duality',
            title: '06: Signal Duality (Resonance/Inference)',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 10 },
                    description: `Operator, your field experience has unlocked a new analysis paradigm. Initial mapping focused on peak signal amplitude, designated 'Resonance'.\n\nContinued analysis reveals a secondary pattern derived from signal termination vectors. This 'Inference' map represents alternative models of communicative context.\n\nSwitching between these maps may reveal divergent patterns.`
                }
            ]
        },
        { // --- NEW ENTRY FOR EXPLICIT PROTOCOL UNLOCK ---
            id: 'explicit_protocol',
            title: '07: The Explicit Protocol',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 20 },
                    description: `High-rank operator status has granted you access to a modern, analytical interface: the Explicit Protocol.\n\nYour training began on the 'Implicit' protocol, which packages precursorial socio-linguistic cues alongside the raw signals of your intent.\n\nThe Explicit protocol removes this context, sending only the precise data of your input - no more or less.\n\nThough some may label this method of communication sterile, it is a more mathematically pure signal.\n\nNote: system mappings may differ significantly between protocols.`
                }
            ]
        },
        {
            id: 'cache_analysis',
            title: '08: Cache Analysis Unit', // Renumbered
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [
                {
                    unlockConditions: { type: 'rank', value: 10 }, // Lowered rank to match new pacing
                    description: `Sufficient field experience has been logged. You are now authorized to operate the Cache Analysis Unit, which automates discovery by scanning fragmented data caches.\n\nWarning: Cache Units are prototype technology.\n\nMemory overflows may occur.\nConsult a qualified neural specialist if you feel unwell after Operation.`
                }
            ]
        }
    ],

    // --- LAYER 2 & 3 (Unchanged) ---
    ursa: [
        {
            id: 'welcome',
            title: 'Embroidery Files',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'initial' }, description: `we're inside something strange\n\nit may not make sense at first\n\ntry curiosity` } ]
        },
        {
            id: 'resonance_basics',
            title: 'Darkness',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 1 }, description: `resonance means what responded the most to your call, not what was the response` } ]
        },
        {
            id: 'atlas_use',
            title: 'Design: The Atlas',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 2 }, description: `we found our kin and found them strange` } ]
        },
         {
            id: 'discovery_filter',
            title: 'Dev Note: First Big Unlock',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 2 }, description: `it's not much of an existence if there isn't the possibility for discovery\n\nturn the filter on\nthis place won't epistemologize itself\n\nnow you can see your imprints` } ]
        },
        {
            id: 'resonant_hotspots',
            title: 'Dev Note: On "That One Neuron"',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'repeatHit', count: 5 }, description: `you found one eh? there are a few friends in here that perk up for a huge range of inputs\n\nor maybe this layer isn't responsive to those inputs, and they get sent off down one of the big highways elsewhere to be dealt with, or maybe they arrived already knowing where they're going next, and were already headed for the highway\n\nI try not to overinterpret, but these are curious things aren't they` } ]
        },
        {
            id: 'gravity_filter',
            title: 'Dev Note: Adding Some Physics',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 5 }, description: `the gravity filter is a partly aesthetic thing, but by imposing a topology with our own arbitrary rules we can still reveal insightful structure\n\nit basically translates hit count to topology` } ]
        },
        {
            id: 'cache_analysis',
            title: 'Design: The Auto-Scanner',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 10 }, description: `the auto-scanner is a remnant of the 'idle game' concept this grew out of\n\ni've pushed it earlier in the progression for now so testing/demo'ing this all is quick.` } ]
        }
    ],
    interpretability: [
        {
            id: 'welcome',
            title: 'Concept: AI Interpretability',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'initial' }, description: `AI models like GPT are often called 'black boxes' because it's hard to understand why they give a certain output. Interpretability is the field of research dedicated to looking inside these boxes.` } ]
        },
        {
            id: 'resonance_basics',
            title: 'Tech: Neuron Activation',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 1 }, description: `When you enter a prompt, it's sent to a live GPT-2 model. The server identifies which 'neuron' in a specific layer of the model's neural network has the highest activation value. The ID of that single, most-activated neuron is what returns back to us, here.` } ]
        },
        {
            id: 'atlas_use',
            title: 'Tech: Mapping the Model',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 2 }, description: `The 3,072 stars in the cube directly correspond to the 3,072 neurons in layer 5 of the GPT-2 Small model. Our Atlas is a personal, partial map showing what kinds of concepts activate specific neurons.` } ]
        },
        {
            id: 'discovery_filter',
            title: 'Tech: Differentiating Data',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 2 }, description: `The Discovery Filter is a simple data visualization tool. It colors a star based on a boolean flag in your local save state, allowing you to visually separate known from unknown territory.` } ]
        },
        {
            id: 'resonant_hotspots',
            title: 'Tech: Activation Frequency',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'repeatHit', count: 5 }, description: `You have repeatedly activated the same neuron. Some neurons are tuned to very general concepts and thus have a high activation frequency. This 'hit count' is a primary metric in research for identifying neuron function.` } ]
        },
        {
            id: 'gravity_filter',
            title: 'Tech: Visualizing Influence',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 5 }, description: `The Gravity View is a cosmetic visualization that links the 'hits' variable of a neuron to a physics-based pull towards the center. While not a direct scientific tool, it helps to intuitively grasp which of your discoveries are the most 'influential'.` } ]
        },
        {
            id: 'cache_analysis',
            title: 'Tech: Pre-computed Data',
            videoSrc: 'assets/video/codex_placeholder.mp4',
            contentBlocks: [ { unlockConditions: { type: 'rank', value: 10 }, description: `The data caches are JSON files containing pre-computed prompt-neuron pairs. This provides a way to discover systems without needing to run a live query, which is computationally more expensive.` } ]
        }
    ]
};