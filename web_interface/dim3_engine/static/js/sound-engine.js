// ============================================================================
// DIM3 — SoundEngine
// Procedural sound effects using Web Audio API.
// Initialized only after first user interaction (browser autoplay policy).
// ============================================================================

class _SoundEngine {
    constructor() {
        this.ctx = null;
        this.initialized = false;
        this._initPromise = null;
        this._masterGain = null;
    }

    init() {
        if (this.initialized) return Promise.resolve();
        if (this._initPromise) return this._initPromise;

        this._initPromise = new Promise((resolve) => {
            try {
                this.ctx = new (window.AudioContext || window.webkitAudioContext)();
                this._masterGain = this.ctx.createGain();
                this._masterGain.gain.value = 0.35;
                this._masterGain.connect(this.ctx.destination);
                this.initialized = true;
                resolve();
            } catch (e) {
                console.warn('SoundEngine: Web Audio not available', e);
                resolve();
            }
        });
        return this._initPromise;
    }

    async play(name) {
        if (!this.initialized) await this.init();
        if (!this.ctx) return;

        if (this.ctx.state === 'suspended') {
            await this.ctx.resume();
        }

        switch (name) {
            case 'swipe': this._playSwipe(); break;
            case 'switch': this._playSwitch(); break;
            case 'scan': this._playScan(); break;
            case 'tick': this._playTick(); break;
            default: break;
        }
    }

    // Soft whoosh for panel transitions
    _playSwipe() {
        const now = this.ctx.currentTime;
        const duration = 0.25;

        const bufferSize = this.ctx.sampleRate * duration;
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            data[i] = (Math.random() * 2 - 1) * 0.5;
        }

        const noise = this.ctx.createBufferSource();
        noise.buffer = buffer;

        const filter = this.ctx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(2000, now);
        filter.frequency.exponentialRampToValueAtTime(600, now + duration);
        filter.Q.value = 1.2;

        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0.0, now);
        gain.gain.linearRampToValueAtTime(0.15, now + 0.04);
        gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this._masterGain);

        noise.start(now);
        noise.stop(now + duration);
    }

    // Deeper transition sound for page switches
    _playSwitch() {
        const now = this.ctx.currentTime;

        const osc1 = this.ctx.createOscillator();
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(280, now);
        osc1.frequency.exponentialRampToValueAtTime(180, now + 0.15);

        const osc2 = this.ctx.createOscillator();
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(420, now + 0.06);
        osc2.frequency.exponentialRampToValueAtTime(320, now + 0.2);

        const gain1 = this.ctx.createGain();
        gain1.gain.setValueAtTime(0.12, now);
        gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.2);

        const gain2 = this.ctx.createGain();
        gain2.gain.setValueAtTime(0.0, now);
        gain2.gain.linearRampToValueAtTime(0.1, now + 0.06);
        gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.25);

        osc1.connect(gain1);
        gain1.connect(this._masterGain);
        osc1.start(now);
        osc1.stop(now + 0.2);

        osc2.connect(gain2);
        gain2.connect(this._masterGain);
        osc2.start(now + 0.04);
        osc2.stop(now + 0.25);
    }

    // Sci-fi sweep for LIDAR scan
    _playScan() {
        const now = this.ctx.currentTime;
        const duration = 2.5;

        const osc = this.ctx.createOscillator();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(80, now);
        osc.frequency.exponentialRampToValueAtTime(2000, now + duration * 0.7);
        osc.frequency.exponentialRampToValueAtTime(150, now + duration);

        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(400, now);
        filter.frequency.exponentialRampToValueAtTime(3000, now + duration * 0.5);
        filter.frequency.exponentialRampToValueAtTime(200, now + duration);
        filter.Q.value = 5;

        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0.0, now);
        gain.gain.linearRampToValueAtTime(0.08, now + 0.3);
        gain.gain.setValueAtTime(0.08, now + duration * 0.8);
        gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

        // Add subtle noise layer
        const noiseDur = duration;
        const bufferSize = this.ctx.sampleRate * noiseDur;
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            data[i] = (Math.random() * 2 - 1);
        }
        const noiseSrc = this.ctx.createBufferSource();
        noiseSrc.buffer = buffer;
        const noiseFilter = this.ctx.createBiquadFilter();
        noiseFilter.type = 'bandpass';
        noiseFilter.frequency.value = 1500;
        noiseFilter.Q.value = 0.5;
        const noiseGain = this.ctx.createGain();
        noiseGain.gain.setValueAtTime(0.0, now);
        noiseGain.gain.linearRampToValueAtTime(0.04, now + 0.5);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this._masterGain);
        osc.start(now);
        osc.stop(now + duration);

        noiseSrc.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(this._masterGain);
        noiseSrc.start(now);
        noiseSrc.stop(now + duration);
    }

    // Tactile tick for dropdown scroll
    _playTick() {
        const now = this.ctx.currentTime;

        const osc = this.ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(1800, now);
        osc.frequency.exponentialRampToValueAtTime(1200, now + 0.03);

        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

        osc.connect(gain);
        gain.connect(this._masterGain);
        osc.start(now);
        osc.stop(now + 0.06);
    }

    setVolume(val) {
        if (this._masterGain) {
            this._masterGain.gain.value = Math.max(0, Math.min(1, val));
        }
    }
}

// Singleton
export const SoundEngine = new _SoundEngine();

// Auto-init on first user interaction
const _initOnce = () => {
    SoundEngine.init();
    document.removeEventListener('click', _initOnce);
    document.removeEventListener('keydown', _initOnce);
    document.removeEventListener('touchstart', _initOnce);
};
document.addEventListener('click', _initOnce);
document.addEventListener('keydown', _initOnce);
document.addEventListener('touchstart', _initOnce);
