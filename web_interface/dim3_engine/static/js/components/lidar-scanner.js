// ============================================================================
// DIM3 — LIDAR Scanner Animation
// Adds a scanning sweep effect when a model is first loaded.
// ============================================================================
import * as THREE from 'three';

export class LidarScanner {
    constructor(viewport, config = {}) {
        this.viewport = viewport;
        this.config = {
            duration: config.duration ?? 2500,
            color: config.color ?? 0xd4a017,
            thickness: config.thickness ?? 0.03,
            rotationSpeed: config.rotationSpeed ?? 0.4,
            onComplete: config.onComplete ?? null,
        };
        this.scanPlane = null;
        this.glowPlane = null;
        this.startTime = null;
        this.animating = false;
    }

    start() {
        if (this.animating) return;
        this.animating = true;
        const scene = this.viewport.getScene();

        // Scan line: a thin plane that sweeps vertically
        const geo = new THREE.PlaneGeometry(6, this.config.thickness);
        const mat = new THREE.MeshBasicMaterial({
            color: this.config.color,
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide,
            depthWrite: false,
        });
        this.scanPlane = new THREE.Mesh(geo, mat);
        this.scanPlane.position.y = 2;
        this.scanPlane.rotation.x = Math.PI / 2;
        scene.add(this.scanPlane);

        // Glow plane behind
        const glowGeo = new THREE.PlaneGeometry(6, 0.3);
        const glowMat = new THREE.MeshBasicMaterial({
            color: this.config.color,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
            depthWrite: false,
        });
        this.glowPlane = new THREE.Mesh(glowGeo, glowMat);
        this.glowPlane.position.y = 2;
        this.glowPlane.rotation.x = Math.PI / 2;
        scene.add(this.glowPlane);

        this.startTime = performance.now();
        this._tick();
    }

    _tick() {
        if (!this.animating) return;
        const elapsed = performance.now() - this.startTime;
        const progress = Math.min(elapsed / this.config.duration, 1);

        // Sweep from top to bottom
        const yStart = 2.0;
        const yEnd = -2.0;
        const y = yStart + (yEnd - yStart) * this._easeInOutCubic(progress);

        this.scanPlane.position.y = y;
        this.glowPlane.position.y = y;

        // Slow rotation
        const rot = progress * Math.PI * 2 * this.config.rotationSpeed;
        this.scanPlane.rotation.z = rot;
        this.glowPlane.rotation.z = rot;

        // Fade out in last 20%
        if (progress > 0.8) {
            const fadeProgress = (progress - 0.8) / 0.2;
            this.scanPlane.material.opacity = 0.9 * (1 - fadeProgress);
            this.glowPlane.material.opacity = 0.15 * (1 - fadeProgress);
        }

        if (progress < 1) {
            requestAnimationFrame(() => this._tick());
        } else {
            this._cleanup();
        }
    }

    _easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    _cleanup() {
        const scene = this.viewport.getScene();
        if (this.scanPlane) {
            scene.remove(this.scanPlane);
            this.scanPlane.geometry.dispose();
            this.scanPlane.material.dispose();
            this.scanPlane = null;
        }
        if (this.glowPlane) {
            scene.remove(this.glowPlane);
            this.glowPlane.geometry.dispose();
            this.glowPlane.material.dispose();
            this.glowPlane = null;
        }
        this.animating = false;
        if (this.config.onComplete) this.config.onComplete();
    }

    stop() {
        this.animating = false;
        this._cleanup();
    }
}
