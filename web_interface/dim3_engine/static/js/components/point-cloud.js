// ============================================================================
// DIM3 — PointCloudViewer Component
// Renders point clouds using THREE.Points with glow effect.
// ============================================================================
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class PointCloudViewer {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            pointSize: config.pointSize ?? 3.0,
            pointColor: config.pointColor ?? 0xd4a017,
            backgroundColor: config.backgroundColor ?? null,
            accentColor: config.accentColor ?? 0xd4a017,
            ...config,
        };
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.points = null;
        this.running = false;
        this.animationId = null;
        this._resizeHandler = null;
        this._init();
    }

    _init() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.scene = new THREE.Scene();

        this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 500);
        this.camera.position.set(3, 2, 3);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x000000, 0);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.06;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 0.3;

        // Subtle ambient light
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));

        this._resizeHandler = () => this.resize();
        window.addEventListener('resize', this._resizeHandler);

        this.running = true;
        this._animate();
    }

    /**
     * Load points from array of [x,y,z] coordinates.
     * @param {Array} pointsArray - Flat Float32Array or array of [x,y,z]
     */
    loadPoints(pointsArray) {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        let positions;
        if (pointsArray instanceof Float32Array) {
            positions = pointsArray;
        } else {
            positions = new Float32Array(pointsArray.length * 3);
            for (let i = 0; i < pointsArray.length; i++) {
                positions[i * 3] = pointsArray[i][0];
                positions[i * 3 + 1] = pointsArray[i][1];
                positions[i * 3 + 2] = pointsArray[i][2];
            }
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Compute colors based on height (y-coordinate) for visual richness
        const colors = new Float32Array(positions.length);
        let minY = Infinity, maxY = -Infinity;
        for (let i = 1; i < positions.length; i += 3) {
            minY = Math.min(minY, positions[i]);
            maxY = Math.max(maxY, positions[i]);
        }
        const rangeY = maxY - minY || 1;
        const baseColor = new THREE.Color(this.config.pointColor);
        const topColor = new THREE.Color(0xffffff);

        for (let i = 0; i < positions.length / 3; i++) {
            const t = (positions[i * 3 + 1] - minY) / rangeY;
            const c = baseColor.clone().lerp(topColor, t * 0.3);
            colors[i * 3] = c.r;
            colors[i * 3 + 1] = c.g;
            colors[i * 3 + 2] = c.b;
        }
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const mat = new THREE.PointsMaterial({
            size: this.config.pointSize,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9,
        });

        this.points = new THREE.Points(geo, mat);

        // Center and scale
        geo.computeBoundingBox();
        const box = geo.boundingBox;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const scale = 2.5 / maxDim;
        this.points.scale.setScalar(scale);
        this.points.position.set(-center.x * scale, -center.y * scale, -center.z * scale);

        this.scene.add(this.points);

        this.camera.position.set(3, 2, 3);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w === 0 || h === 0) return;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    pause() { this.running = false; }
    resume() {
        if (!this.running) {
            this.running = true;
            this._animate();
        }
    }

    _animate() {
        if (!this.running) return;
        this.animationId = requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        this.running = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
        window.removeEventListener('resize', this._resizeHandler);
        if (this.points) {
            this.points.geometry.dispose();
            this.points.material.dispose();
        }
        this.controls.dispose();
        this.renderer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }
}
