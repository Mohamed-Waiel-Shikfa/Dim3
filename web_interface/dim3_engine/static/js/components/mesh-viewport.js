// ============================================================================
// DIM3 — MeshViewport Component
// A reusable Three.js 3D viewer for OBJ meshes, point clouds, and wireframes.
// ============================================================================
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

export class MeshViewport {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            wireframe: config.wireframe ?? false,
            wireframeOpacity: config.wireframeOpacity ?? 0.08,
            materialColor: config.materialColor ?? 0x909090,
            backgroundColor: config.backgroundColor ?? null,
            accentColor: config.accentColor ?? 0xd4a017,
            showGrid: config.showGrid ?? false,
            autoRotate: config.autoRotate ?? false,
            autoRotateSpeed: config.autoRotateSpeed ?? 0.5,
            ...config
        };
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;
        this.wireframeOverlay = null;
        this.running = false;
        this.animationId = null;
        this._resizeHandler = null;
        this._init();
    }

    _init() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        this.scene = new THREE.Scene();
        if (this.config.backgroundColor !== null) {
            this.scene.background = new THREE.Color(this.config.backgroundColor);
        }

        this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 500);
        this.camera.position.set(3, 2.5, 3);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.setClearColor(0x000000, 0);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.06;
        this.controls.autoRotate = this.config.autoRotate;
        this.controls.autoRotateSpeed = this.config.autoRotateSpeed;

        this._setupLighting();

        if (this.config.showGrid) {
            const accent = new THREE.Color(this.config.accentColor);
            const gridHelper = new THREE.GridHelper(4, 20, accent, accent);
            gridHelper.material.opacity = 0.07;
            gridHelper.material.transparent = true;
            this.scene.add(gridHelper);
        }

        this._resizeHandler = () => this.resize();
        window.addEventListener('resize', this._resizeHandler);

        this.running = true;
        this._animate();
    }

    _setupLighting() {
        const ambient = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambient);

        const dir1 = new THREE.DirectionalLight(0xffffff, 1.2);
        dir1.position.set(5, 10, 7);
        dir1.castShadow = true;
        this.scene.add(dir1);

        const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dir2.position.set(-5, 3, -5);
        this.scene.add(dir2);

        const rim = new THREE.DirectionalLight(new THREE.Color(this.config.accentColor), 0.15);
        rim.position.set(0, -5, -5);
        this.scene.add(rim);
    }

    async loadOBJ(url) {
        return new Promise((resolve, reject) => {
            const loader = new OBJLoader();
            loader.load(url, (obj) => {
                this._clearModel();
                this.model = obj;
                this._applyMaterial(obj);
                this._fitToView(obj);
                this.scene.add(obj);
                if (this.config.wireframe) {
                    this._addWireframeOverlay(obj);
                }
                resolve(obj);
            }, undefined, (err) => {
                console.error('MeshViewport: Failed to load OBJ', err);
                reject(err);
            });
        });
    }

    loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(file);
            const loader = new OBJLoader();
            loader.load(url, (obj) => {
                this._clearModel();
                this.model = obj;
                this._applyMaterial(obj);
                this._fitToView(obj);
                this.scene.add(obj);
                if (this.config.wireframe) {
                    this._addWireframeOverlay(obj);
                }
                URL.revokeObjectURL(url);
                resolve(obj);
            }, undefined, (err) => {
                URL.revokeObjectURL(url);
                reject(err);
            });
        });
    }

    loadGeometry(geometry, material = null) {
        this._clearModel();
        const mat = material || new THREE.MeshStandardMaterial({
            color: this.config.materialColor,
            roughness: 0.6,
            metalness: 0.1,
        });
        const mesh = new THREE.Mesh(geometry, mat);
        this.model = mesh;
        this._fitToView(mesh);
        this.scene.add(mesh);
        return mesh;
    }

    addToScene(object3d) {
        this.scene.add(object3d);
    }

    removeFromScene(object3d) {
        this.scene.remove(object3d);
    }

    _applyMaterial(obj) {
        const mat = new THREE.MeshStandardMaterial({
            color: this.config.materialColor,
            roughness: 0.55,
            metalness: 0.05,
            side: THREE.DoubleSide,
        });
        const lineMat = new THREE.LineBasicMaterial({
            color: this.config.accentColor,
            transparent: true,
            opacity: 0.75,
        });
        obj.traverse((child) => {
            if (child.isMesh) {
                child.material = mat;
                child.castShadow = true;
                child.receiveShadow = true;
            } else if (child.isLine) {
                child.material = lineMat;
            }
        });
    }

    _addWireframeOverlay(obj) {
        if (this.wireframeOverlay) {
            this.scene.remove(this.wireframeOverlay);
            this.wireframeOverlay = null;
        }
        const wireGroup = new THREE.Group();
        obj.traverse((child) => {
            if (child.isMesh) {
                const wireGeo = new THREE.WireframeGeometry(child.geometry);
                const wireMat = new THREE.LineBasicMaterial({
                    color: this.config.accentColor,
                    opacity: this.config.wireframeOpacity,
                    transparent: true,
                });
                const wireLines = new THREE.LineSegments(wireGeo, wireMat);
                wireLines.position.copy(child.position);
                wireLines.rotation.copy(child.rotation);
                wireLines.scale.copy(child.scale);
                wireGroup.add(wireLines);
            }
        });
        wireGroup.position.copy(obj.position);
        wireGroup.rotation.copy(obj.rotation);
        wireGroup.scale.copy(obj.scale);
        this.wireframeOverlay = wireGroup;
        this.scene.add(wireGroup);
    }

    _fitToView(obj) {
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim === 0) return;

        const scale = 2.5 / maxDim;
        obj.scale.setScalar(scale);
        box.setFromObject(obj);
        box.getCenter(center);
        obj.position.sub(center);

        // After scaling, the model is always 2.5 units — use a fixed readable distance
        this.camera.position.set(4, 3, 4);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    _clearModel() {
        if (this.model) {
            this.scene.remove(this.model);
            this.model.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
            this.model = null;
        }
        // Dispose any extra scene objects (line segments, points) that were added externally
        // Those are tracked via the scene; caller is responsible for removing references.
        if (this.wireframeOverlay) {
            this.scene.remove(this.wireframeOverlay);
            this.wireframeOverlay = null;
        }
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
        this._clearModel();
        this.controls.dispose();
        this.renderer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }

    getScene() { return this.scene; }
    getCamera() { return this.camera; }
    getModel() { return this.model; }
}
