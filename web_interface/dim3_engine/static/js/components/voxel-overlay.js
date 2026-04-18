// ============================================================================
// DIM3 — VoxelOverlay Component
// Renders transparent instanced cubes over a mesh to show voxel occupancy.
// ============================================================================
import * as THREE from 'three';

export class VoxelOverlay {
    constructor(scene, config = {}) {
        this.scene = scene;
        this.config = {
            color: config.color ?? 0x888888,
            opacity: config.opacity ?? 0.25,
            edgeColor: config.edgeColor ?? 0xd4a017,
            edgeOpacity: config.edgeOpacity ?? 0.12,
            ...config,
        };
        this.instancedMesh = null;
        this.edgeMesh = null;
    }

    /**
     * Build voxel overlay from occupancy data.
     * @param {Object} voxelData - { matrix: 3D bool array flattened, shape: [x,y,z], origin: [x,y,z], pitch: number }
     */
    build(voxelData) {
        this.dispose();

        const { matrix, shape, origin, pitch } = voxelData;
        const [sx, sy, sz] = shape;
        const [ox, oy, oz] = origin;

        // Count occupied voxels
        let count = 0;
        for (let i = 0; i < matrix.length; i++) {
            if (matrix[i]) count++;
        }

        if (count === 0) return;

        // Create instanced mesh for occupied cubes
        const cubeGeo = new THREE.BoxGeometry(pitch * 0.95, pitch * 0.95, pitch * 0.95);
        const cubeMat = new THREE.MeshStandardMaterial({
            color: this.config.color,
            transparent: true,
            opacity: this.config.opacity,
            roughness: 0.8,
            metalness: 0.0,
            side: THREE.FrontSide,
            depthWrite: false,
        });

        this.instancedMesh = new THREE.InstancedMesh(cubeGeo, cubeMat, count);
        this.instancedMesh.renderOrder = 1;

        // Create edge lines for each voxel
        const edgeGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(pitch * 0.95, pitch * 0.95, pitch * 0.95));
        const edgeMat = new THREE.LineBasicMaterial({
            color: this.config.edgeColor,
            transparent: true,
            opacity: this.config.edgeOpacity,
        });

        const dummy = new THREE.Object3D();
        let idx = 0;

        const edgesGroup = new THREE.Group();

        for (let x = 0; x < sx; x++) {
            for (let y = 0; y < sy; y++) {
                for (let z = 0; z < sz; z++) {
                    const flatIdx = x * sy * sz + y * sz + z;
                    if (matrix[flatIdx]) {
                        const px = ox + (x + 0.5) * pitch;
                        const py = oy + (y + 0.5) * pitch;
                        const pz = oz + (z + 0.5) * pitch;

                        dummy.position.set(px, py, pz);
                        dummy.updateMatrix();
                        this.instancedMesh.setMatrix(idx, dummy.matrix);
                        idx++;

                        // Add edge lines (only for a subset to keep perf)
                        if (count < 5000 || Math.random() < 5000 / count) {
                            const edgeLine = new THREE.LineSegments(edgeGeo, edgeMat);
                            edgeLine.position.set(px, py, pz);
                            edgesGroup.add(edgeLine);
                        }
                    }
                }
            }
        }

        this.instancedMesh.instanceMatrix.needsUpdate = true;
        this.scene.add(this.instancedMesh);

        this.edgeMesh = edgesGroup;
        this.scene.add(this.edgeMesh);
    }

    setVisible(visible) {
        if (this.instancedMesh) this.instancedMesh.visible = visible;
        if (this.edgeMesh) this.edgeMesh.visible = visible;
    }

    dispose() {
        if (this.instancedMesh) {
            this.scene.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
            this.instancedMesh = null;
        }
        if (this.edgeMesh) {
            this.scene.remove(this.edgeMesh);
            this.edgeMesh.traverse(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            this.edgeMesh = null;
        }
    }
}
