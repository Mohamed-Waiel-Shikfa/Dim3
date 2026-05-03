// ============================================================================
// DIM3 — VoxelOverlay Component
// Renders the full voxel grid (orange edges) with filled cells in transparent blue.
// ============================================================================
import * as THREE from 'three';

export class VoxelOverlay {
    constructor(scene, config = {}) {
        this.scene = scene;
        this.config = {
            filledColor: config.filledColor ?? 0x3a9bdc,
            filledOpacity: config.filledOpacity ?? 0.35,
            gridColor: config.gridColor ?? 0xd4a017,
            gridOpacity: config.gridOpacity ?? 0.55,
            ...config,
        };
        this.objects = [];
    }

    build(voxelData) {
        this.dispose();
        const { matrix, shape, origin, pitch } = voxelData;
        const [sx, sy, sz] = shape;
        const [ox, oy, oz] = origin;

        // --- Filled voxels: transparent blue instanced mesh ---
        let filledCount = 0;
        for (let i = 0; i < matrix.length; i++) if (matrix[i]) filledCount++;

        if (filledCount > 0) {
            const cubeGeo = new THREE.BoxGeometry(pitch * 0.98, pitch * 0.98, pitch * 0.98);
            const cubeMat = new THREE.MeshStandardMaterial({
                color: this.config.filledColor,
                transparent: true,
                opacity: this.config.filledOpacity,
                depthWrite: false,
                side: THREE.DoubleSide,
            });
            const instMesh = new THREE.InstancedMesh(cubeGeo, cubeMat, filledCount);
            instMesh.renderOrder = 2;
            const dummy = new THREE.Object3D();
            let idx = 0;
            for (let x = 0; x < sx; x++) {
                for (let y = 0; y < sy; y++) {
                    for (let z = 0; z < sz; z++) {
                        if (matrix[x * sy * sz + y * sz + z]) {
                            dummy.position.set(ox + (x + 0.5) * pitch, oy + (y + 0.5) * pitch, oz + (z + 0.5) * pitch);
                            dummy.updateMatrix();
                            instMesh.setMatrix(idx++, dummy.matrix);
                        }
                    }
                }
            }
            instMesh.instanceMatrix.needsUpdate = true;
            this.scene.add(instMesh);
            this.objects.push(instMesh);
        }

        // --- Full grid: orange edge lines for EVERY cell ---
        const edgeGeo = new THREE.BoxGeometry(pitch, pitch, pitch);
        const edgesTemplate = new THREE.EdgesGeometry(edgeGeo);
        const edgeMat = new THREE.LineBasicMaterial({
            color: this.config.gridColor,
            transparent: true,
            opacity: this.config.gridOpacity,
        });

        // Merge all grid edges into one BufferGeometry for performance
        const allPositions = [];
        const posAttr = edgesTemplate.attributes.position;
        for (let x = 0; x < sx; x++) {
            for (let y = 0; y < sy; y++) {
                for (let z = 0; z < sz; z++) {
                    const cx = ox + (x + 0.5) * pitch;
                    const cy = oy + (y + 0.5) * pitch;
                    const cz = oz + (z + 0.5) * pitch;
                    for (let i = 0; i < posAttr.count; i++) {
                        allPositions.push(
                            posAttr.getX(i) + cx,
                            posAttr.getY(i) + cy,
                            posAttr.getZ(i) + cz
                        );
                    }
                }
            }
        }
        const mergedGeo = new THREE.BufferGeometry();
        mergedGeo.setAttribute('position', new THREE.Float32BufferAttribute(allPositions, 3));
        const gridLines = new THREE.LineSegments(mergedGeo, edgeMat);
        gridLines.renderOrder = 3;
        this.scene.add(gridLines);
        this.objects.push(gridLines);

        edgesTemplate.dispose();
        edgeGeo.dispose();
    }

    dispose() {
        for (const obj of this.objects) {
            this.scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }
        this.objects = [];
    }
}
