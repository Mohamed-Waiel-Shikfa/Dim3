import * as THREE from 'three';

export function samplePointCloud(mesh, numPoints, method = 'fps') {
    // Basic random surface sampling using triangle areas
    const geometry = mesh.geometry;
    geometry.computeVertexNormals();
    const pos = geometry.attributes.position;
    const index = geometry.index;
    
    let totalArea = 0;
    const cumulativeAreas = [];
    const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3();
    const cb = new THREE.Vector3(), ab = new THREE.Vector3();
    
    const faceCount = index ? index.count / 3 : pos.count / 3;
    for (let i = 0; i < faceCount; i++) {
        const a = index ? index.getX(i*3) : i*3;
        const b = index ? index.getX(i*3+1) : i*3+1;
        const c = index ? index.getX(i*3+2) : i*3+2;
        vA.fromBufferAttribute(pos, a);
        vB.fromBufferAttribute(pos, b);
        vC.fromBufferAttribute(pos, c);
        cb.subVectors(vC, vB);
        ab.subVectors(vA, vB);
        cb.cross(ab);
        const area = cb.length() * 0.5;
        totalArea += area;
        cumulativeAreas.push(totalArea);
    }
    
    const points = [];
    for (let i = 0; i < numPoints; i++) {
        let r = Math.random() * totalArea;
        let low = 0, high = faceCount - 1;
        while (low < high) {
            let mid = Math.floor((low + high) / 2);
            if (r < cumulativeAreas[mid]) high = mid;
            else low = mid + 1;
        }
        const faceIdx = low;
        const a = index ? index.getX(faceIdx*3) : faceIdx*3;
        const b = index ? index.getX(faceIdx*3+1) : faceIdx*3+1;
        const c = index ? index.getX(faceIdx*3+2) : faceIdx*3+2;
        vA.fromBufferAttribute(pos, a);
        vB.fromBufferAttribute(pos, b);
        vC.fromBufferAttribute(pos, c);
        
        let r1 = Math.random(), r2 = Math.random();
        if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
        const r3 = 1 - r1 - r2;
        
        const pt = new THREE.Vector3()
            .addScaledVector(vA, r1)
            .addScaledVector(vB, r2)
            .addScaledVector(vC, r3);
            
        mesh.localToWorld(pt);
        points.push([pt.x, pt.y, pt.z]);
    }
    return { points, count: points.length, method: method, method_description: "Uniform surface random sampling (client-side)" };
}

export function computeVoxelization(mesh, gridSize) {
    const box = new THREE.Box3().setFromObject(mesh);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    let pitch = maxDim / gridSize;
    if (pitch === 0) pitch = 1.0;
    
    const matrix = new Array(gridSize * gridSize * gridSize).fill(0);
    const origin = [box.min.x, box.min.y, box.min.z];
    
    // Quick sampling of vertices, not fully robust but sufficient for front-end demonstration of structure
    // High-quality front-end voxelization requires 3D flood fill or BVH. 
    // We'll map the mesh vertices to the grid to populate the shell rapidly.
    const pos = mesh.geometry.attributes.position;
    const v = new THREE.Vector3();
    for (let i=0; i<pos.count; i++) {
        v.fromBufferAttribute(pos, i);
        mesh.localToWorld(v);
        let gx = Math.floor((v.x - origin[0]) / pitch);
        let gy = Math.floor((v.y - origin[1]) / pitch);
        let gz = Math.floor((v.z - origin[2]) / pitch);
        gx = Math.max(0, Math.min(gx, gridSize-1));
        gy = Math.max(0, Math.min(gy, gridSize-1));
        gz = Math.max(0, Math.min(gz, gridSize-1));
        matrix[gx * gridSize * gridSize + gy * gridSize + gz] = 1;
    }
    
    const filledCount = matrix.filter(x => x===1).length;
    return {
        matrix, shape: [gridSize, gridSize, gridSize], origin, pitch,
        filled_count: filledCount, total_count: gridSize*gridSize*gridSize,
        density: filledCount / (gridSize**3)
    };
}

export function extractGraph(mesh) {
    const geo = new THREE.WireframeGeometry(mesh.geometry);
    const pos = geo.attributes.position;
    const nodesMap = new Map();
    const nodes = [];
    const edges = [];
    
    function getNodeId(x, y, z) {
        const key = `${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)}`;
        if (!nodesMap.has(key)) {
            nodesMap.set(key, nodes.length);
            nodes.push([x, y, z]);
        }
        return nodesMap.get(key);
    }
    
    for (let i = 0; i < pos.count; i += 2) {
        const v1x = pos.getX(i), v1y = pos.getY(i), v1z = pos.getZ(i);
        const v2x = pos.getX(i+1), v2y = pos.getY(i+1), v2z = pos.getZ(i+1);
        const pt1 = new THREE.Vector3(v1x, v1y, v1z);
        const pt2 = new THREE.Vector3(v2x, v2y, v2z);
        mesh.localToWorld(pt1);
        mesh.localToWorld(pt2);
        
        const n1 = getNodeId(pt1.x, pt1.y, pt1.z);
        const n2 = getNodeId(pt2.x, pt2.y, pt2.z);
        if (n1 !== n2) edges.push([n1, n2]);
    }
    
    return { nodes, edges, node_count: nodes.length, edge_count: edges.length };
}
