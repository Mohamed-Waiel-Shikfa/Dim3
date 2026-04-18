// ============================================================================
// DIM3 — GraphViewer Component
// Force-directed 2D graph layout using D3.js for GNN pipeline visualization.
// ============================================================================

export class GraphViewer {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            nodeColor: config.nodeColor ?? '#d4a017',
            edgeColor: config.edgeColor ?? 'rgba(212,160,23,0.15)',
            highlightColor: config.highlightColor ?? '#fff',
            bgColor: config.bgColor ?? 'transparent',
            maxNodes: config.maxNodes ?? 2000,
            maxEdges: config.maxEdges ?? 6000,
            ...config,
        };
        this.canvas = null;
        this.ctx = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.transform = { x: 0, y: 0, k: 1 };
        this.hoveredNode = null;
        this.draggingNode = null;
        this.animationId = null;
        this._resizeHandler = null;
    }

    /**
     * Build graph visualization.
     * @param {Object} graphData - { nodes: [[x,y,z],...], edges: [[src,tgt],...], weights: [...] }
     */
    build(graphData) {
        this.dispose();

        let { nodes: rawNodes, edges: rawEdges, weights } = graphData;

        // Subsample if too large
        if (rawNodes.length > this.config.maxNodes) {
            const ratio = this.config.maxNodes / rawNodes.length;
            const kept = new Set();
            for (let i = 0; i < rawNodes.length; i++) {
                if (Math.random() < ratio) kept.add(i);
            }
            const indexMap = {};
            let newIdx = 0;
            rawNodes = rawNodes.filter((_, i) => {
                if (kept.has(i)) { indexMap[i] = newIdx++; return true; }
                return false;
            });
            rawEdges = rawEdges.filter(([s, t]) => kept.has(s) && kept.has(t))
                               .map(([s, t]) => [indexMap[s], indexMap[t]]);
            if (weights) {
                weights = rawEdges.map((_, i) => weights[i] || 1);
            }
        }

        // Cap edges
        if (rawEdges.length > this.config.maxEdges) {
            rawEdges = rawEdges.slice(0, this.config.maxEdges);
        }

        // Compute degree for node sizing
        const degree = new Array(rawNodes.length).fill(0);
        for (const [s, t] of rawEdges) {
            if (s < degree.length) degree[s]++;
            if (t < degree.length) degree[t]++;
        }
        const maxDeg = Math.max(...degree, 1);

        this.nodes = rawNodes.map((pos, i) => ({
            id: i,
            x: pos[0] * 100 + (Math.random() - 0.5) * 10,
            y: pos[1] * 100 + (Math.random() - 0.5) * 10,
            degree: degree[i],
            radius: 2 + (degree[i] / maxDeg) * 4,
        }));

        this.links = rawEdges.map(([s, t], i) => ({
            source: s,
            target: t,
            weight: weights ? weights[i] : 1,
        }));

        this._createCanvas();
        this._setupInteraction();
        this._startSimulation();
    }

    _createCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.display = 'block';
        this.canvas.style.cursor = 'grab';
        this.container.appendChild(this.canvas);

        this._resizeCanvas();
        this._resizeHandler = () => this._resizeCanvas();
        window.addEventListener('resize', this._resizeHandler);

        this.ctx = this.canvas.getContext('2d');
    }

    _resizeCanvas() {
        if (!this.canvas) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const dpr = Math.min(window.devicePixelRatio, 2);
        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;
        if (this.ctx) {
            this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        this._draw();
    }

    _setupInteraction() {
        let dragStartX, dragStartY;
        let isPanning = false;

        this.canvas.addEventListener('mousedown', (e) => {
            const [mx, my] = this._screenToWorld(e.offsetX, e.offsetY);
            const node = this._findNode(mx, my);
            if (node) {
                this.draggingNode = node;
                node.fx = node.x;
                node.fy = node.y;
                this.canvas.style.cursor = 'grabbing';
            } else {
                isPanning = true;
                dragStartX = e.offsetX;
                dragStartY = e.offsetY;
                this.canvas.style.cursor = 'grabbing';
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (this.draggingNode) {
                const [mx, my] = this._screenToWorld(e.offsetX, e.offsetY);
                this.draggingNode.fx = mx;
                this.draggingNode.fy = my;
                if (this.simulation) this.simulation.alpha(0.3).restart();
            } else if (isPanning) {
                this.transform.x += e.offsetX - dragStartX;
                this.transform.y += e.offsetY - dragStartY;
                dragStartX = e.offsetX;
                dragStartY = e.offsetY;
                this._draw();
            } else {
                const [mx, my] = this._screenToWorld(e.offsetX, e.offsetY);
                const node = this._findNode(mx, my);
                this.hoveredNode = node;
                this.canvas.style.cursor = node ? 'pointer' : 'grab';
                this._draw();
            }
        });

        this.canvas.addEventListener('mouseup', () => {
            if (this.draggingNode) {
                this.draggingNode.fx = null;
                this.draggingNode.fy = null;
                this.draggingNode = null;
            }
            isPanning = false;
            this.canvas.style.cursor = 'grab';
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const scale = e.deltaY > 0 ? 0.9 : 1.1;
            this.transform.k *= scale;
            this.transform.x = e.offsetX - (e.offsetX - this.transform.x) * scale;
            this.transform.y = e.offsetY - (e.offsetY - this.transform.y) * scale;
            this._draw();
        });
    }

    _screenToWorld(sx, sy) {
        return [
            (sx - this.transform.x) / this.transform.k,
            (sy - this.transform.y) / this.transform.k,
        ];
    }

    _worldToScreen(wx, wy) {
        return [
            wx * this.transform.k + this.transform.x,
            wy * this.transform.k + this.transform.y,
        ];
    }

    _findNode(wx, wy) {
        for (const node of this.nodes) {
            const dx = node.x - wx;
            const dy = node.y - wy;
            if (dx * dx + dy * dy < (node.radius + 3) * (node.radius + 3)) {
                return node;
            }
        }
        return null;
    }

    _startSimulation() {
        // Simple force simulation (no D3 dependency)
        const centerX = this.canvas.width / (2 * Math.min(window.devicePixelRatio, 2));
        const centerY = this.canvas.height / (2 * Math.min(window.devicePixelRatio, 2));

        this.transform.x = centerX;
        this.transform.y = centerY;

        let alpha = 1;
        const decay = 0.995;

        const tick = () => {
            if (alpha < 0.001) {
                this._draw();
                return;
            }

            // Simple force: repulsion between nodes
            for (let i = 0; i < this.nodes.length; i++) {
                for (let j = i + 1; j < Math.min(this.nodes.length, i + 50); j++) {
                    const ni = this.nodes[i], nj = this.nodes[j];
                    let dx = nj.x - ni.x;
                    let dy = nj.y - ni.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    const force = -30 / (dist * dist) * alpha;
                    dx *= force; dy *= force;
                    if (!ni.fx) { ni.x -= dx; ni.y -= dy; }
                    if (!nj.fx) { nj.x += dx; nj.y += dy; }
                }
            }

            // Attraction along edges
            for (const link of this.links) {
                const s = this.nodes[link.source];
                const t = this.nodes[link.target];
                if (!s || !t) continue;
                let dx = t.x - s.x;
                let dy = t.y - s.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = (dist - 20) * 0.01 * alpha;
                dx = (dx / dist) * force;
                dy = (dy / dist) * force;
                if (!s.fx) { s.x += dx; s.y += dy; }
                if (!t.fx) { t.x -= dx; t.y -= dy; }
            }

            // Center gravity
            for (const n of this.nodes) {
                if (!n.fx) {
                    n.x += (0 - n.x) * 0.001 * alpha;
                    n.y += (0 - n.y) * 0.001 * alpha;
                }
            }

            alpha *= decay;
            this._draw();
            this.animationId = requestAnimationFrame(tick);
        };

        this.simulation = { alpha: (a) => { alpha = a; return this.simulation; }, restart: () => { tick(); return this.simulation; } };
        tick();
    }

    _draw() {
        if (!this.ctx || !this.canvas) return;
        const w = this.canvas.width / Math.min(window.devicePixelRatio, 2);
        const h = this.canvas.height / Math.min(window.devicePixelRatio, 2);
        this.ctx.clearRect(0, 0, w, h);

        const highlightedNeighbors = new Set();
        if (this.hoveredNode) {
            for (const link of this.links) {
                if (link.source === this.hoveredNode.id) highlightedNeighbors.add(link.target);
                if (link.target === this.hoveredNode.id) highlightedNeighbors.add(link.source);
            }
        }

        // Draw edges
        this.ctx.lineWidth = 0.5;
        for (const link of this.links) {
            const s = this.nodes[link.source];
            const t = this.nodes[link.target];
            if (!s || !t) continue;

            const [sx, sy] = this._worldToScreen(s.x, s.y);
            const [tx, ty] = this._worldToScreen(t.x, t.y);

            const isHighlight = this.hoveredNode &&
                (link.source === this.hoveredNode.id || link.target === this.hoveredNode.id);

            this.ctx.strokeStyle = isHighlight ? 'rgba(255,255,255,0.5)' : this.config.edgeColor;
            this.ctx.lineWidth = isHighlight ? 1 : 0.5;
            this.ctx.beginPath();
            this.ctx.moveTo(sx, sy);
            this.ctx.lineTo(tx, ty);
            this.ctx.stroke();
        }

        // Draw nodes
        for (const node of this.nodes) {
            const [sx, sy] = this._worldToScreen(node.x, node.y);
            const isHovered = this.hoveredNode && node.id === this.hoveredNode.id;
            const isNeighbor = highlightedNeighbors.has(node.id);

            this.ctx.beginPath();
            this.ctx.arc(sx, sy, node.radius * this.transform.k, 0, Math.PI * 2);

            if (isHovered) {
                this.ctx.fillStyle = this.config.highlightColor;
                this.ctx.shadowColor = this.config.nodeColor;
                this.ctx.shadowBlur = 10;
            } else if (isNeighbor) {
                this.ctx.fillStyle = this.config.highlightColor;
                this.ctx.shadowBlur = 0;
            } else {
                this.ctx.fillStyle = this.config.nodeColor;
                this.ctx.shadowBlur = 0;
            }
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }
    }

    dispose() {
        if (this.animationId) cancelAnimationFrame(this.animationId);
        if (this._resizeHandler) window.removeEventListener('resize', this._resizeHandler);
        if (this.canvas) {
            this.canvas.remove();
            this.canvas = null;
        }
        this.ctx = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
    }
}
