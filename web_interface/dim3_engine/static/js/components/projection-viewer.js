// ============================================================================
// DIM3 — ProjectionViewer Component
// Displays 3 axis-aligned projections of a point cloud side by side.
// ============================================================================

export class ProjectionViewer {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            dotSize: config.dotSize ?? 2,
            dotColor: config.dotColor ?? '#d4a017',
            bgColor: config.bgColor ?? 'rgba(0,0,0,0.35)',
            borderColor: config.borderColor ?? 'rgba(212,160,23,0.15)',
            labelColor: config.labelColor ?? 'rgba(212,160,23,0.7)',
            ...config,
        };
        this.wrapper = null;
        this.canvases = [];
    }

    /**
     * Render projections from point data.
     * @param {Array} points - Array of [x,y,z]
     */
    build(points) {
        this.dispose();

        this.wrapper = document.createElement('div');
        Object.assign(this.wrapper.style, {
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            padding: '60px 24px 24px',
            boxSizing: 'border-box',
        });

        const projections = [
            { label: 'XY (Top)', extract: (p) => [p[0], p[1]] },
            { label: 'XZ (Front)', extract: (p) => [p[0], p[2]] },
            { label: 'YZ (Side)', extract: (p) => [p[1], p[2]] },
        ];

        for (const proj of projections) {
            const panel = this._renderProjection(points, proj.extract, proj.label);
            this.wrapper.appendChild(panel);
        }

        this.container.appendChild(this.wrapper);
    }

    _renderProjection(points, extractFn, label) {
        const panel = document.createElement('div');
        Object.assign(panel.style, {
            flex: '1',
            maxWidth: '33%',
            aspectRatio: '1',
            position: 'relative',
            borderRadius: '16px',
            overflow: 'hidden',
            border: `1px solid ${this.config.borderColor}`,
            background: this.config.bgColor,
        });

        const canvas = document.createElement('canvas');
        const size = 400;
        canvas.width = size;
        canvas.height = size;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        panel.appendChild(canvas);

        const ctx = canvas.getContext('2d');

        // Extract 2D coordinates
        const coords = points.map(extractFn);

        // Find bounds
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const [x, y] of coords) {
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        const padding = 20;

        // Draw points
        ctx.fillStyle = this.config.dotColor;
        for (const [x, y] of coords) {
            const px = padding + ((x - minX) / rangeX) * (size - 2 * padding);
            const py = padding + ((y - minY) / rangeY) * (size - 2 * padding);
            ctx.beginPath();
            ctx.arc(px, size - py, this.config.dotSize, 0, Math.PI * 2);
            ctx.fill();
        }

        // Label
        const labelEl = document.createElement('div');
        Object.assign(labelEl.style, {
            position: 'absolute',
            bottom: '8px',
            left: '0',
            width: '100%',
            textAlign: 'center',
            fontSize: '10px',
            fontWeight: '700',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: this.config.labelColor,
            fontFamily: 'system-ui, sans-serif',
        });
        labelEl.textContent = label;
        panel.appendChild(labelEl);

        this.canvases.push(canvas);
        return panel;
    }

    dispose() {
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }
        this.canvases = [];
    }
}
