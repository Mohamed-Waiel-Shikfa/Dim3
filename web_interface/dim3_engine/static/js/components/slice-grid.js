// ============================================================================
// DIM3 — SliceGridViewer Component
// Renders a grid of 2D canvas images from voxel axis slices.
// ============================================================================

export class SliceGridViewer {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            axis: config.axis ?? 'z',         // 'x', 'y', or 'z'
            cellSize: config.cellSize ?? 80,
            gap: config.gap ?? 6,
            filledColor: config.filledColor ?? '#d4a017',
            emptyColor: config.emptyColor ?? 'transparent',
            bgColor: config.bgColor ?? 'rgba(0,0,0,0.3)',
            borderColor: config.borderColor ?? 'rgba(212,160,23,0.15)',
            ...config,
        };
        this.wrapper = null;
        this.canvases = [];
    }

    /**
     * Build the slice grid.
     * @param {Object} voxelData - { matrix: flat array, shape: [x,y,z] }
     */
    build(voxelData) {
        this.dispose();

        const { matrix, shape } = voxelData;
        const [sx, sy, sz] = shape;

        this.wrapper = document.createElement('div');
        Object.assign(this.wrapper.style, {
            width: '100%',
            height: '100%',
            overflowY: 'auto',
            overflowX: 'hidden',
            display: 'flex',
            flexWrap: 'wrap',
            alignContent: 'flex-start',
            justifyContent: 'center',
            gap: `${this.config.gap}px`,
            padding: '60px 20px 20px',
            boxSizing: 'border-box',
        });

        // Hide scrollbar
        this.wrapper.style.scrollbarWidth = 'none';
        this.wrapper.style.msOverflowStyle = 'none';

        let sliceCount, getSlice;

        switch (this.config.axis) {
            case 'z':
                sliceCount = sz;
                getSlice = (idx) => {
                    const slice = [];
                    for (let x = 0; x < sx; x++) {
                        const row = [];
                        for (let y = 0; y < sy; y++) {
                            row.push(matrix[x * sy * sz + y * sz + idx]);
                        }
                        slice.push(row);
                    }
                    return { data: slice, w: sy, h: sx };
                };
                break;
            case 'x':
                sliceCount = sx;
                getSlice = (idx) => {
                    const slice = [];
                    for (let y = 0; y < sy; y++) {
                        const row = [];
                        for (let z = 0; z < sz; z++) {
                            row.push(matrix[idx * sy * sz + y * sz + z]);
                        }
                        slice.push(row);
                    }
                    return { data: slice, w: sz, h: sy };
                };
                break;
            case 'y':
                sliceCount = sy;
                getSlice = (idx) => {
                    const slice = [];
                    for (let x = 0; x < sx; x++) {
                        const row = [];
                        for (let z = 0; z < sz; z++) {
                            row.push(matrix[x * sy * sz + idx * sz + z]);
                        }
                        slice.push(row);
                    }
                    return { data: slice, w: sz, h: sx };
                };
                break;
        }

        for (let i = 0; i < sliceCount; i++) {
            const { data, w, h } = getSlice(i);
            const canvas = this._renderSlice(data, w, h, i);
            this.wrapper.appendChild(canvas);
            this.canvases.push(canvas);
        }

        this.container.appendChild(this.wrapper);
    }

    _renderSlice(data, w, h, index) {
        const cellPx = this.config.cellSize;
        const container = document.createElement('div');
        Object.assign(container.style, {
            position: 'relative',
            borderRadius: '8px',
            overflow: 'hidden',
            border: `1px solid ${this.config.borderColor}`,
            background: this.config.bgColor,
        });

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = `${cellPx}px`;
        canvas.style.height = `${cellPx}px`;
        canvas.style.imageRendering = 'pixelated';
        canvas.style.display = 'block';

        const ctx = canvas.getContext('2d');

        for (let row = 0; row < data.length; row++) {
            for (let col = 0; col < data[row].length; col++) {
                if (data[row][col]) {
                    ctx.fillStyle = this.config.filledColor;
                    ctx.fillRect(col, row, 1, 1);
                }
            }
        }

        container.appendChild(canvas);

        // Slice index label
        const label = document.createElement('div');
        Object.assign(label.style, {
            position: 'absolute',
            bottom: '2px',
            right: '4px',
            fontSize: '8px',
            fontWeight: '700',
            color: 'rgba(212,160,23,0.5)',
            fontFamily: 'monospace',
        });
        label.textContent = index;
        container.appendChild(label);

        return container;
    }

    dispose() {
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }
        this.canvases = [];
    }
}
