// ============================================================================
// DIM3 — PanelNote Component
// Frosted glass info overlay for pipeline step explanations.
// ============================================================================

export class PanelNote {
    constructor(container, config = {}) {
        this.container = container;
        this.title = config.title || '';
        this.bullets = config.bullets || [];
        this.accentColor = config.accentColor || 'var(--accent)';
        this.el = null;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'panel-note';

        Object.assign(this.el.style, {
            position: 'absolute',
            top: '24px',
            left: '24px',
            maxWidth: '340px',
            padding: '16px 20px',
            borderRadius: '16px',
            zIndex: '20',
            background: `linear-gradient(135deg, 
                color-mix(in srgb, ${this.accentColor} 10%, rgba(0,0,0,0.45)),
                color-mix(in srgb, ${this.accentColor} 5%, rgba(0,0,0,0.55)))`,
            backdropFilter: 'blur(16px) saturate(1.3)',
            WebkitBackdropFilter: 'blur(16px) saturate(1.3)',
            border: `1px solid color-mix(in srgb, ${this.accentColor} 15%, transparent)`,
            boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
            color: '#e2e8f0',
            fontFamily: 'system-ui, -apple-system, sans-serif',
        });

        const titleEl = document.createElement('div');
        Object.assign(titleEl.style, {
            fontWeight: '800',
            fontSize: '10px',
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            color: this.accentColor,
            marginBottom: '10px',
            lineHeight: '1.4',
        });
        titleEl.textContent = this.title;
        this.el.appendChild(titleEl);

        if (this.bullets.length > 0) {
            const list = document.createElement('ul');
            Object.assign(list.style, {
                margin: '0',
                padding: '0 0 0 14px',
                fontSize: '11px',
                lineHeight: '1.65',
                color: 'rgba(226, 232, 240, 0.85)',
            });
            this.bullets.forEach(text => {
                const li = document.createElement('li');
                li.style.marginBottom = '4px';
                li.textContent = text;
                list.appendChild(li);
            });
            this.el.appendChild(list);
        }

        this.container.style.position = 'relative';
        this.container.appendChild(this.el);
    }

    update(config) {
        if (config.title !== undefined) this.title = config.title;
        if (config.bullets !== undefined) this.bullets = config.bullets;
        if (this.el) this.el.remove();
        this._render();
    }

    dispose() {
        if (this.el) {
            this.el.remove();
            this.el = null;
        }
    }
}
