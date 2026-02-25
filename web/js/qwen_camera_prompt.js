/**
 * Qwen Camera Prompt — 3D viewport for yaple-nodes
 *
 * Interface inspired by the linoyts/Qwen-Image-Edit-Angles HuggingFace space.
 * Three colored draggable handles control the camera position:
 *
 *   Green  (●) — drag L/R  → azimuth   (8 stops, 45° apart, full 360°)
 *   Pink   (●) — drag U/D  → elevation  (4 stops: −30°, 0°, 30°, 60°)
 *   Orange (●) — drag U/D  → distance   (4 stops: close-up, forward, medium/neutral, wide)
 *
 * Dragging outside a handle orbits the viewer camera for 3D inspection.
 * The generated prompt is shown live below the viewport.
 */

import { app } from "../../../scripts/app.js";

// ─── Camera parameter tables (must match qwen_camera_prompt.py) ──────────────

const AZIMUTHS = [
    { deg:   0, label: "front view" },
    { deg:  45, label: "front-right quarter view" },
    { deg:  90, label: "right side view" },
    { deg: 135, label: "back-right quarter view" },
    { deg: 180, label: "back view" },
    { deg: 225, label: "back-left quarter view" },
    { deg: 270, label: "left side view" },
    { deg: 315, label: "front-left quarter view" },
];

const ELEVATIONS = [
    { deg: -30, label: "low-angle shot" },
    { deg:   0, label: "eye-level shot" },
    { deg:  30, label: "elevated shot" },
    { deg:  60, label: "high-angle shot" },
];

const DISTANCES = [
    { label: "close-up",    r: 2.0 },  // dist=0  (dx8152: close-up)
    { label: "medium shot", r: 2.7 },  // dist=1  (dx8152: forward)
    { label: "medium shot", r: 3.5 },  // dist=2  (dx8152: neutral — default)
    { label: "wide shot",   r: 4.8 },  // dist=3  (dx8152: backward)
];

// ─── Colors ──────────────────────────────────────────────────────────────────

const C_GREEN  = 0x4caf50;
const C_PINK   = 0xe91e8c;
const C_ORANGE = 0xff9800;
const C_CAM    = 0x00bcd4;
const C_BG     = 0x111827;

// ─── Scene layout constants (match HuggingFace space proportions) ─────────────
const RING_RADIUS  = 3.5;   // equatorial azimuth ring radius
const TILT_RADIUS  = 1.6;   // vertical tilt arc radius (matches HF space)
const CENTER_Y     = 0.75;  // y-offset for the camera system (chest height)

// ─── Three.js loader (singleton) ─────────────────────────────────────────────

const THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
let THREE = null;
let threePromise = null;

function loadThree() {
    if (threePromise) return threePromise;
    threePromise = new Promise((resolve, reject) => {
        if (window.THREE) { THREE = window.THREE; return resolve(THREE); }
        const s = document.createElement("script");
        s.src = THREE_CDN;
        s.onload  = () => { THREE = window.THREE; resolve(THREE); };
        s.onerror = () => { threePromise = null; reject(new Error("Three.js CDN load failed")); };
        document.head.appendChild(s);
    });
    return threePromise;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const toRad = deg => (deg * Math.PI) / 180;

/** 3D position of the camera marker for given indices. */
function markerPos(azIdx, elIdx, distIdx) {
    const az = toRad(AZIMUTHS[azIdx].deg);
    const el = toRad(ELEVATIONS[elIdx].deg);
    const r  = DISTANCES[distIdx].r;
    return new THREE.Vector3(
        r * Math.cos(el) * Math.sin(az),
        r * Math.sin(el),
        r * Math.cos(el) * Math.cos(az)
    );
}

// ─── Widget ──────────────────────────────────────────────────────────────────

class QwenCameraWidget {
    constructor(node) {
        this.node = node;

        // Current selection (indices into the tables above)
        this.azIdx   = 0;  // front view
        this.elIdx   = 1;  // eye-level
        this.distIdx = 2;  // medium/neutral
        this.fmt     = "fal";  // "fal" | "dx8152"

        // Three.js
        this.scene    = null;
        this.viewCam  = null;  // viewer/orbit camera (not the subject camera)
        this.renderer = null;
        this.raycaster = null;
        this.mouse2D   = null;
        this.animId    = null;
        this.resizeObs = null;

        // Scene objects
        this.cameraGroup   = null;  // box body + lens group
        this.cameraLinePts = null;  // BufferGeometry points for the orange distance line
        this.handleGreen   = null;
        this.handlePink    = null;
        this.handleOrange  = null;

        // DOM
        this.container   = null;
        this.viewport    = null;
        this.canvas      = null;
        this.promptLabel = null;

        // Drag state
        this.dragging       = null;  // null | "green" | "pink" | "orange" | "orbit"
        this.dragStartMouse = { x: 0, y: 0 };
        this.dragStartAz    = 0;
        this.dragStartEl    = 0;
        this.dragStartDist  = 0;
        this.hoveredHandle  = null;

        // Bound event handlers (stored so they can be removed)
        this._boundMove = this._onMove.bind(this);
        this._boundUp   = this._onUp.bind(this);
    }

    async initialize() {
        try {
            await loadThree();
        } catch (e) {
            console.error("[QwenCameraPrompt] Failed to load Three.js:", e);
            return false;
        }

        this._buildDOM();
        this._initScene();
        this._buildSubject();
        this._buildGuides();
        this._buildCameraMarker();
        this._buildHandles();
        this._setupEvents();
        this._startAnimation();
        this._loadFromWidget();   // also calls _updateFmtButtons + _refreshLabel
        this._updateFmtButtons(); // ensure initial button state even if no saved state
        return true;
    }

    // ── DOM ────────────────────────────────────────────────────────────────────

    _buildDOM() {
        this.container = document.createElement("div");
        this.container.style.cssText = [
            "width:100%",
            "height:100%",           // fill the DOM widget's allocated height
            "display:flex",
            "flex-direction:column",
            "background:#" + C_BG.toString(16).padStart(6, "0"),
            "border-radius:4px",
            "border:1px solid #1e3a5f",
            "overflow:hidden",
            "box-sizing:border-box",
        ].join(";");

        // 3D viewport — flex:1 fills all height left after the info bar.
        // The canvas is kept square via the ResizeObserver and centered with
        // absolute positioning inside the viewport div.
        this.viewport = document.createElement("div");
        this.viewport.style.cssText = [
            "flex:1",
            "position:relative",
            "overflow:hidden",
            "min-height:200px",
        ].join(";");

        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = "display:block;cursor:grab;outline:none;position:absolute;";
        this.viewport.appendChild(this.canvas);

        this.container.appendChild(this.viewport);

        // Info bar below the viewport — flex-shrink:0 so it's never cropped
        const bar = document.createElement("div");
        bar.style.cssText = [
            "flex-shrink:0",
            "padding:7px 10px",
            "background:#0d1117",
            "border-top:1px solid #1e3a5f",
            "font-family:monospace",
            "font-size:11px",
            "display:flex",
            "flex-direction:column",
            "gap:4px",
        ].join(";");

        this.promptLabel = document.createElement("div");
        this.promptLabel.style.cssText = "color:#e2e8f0;letter-spacing:0.02em;min-height:1.4em;";
        bar.appendChild(this.promptLabel);

        // LoRA format toggle
        const fmtRow = document.createElement("div");
        fmtRow.style.cssText = "display:flex;gap:5px;align-items:center;";

        const fmtLabel = document.createElement("span");
        fmtLabel.style.cssText = "color:#4a5568;font-size:10px;margin-right:2px;";
        fmtLabel.textContent = "LoRA:";
        fmtRow.appendChild(fmtLabel);

        this._btnFal  = this._makeFmtBtn("fal / 2511",   "fal");
        this._btnDx   = this._makeFmtBtn("dx8152 / 2509", "dx8152");
        fmtRow.appendChild(this._btnFal);
        fmtRow.appendChild(this._btnDx);
        bar.appendChild(fmtRow);

        const legend = document.createElement("div");
        legend.style.cssText = "display:flex;gap:14px;font-size:10px;color:#4a5568;flex-wrap:wrap;";
        legend.innerHTML = [
            '<span><b style="color:#4caf50">●</b> drag L/R = rotate</span>',
            '<span><b style="color:#e91e8c">●</b> drag U/D = tilt</span>',
            '<span><b style="color:#ff9800">●</b> drag U/D = zoom</span>',
            '<span style="margin-left:auto;color:#2d3748">drag scene = orbit view</span>',
        ].join("");
        bar.appendChild(legend);

        this.container.appendChild(bar);
        this._refreshLabel();
    }

    _refreshLabel() {
        if (!this.promptLabel) return;
        this.promptLabel.textContent = this._prompt();
    }

    _prompt() {
        const az = this.azIdx, el = this.elIdx, dist = this.distIdx;

        if (this.fmt === "fal") {
            return `<sks> ${AZIMUTHS[az].label} ${ELEVATIONS[el].label} ${DISTANCES[dist].label}`;
        }

        // dx8152/Qwen-Edit-2509 — bilingual natural language, no trigger word.
        // Azimuth maps to relative rotation (clamped to ±90°).
        // Phrases match the linoyts HuggingFace space training distribution.
        const DX_AZ = [
            null,
            ["将镜头向右旋转45度", "Rotate the camera 45 degrees to the right."],
            ["将镜头向右旋转90度", "Rotate the camera 90 degrees to the right."],
            ["将镜头向右旋转90度", "Rotate the camera 90 degrees to the right."],  // 135° clamped
            ["将镜头向右旋转90度", "Rotate the camera 90 degrees to the right."],  // 180° clamped
            ["将镜头向左旋转90度", "Rotate the camera 90 degrees to the left."],   // 225° clamped
            ["将镜头向左旋转90度", "Rotate the camera 90 degrees to the left."],
            ["将镜头向左旋转45度", "Rotate the camera 45 degrees to the left."],
        ];
        const DX_EL = [
            ["将相机切换到仰视视角", "Turn the camera to a worm's-eye view."],  // -30°
            null,                                                                  //   0° eye-level
            ["将镜头向上移动",       "Move the camera up."],                      //  30° elevated
            ["将相机转向鸟瞰视角",   "Turn the camera to a bird's-eye view."],    //  60° high-angle
        ];
        const DX_DIST = [
            ["将镜头转为特写镜头", "Turn the camera to a close-up."],  // dist=0  close-up
            ["将镜头向前移动",     "Move the camera forward."],         // dist=1  forward
            null,                                                        // dist=2  medium — neutral, no command
            ["将镜头向后移动",     "Move the camera backward."],        // dist=3  wide — zoom out / go wider
        ];

        const parts = [];
        const rot  = DX_AZ[az];   if (rot)  parts.push(`${rot[0]} ${rot[1]}`);
        const tilt = DX_EL[el];   if (tilt) parts.push(`${tilt[0]} ${tilt[1]}`);
        const zoom = DX_DIST[dist]; if (zoom) parts.push(`${zoom[0]} ${zoom[1]}`);
        return parts.length > 0 ? parts.join(" ") : "保持当前视角 Keep the current camera angle.";
    }

    // ── Format toggle helpers ─────────────────────────────────────────────────

    _makeFmtBtn(label, fmtKey) {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.style.cssText = [
            "padding:2px 7px",
            "border-radius:3px",
            "font-size:10px",
            "cursor:pointer",
            "font-family:monospace",
            "transition:all 0.15s",
        ].join(";");
        btn.addEventListener("click", () => this._setFormat(fmtKey));
        return btn;
    }

    _setFormat(fmtKey) {
        this.fmt = fmtKey;
        this._updateFmtButtons();
        this._refreshLabel();
        this._syncToWidget();
    }

    _updateFmtButtons() {
        const active   = "background:#1e3a5f;color:#e2e8f0;border:1px solid #4a90d9;";
        const inactive = "background:transparent;color:#4a5568;border:1px solid #2d3748;";
        if (this._btnFal)  this._btnFal.style.cssText  += this.fmt === "fal"    ? active : inactive;
        if (this._btnDx)   this._btnDx.style.cssText   += this.fmt === "dx8152" ? active : inactive;
    }

    // ── Three.js scene init ────────────────────────────────────────────────────

    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(C_BG);

        // Start with viewport dimensions; the ResizeObserver will update on layout.
        const w0 = this.viewport.clientWidth  || 400;
        const h0 = this.viewport.clientHeight || 320;

        this.viewCam = new THREE.PerspectiveCamera(52, w0 / h0, 0.1, 100);
        this.viewCam.position.set(0, 4.5, 10);
        this.viewCam.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(w0, h0);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambient);

        const sun = new THREE.DirectionalLight(0xffffff, 0.9);
        sun.position.set(5, 10, 8);
        this.scene.add(sun);

        const fill = new THREE.DirectionalLight(0xffffff, 0.25);
        fill.position.set(-5, 3, -6);
        this.scene.add(fill);

        // Subtle ground grid
        const grid = new THREE.GridHelper(12, 24, 0x1e3a5f, 0x162035);
        grid.position.y = -1.6;
        this.scene.add(grid);

        this.raycaster = new THREE.Raycaster();
        this.mouse2D   = new THREE.Vector2();
    }

    // ── Subject (humanoid standing in center, nose pointing +Z = "front") ──────

    _buildSubject() {
        const g   = new THREE.Group();
        const mat = (c, e = 0x000000) =>
            new THREE.MeshPhongMaterial({ color: c, emissive: e, shininess: 40 });

        // Head
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.38, 24, 24), mat(0x0f3460));
        head.position.y = 1.2;
        g.add(head);

        // Nose → indicates "front" direction (+Z)
        const nose = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.18, 8), mat(C_GREEN, 0x1a4a1a));
        nose.rotation.x = -Math.PI / 2;
        nose.position.set(0, 1.2, 0.5);
        g.add(nose);

        // Torso
        const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.32, 0.88, 24), mat(0x16213e));
        torso.position.y = 0.3;
        g.add(torso);

        // Shoulders
        const shoulders = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.13, 0.28), mat(0x1a2a4a));
        shoulders.position.y = 0.78;
        g.add(shoulders);

        // "F" label on base (canvas sprite)
        this._addLabel(g, "F", 0, -0.1, 0.55, "#4caf50");
        this._addLabel(g, "B", 0, -0.1, -0.55, "#f44336");

        this.scene.add(g);
    }

    _addLabel(parent, text, x, y, z, color) {
        const cvs = document.createElement("canvas");
        cvs.width = cvs.height = 64;
        const ctx = cvs.getContext("2d");
        ctx.fillStyle = color;
        ctx.font = "bold 46px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, 32, 32);

        const tex = new THREE.CanvasTexture(cvs);
        const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
        spr.position.set(x, y, z);
        spr.scale.set(0.3, 0.3, 1);
        parent.add(spr);
    }

    // ── Guides: green azimuth ring + pink vertical tilt arc ───────────────────

    _buildGuides() {
        // Green equatorial ring — rotation reference, glows like the HF space arc
        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(RING_RADIUS, 0.035, 8, 80),
            new THREE.MeshStandardMaterial({
                color: C_GREEN, emissive: C_GREEN, emissiveIntensity: 0.25,
                roughness: 0.6, metalness: 0.1,
            })
        );
        ring.rotation.x = Math.PI / 2;
        this.scene.add(ring);

        // 8 faint tick dots on the ring (one per azimuth stop)
        for (let i = 0; i < 8; i++) {
            const az = toRad(AZIMUTHS[i].deg);
            const tick = new THREE.Mesh(
                new THREE.SphereGeometry(0.06, 8, 8),
                new THREE.MeshStandardMaterial({
                    color: C_GREEN, emissive: C_GREEN, emissiveIntensity: 0.15,
                })
            );
            tick.position.set(Math.sin(az) * RING_RADIUS, 0, Math.cos(az) * RING_RADIUS);
            this.scene.add(tick);
        }

        // Pink vertical tilt arc — on the left side (x = -0.7), centered at CENTER_Y.
        // Spans the elevation range: -30° (low-angle) to 60° (high-angle).
        // Matches the HF space's tilt reference curve geometry exactly.
        const tiltPts = [];
        for (let i = 0; i <= 40; i++) {
            const deg = -30 + 90 * (i / 40);   // -30° → +60°
            const rad = toRad(deg);
            tiltPts.push(new THREE.Vector3(
                -0.7,
                TILT_RADIUS * Math.sin(rad) + CENTER_Y,
                TILT_RADIUS * Math.cos(rad)
            ));
        }
        const tiltCurve = new THREE.CatmullRomCurve3(tiltPts);
        const tiltArc = new THREE.Mesh(
            new THREE.TubeGeometry(tiltCurve, 40, 0.035, 8, false),
            new THREE.MeshStandardMaterial({
                color: C_PINK, emissive: C_PINK, emissiveIntensity: 0.3,
                roughness: 0.5, metalness: 0.1,
            })
        );
        this.scene.add(tiltArc);
    }

    // ── Camera marker: box body + conical lens (matches HF space camera shape) ──

    _buildCameraMarker() {
        // Camera group — box body with a tapered cylindrical lens protruding forward.
        // Shape mirrors the compound camera model in the HuggingFace space.
        // MeshStandardMaterial gives the metallic blue-grey look from the space.
        this.cameraGroup = new THREE.Group();

        const camMat = new THREE.MeshStandardMaterial({
            color: C_CAM, emissive: 0x003344, emissiveIntensity: 0.25,
            metalness: 0.5, roughness: 0.3,
        });

        // Rectangular box body  (w × h × d)
        const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.20, 0.35), camMat);
        this.cameraGroup.add(body);

        // Tapered lens cylinder — narrow end (+Z, toward subject) after lookAt
        // CylinderGeometry(topR, bottomR, height, segments): top = 0.07, base = 0.10
        const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.10, 0.16, 16), camMat);
        lens.rotation.x = Math.PI / 2;  // axis now along Z
        lens.position.z = 0.24;         // protrudes forward (+Z = toward subject after lookAt)
        this.cameraGroup.add(lens);

        this.scene.add(this.cameraGroup);

        // Orange distance line — thin line from CENTER to camera (matches HF space).
        // Updated every frame in _placeMarkers().
        const pts = [new THREE.Vector3(0, CENTER_Y, 0), new THREE.Vector3(0, CENTER_Y, 3)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        this.cameraLinePts = geo.attributes.position;
        this.cameraLine = new THREE.Line(
            geo,
            new THREE.LineBasicMaterial({ color: C_ORANGE, transparent: true, opacity: 0.55 })
        );
        this.scene.add(this.cameraLine);

        this._placeMarkers();
    }

    // ── Three colored handles ──────────────────────────────────────────────────

    _buildHandles() {
        const geo = new THREE.SphereGeometry(0.23, 18, 18);

        this.handleGreen = new THREE.Mesh(geo,
            new THREE.MeshPhongMaterial({ color: C_GREEN, emissive: 0x1a4a1a, shininess: 90 })
        );
        this.handleGreen.userData = { handle: "green" };
        this.scene.add(this.handleGreen);

        this.handlePink = new THREE.Mesh(geo,
            new THREE.MeshPhongMaterial({ color: C_PINK, emissive: 0x4a0a2a, shininess: 90 })
        );
        this.handlePink.userData = { handle: "pink" };
        this.scene.add(this.handlePink);

        this.handleOrange = new THREE.Mesh(geo,
            new THREE.MeshPhongMaterial({ color: C_ORANGE, emissive: 0x4a2a00, shininess: 90 })
        );
        this.handleOrange.userData = { handle: "orange" };
        this.scene.add(this.handleOrange);

        this._placeMarkers();
    }

    /**
     * Update all marker/handle positions to reflect current az/el/dist state.
     *
     * Layout mirrors the HuggingFace space:
     *   Green  — on the equatorial ring at current azimuth (moves around the ring).
     *   Pink   — on the vertical tilt arc at x=−0.7 (fixed x, moves up/down the arc).
     *   Orange — along the camera ray between CENTER and the camera (moves in/out).
     *   Camera — at spherical position (az, el, r), body+lens facing CENTER.
     *   Line   — thin orange line from CENTER to camera (distance reference).
     */
    _placeMarkers() {
        if (!this.cameraGroup || !this.handleGreen) return;

        const az = toRad(AZIMUTHS[this.azIdx].deg);
        const el = toRad(ELEVATIONS[this.elIdx].deg);
        const r  = DISTANCES[this.distIdx].r;

        // Camera group position (spherical coords, offset up by CENTER_Y)
        const cx = r * Math.cos(el) * Math.sin(az);
        const cy = r * Math.sin(el) + CENTER_Y;
        const cz = r * Math.cos(el) * Math.cos(az);

        this.cameraGroup.position.set(cx, cy, cz);
        // lookAt on a Group makes +Z face the target → lens (at local z=+0.24) faces CENTER
        this.cameraGroup.lookAt(0, CENTER_Y, 0);

        // Orange distance line: thin line from CENTER to camera
        this.cameraLinePts.setXYZ(0, 0, CENTER_Y, 0);
        this.cameraLinePts.setXYZ(1, cx, cy, cz);
        this.cameraLinePts.needsUpdate = true;

        // Green handle: on equatorial ring at y=0, current azimuth
        this.handleGreen.position.set(
            Math.sin(az) * RING_RADIUS,
            0,
            Math.cos(az) * RING_RADIUS
        );

        // Pink handle: on the vertical tilt arc (x=−0.7, centered at CENTER_Y).
        // Position tracks the current elevation stop on the arc.
        this.handlePink.position.set(
            -0.7,
            TILT_RADIUS * Math.sin(el) + CENTER_Y,
            TILT_RADIUS * Math.cos(el)
        );

        // Orange handle: along the camera ray between CENTER and the camera.
        // Sits at ~65% of the camera distance so it's visually between them.
        const oR = r * 0.65;
        this.handleOrange.position.set(
            oR * Math.cos(el) * Math.sin(az),
            oR * Math.sin(el) + CENTER_Y,
            oR * Math.cos(el) * Math.cos(az)
        );
    }

    // ── Events ────────────────────────────────────────────────────────────────

    _setupEvents() {
        this.canvas.addEventListener("mousedown",  this._onDown.bind(this));
        this.canvas.addEventListener("mousemove",  this._onHover.bind(this));
        this.canvas.addEventListener("mouseleave", this._onLeave.bind(this));
        this.canvas.addEventListener("wheel",      this._onWheel.bind(this), { passive: false });

        // Move and up go on window so drags outside the canvas still register
        window.addEventListener("mousemove", this._boundMove);
        window.addEventListener("mouseup",   this._boundUp);

        this.resizeObs = new ResizeObserver(entries => {
            for (const e of entries) {
                const w = Math.floor(e.contentRect.width);
                const h = Math.floor(e.contentRect.height);
                if (w > 0 && h > 0 && this.renderer) {
                    // Fill the full viewport — update camera aspect to match
                    this.renderer.setSize(w, h);
                    this.canvas.style.width  = w + "px";
                    this.canvas.style.height = h + "px";
                    this.canvas.style.left   = "0";
                    this.canvas.style.top    = "0";
                    this.viewCam.aspect = w / h;
                    this.viewCam.updateProjectionMatrix();
                }
            }
        });
        this.resizeObs.observe(this.viewport);
    }

    _mousePosNDC(e) {
        const r = this.canvas.getBoundingClientRect();
        return {
            x:  ((e.clientX - r.left)  / r.width)  * 2 - 1,
            y: -((e.clientY - r.top)   / r.height) * 2 + 1,
        };
    }

    _hitHandle(ndcX, ndcY) {
        this.mouse2D.set(ndcX, ndcY);
        this.raycaster.setFromCamera(this.mouse2D, this.viewCam);
        const hits = this.raycaster.intersectObjects(
            [this.handleGreen, this.handlePink, this.handleOrange]
        );
        return hits.length > 0 ? hits[0].object : null;
    }

    _onDown(e) {
        const { x, y } = this._mousePosNDC(e);
        const hit = this._hitHandle(x, y);

        this.dragStartMouse = { x: e.clientX, y: e.clientY };
        this.dragStartAz    = this.azIdx;
        this.dragStartEl    = this.elIdx;
        this.dragStartDist  = this.distIdx;

        if (hit) {
            this.dragging = hit.userData.handle;
            this._setEmissive(hit, true);
        } else {
            this.dragging = "orbit";
        }
        this.canvas.style.cursor = "grabbing";
        e.preventDefault();
    }

    _onMove(e) {
        if (!this.dragging) return;

        const dx = e.clientX - this.dragStartMouse.x;
        const dy = e.clientY - this.dragStartMouse.y;

        if (this.dragging === "orbit") {
            // Rotate the entire scene group for viewer orbit
            this.scene.rotation.y += (e.movementX || 0) * 0.006;
            this.scene.rotation.x += (e.movementY || 0) * 0.006;
            this.scene.rotation.x = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, this.scene.rotation.x));
            return;
        }

        let changed = false;

        if (this.dragging === "green") {
            // Drag right → clockwise rotation (increasing az index)
            // ~70 px per 45° step
            const steps  = Math.round(dx / 70);
            const newIdx = ((this.dragStartAz + steps) % 8 + 8) % 8;
            if (newIdx !== this.azIdx) { this.azIdx = newIdx; changed = true; }

        } else if (this.dragging === "pink") {
            // Drag up (neg dy) → higher elevation index
            // ~65 px per step
            const steps  = Math.round(-dy / 65);
            const newIdx = Math.max(0, Math.min(3, this.dragStartEl + steps));
            if (newIdx !== this.elIdx) { this.elIdx = newIdx; changed = true; }

        } else if (this.dragging === "orange") {
            // Drag down → wider/farther (higher dist index)
            // ~40 px per step
            const steps  = Math.round(dy / 40);
            const newIdx = Math.max(0, Math.min(3, this.dragStartDist + steps));
            if (newIdx !== this.distIdx) { this.distIdx = newIdx; changed = true; }
        }

        if (changed) {
            this._placeMarkers();
            this._refreshLabel();
        }
    }

    _onUp(e) {
        if (this.dragging && this.dragging !== "orbit") {
            this._clearEmissive(this.handleGreen);
            this._clearEmissive(this.handlePink);
            this._clearEmissive(this.handleOrange);
            this._syncToWidget();
        }
        this.dragging = null;
        this.canvas.style.cursor = "grab";
    }

    _onHover(e) {
        if (this.dragging) return;
        const { x, y } = this._mousePosNDC(e);
        const hit = this._hitHandle(x, y);

        if (hit !== this.hoveredHandle) {
            if (this.hoveredHandle) this._setEmissive(this.hoveredHandle, false);
            this.hoveredHandle = hit;
            if (hit) {
                this._setEmissive(hit, true, 0.3);
                this.canvas.style.cursor = "pointer";
            } else {
                this.canvas.style.cursor = "grab";
            }
        }
    }

    _onLeave() {
        if (this.hoveredHandle) {
            this._setEmissive(this.hoveredHandle, false);
            this.hoveredHandle = null;
        }
        if (!this.dragging) this.canvas.style.cursor = "grab";
    }

    _onWheel(e) {
        e.preventDefault();
        // Scroll to dolly the viewer camera in/out
        const delta = e.deltaY > 0 ? 0.6 : -0.6;
        const dist  = this.viewCam.position.length();
        const nd    = Math.max(4, Math.min(20, dist + delta));
        this.viewCam.position.normalize().multiplyScalar(nd);
    }

    // ── Emissive helpers ──────────────────────────────────────────────────────

    _setEmissive(mesh, on, intensity = 0.55) {
        if (!mesh?.material) return;
        mesh.material.emissiveIntensity = on ? intensity : 0;
    }

    _clearEmissive(mesh) {
        this._setEmissive(mesh, false);
    }

    // ── Animation loop ────────────────────────────────────────────────────────

    _startAnimation() {
        const tick = () => {
            this.animId = requestAnimationFrame(tick);
            this.renderer.render(this.scene, this.viewCam);
        };
        tick();
    }

    // ── Widget sync ───────────────────────────────────────────────────────────

    _syncToWidget() {
        const value = JSON.stringify({
            az: this.azIdx, el: this.elIdx, dist: this.distIdx, fmt: this.fmt,
        });
        const w = this.node.widgets?.find(w => w.name === "camera_state");
        if (w) {
            w.value = value;
            w.callback?.(value);
        }
        app.graph?.setDirtyCanvas(true, true);
    }

    _loadFromWidget() {
        const w = this.node.widgets?.find(w => w.name === "camera_state");
        if (!w?.value) return;
        try {
            const s = JSON.parse(w.value);
            this.azIdx   = Math.max(0, Math.min(7, s.az   ?? 0));
            this.elIdx   = Math.max(0, Math.min(3, s.el   ?? 1));
            this.distIdx = Math.max(0, Math.min(3, s.dist ?? 2));
            this.fmt     = (s.fmt === "dx8152") ? "dx8152" : "fal";
        } catch (_) { /* keep defaults */ }
        this._updateFmtButtons();
        this._placeMarkers();
        this._refreshLabel();
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────

    destroy() {
        cancelAnimationFrame(this.animId);
        this.resizeObs?.disconnect();
        window.removeEventListener("mousemove", this._boundMove);
        window.removeEventListener("mouseup",   this._boundUp);
        this.renderer?.dispose();
        this.renderer?.forceContextLoss();
        this.scene?.traverse(c => {
            if (!c.isMesh) return;
            c.geometry?.dispose();
            const mats = Array.isArray(c.material) ? c.material : [c.material];
            mats.forEach(m => m?.dispose());
        });
    }
}

// ─── ComfyUI extension registration ──────────────────────────────────────────

app.registerExtension({
    name: "yaple.QwenCameraPrompt",

    async nodeCreated(node) {
        if (node.comfyClass !== "QwenCameraPrompt") return;

        const widget = new QwenCameraWidget(node);
        const ok = await widget.initialize();
        if (!ok) {
            console.error("[QwenCameraPrompt] 3D widget failed to initialize.");
            return;
        }

        // Hide the raw string widget — the 3D viewport owns it
        const stateWidget = node.widgets?.find(w => w.name === "camera_state");
        if (stateWidget) {
            stateWidget.type = "hidden";
            if (stateWidget.computeSize) stateWidget.computeSize = () => [0, -4];
        }

        // Attach the DOM widget (Three.js canvas + prompt bar)
        node.addDOMWidget("qwen_camera_3d", "customCanvas", widget.container, {
            getValue: () =>
                node.widgets?.find(w => w.name === "camera_state")?.value
                ?? '{"az":0,"el":1,"dist":2}',
            setValue: v => {
                const w = node.widgets?.find(w => w.name === "camera_state");
                if (w) w.value = v;
                widget._loadFromWidget();
            },
        });

        node.setSize([740, 663]);
        node.resizable = true;

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            widget.destroy();
            origRemoved?.call(this);
        };
    },
});
