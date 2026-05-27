// Organ panels (with focus zoom), arrows, and the DeliveryPill.

const bezX = (p, p0, p1, p2) => {
  const u = 1 - p;
  return u*u*p0 + 2*u*p*p1 + p*p*p2;
};

function panelCenter(cfg) {
  return { x: cfg.x + PANEL_W / 2, y: cfg.y + PANEL_H * 0.42 };
}
function panelEdge(cfg) {
  return {
    x: cfg.side === 'left' ? cfg.x + PANEL_W : cfg.x,
    y: cfg.y + PANEL_H / 2,
  };
}
function arrowControl(cfg) {
  const a = { x: PATH.x, y: cfg.anchorY };
  const b = panelEdge(cfg);
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 + (b.y < a.y ? -40 : 30) };
}

// ─────────────────────────────────────────────────────────────────────────────
// OrganPanel — frame, organ icon (with zoom), red radial spread, pulse rings.
// When focused (focusP > 0), translates + scales toward canvas center so the
// panel and its contents grow to nearly full-screen.

function OrganPanel({ cfg, t, activeAt, focusP, dimOpacity }) {
  const since = t - activeAt;
  const fp = focusP || 0;
  const dim = dimOpacity || 0;

  // Sub-phase progress
  const zoomP        = clamp((since - SUB.focusIn.s) / (SUB.focusIn.e - SUB.focusIn.s), 0, 1);
  const spreadP      = clamp((since - SUB.spread.s) / (SUB.spread.e - SUB.spread.s), 0, 1);
  const spreadEased  = Easing.easeOutCubic(spreadP);
  const affected     = since >= SUB.spread.s;
  const deliveryStarted = since >= 0;

  // Inner organ icon scale (independent of focus scale).
  // It's a subtle extra pop inside the panel as it zooms.
  let iconScale = 1.0;
  if (since >= SUB.focusIn.s) {
    const inner = clamp((since - SUB.focusIn.s) / 0.9, 0, 1);
    iconScale = 1 + 0.18 * Easing.easeOutCubic(inner);
    if (since > SUB.focusIn.e + 0.5) {
      iconScale += Math.sin((since - SUB.focusIn.e - 0.5) * 2.2) * 0.02;
    }
  }

  // Slight shake on delivery impact (≈0.7s)
  let shakeX = 0, shakeY = 0;
  if (since >= SUB.delivery.e && since < SUB.delivery.e + 0.4) {
    const k = (1 - (since - SUB.delivery.e) / 0.4) * 5;
    shakeX = Math.sin(since * 60) * k;
    shakeY = Math.cos(since * 55) * k;
  }

  // Focus-zoom transform: translate panel center to canvas center, scale up.
  const baseCx = cfg.x + PANEL_W / 2;
  const baseCy = cfg.y + PANEL_H / 2;
  const tx = (FOCUS_CX - baseCx) * fp;
  const ty = (FOCUS_CY - baseCy) * fp;
  const s = 1 + (FOCUS_SCALE - 1) * fp;

  // Larger radial spread when focused
  const localSpreadRadius = spreadEased * PANEL_W * 0.95;

  const labelColor = affected ? C.redDeep : C.inkSoft;
  const pulseT = Math.max(0, since - SUB.spread.e);

  let statusText = 'Standby';
  let statusFilled = false;
  if (since >= 0 && since < SUB.dissolve.e) {
    statusText = 'Receiving';
    statusFilled = true;
  } else if (affected) {
    statusText = 'Affected';
    statusFilled = true;
  }

  const zIndex = fp > 0.05 ? 50 : 1;

  return (
    <div style={{
      position: 'absolute',
      left: cfg.x + shakeX, top: cfg.y + shakeY,
      width: PANEL_W, height: PANEL_H,
      transform: `translate(${tx}px, ${ty}px) scale(${s})`,
      transformOrigin: 'center center',
      opacity: 1 - dim,
      zIndex,
      background: C.boxBg,
      border: `3px solid ${affected ? C.redDeep : C.boxStroke}`,
      borderRadius: 4,
      overflow: 'hidden',
      boxShadow: affected
        ? `0 0 0 4px ${C.red}11, 0 12px 32px rgba(165,26,20,${0.05 + 0.15 * spreadEased})`
        : '0 4px 12px rgba(80,40,20,0.06)',
      willChange: 'transform',
    }}>
      {/* Red radial spread (behind icon) */}
      <div style={{
        position: 'absolute',
        left: '50%', top: '46%',
        width: localSpreadRadius * 2, height: localSpreadRadius * 2,
        marginLeft: -localSpreadRadius, marginTop: -localSpreadRadius,
        borderRadius: '50%',
        background: `radial-gradient(circle, ${C.red}dd 0%, ${C.red}99 30%, ${C.red}55 55%, ${C.red}22 75%, transparent 90%)`,
        opacity: 0.6,
        pointerEvents: 'none',
      }}/>

      {/* Leading wavefront ring */}
      {affected && spreadP < 1 && (
        <div style={{
          position: 'absolute',
          left: '50%', top: '46%',
          width: localSpreadRadius * 2.2, height: localSpreadRadius * 2.2,
          marginLeft: -localSpreadRadius * 1.1, marginTop: -localSpreadRadius * 1.1,
          borderRadius: '50%',
          border: `3px solid ${C.red}`,
          opacity: 0.4 * (1 - spreadP),
          pointerEvents: 'none',
        }}/>
      )}

      {/* Sustained pulse rings */}
      {pulseT > 0 && [0, 1].map(i => {
        const phase = ((pulseT + i * 1.1) % 2.2) / 2.2;
        const r = 40 + phase * (PANEL_W * 0.5);
        return (
          <div key={i} style={{
            position: 'absolute',
            left: '50%', top: '46%',
            width: r * 2, height: r * 2,
            marginLeft: -r, marginTop: -r,
            borderRadius: '50%',
            border: `2px solid ${C.red}`,
            opacity: 0.32 * (1 - phase),
            pointerEvents: 'none',
          }}/>
        );
      })}

      {/* Organ icon */}
      <div style={{
        position: 'absolute',
        left: 50, top: 40,
        width: PANEL_W - 100, height: PANEL_H - 140,
        mixBlendMode: 'multiply',
        transform: `scale(${iconScale})`,
        transformOrigin: 'center 46%',
        willChange: 'transform',
        filter: deliveryStarted && since < SUB.focusIn.e
          ? `drop-shadow(0 0 ${zoomP * 18}px ${C.red}77)`
          : 'none',
      }}>
        <img src={cfg.src} alt=""
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}/>
      </div>

      {/* Label */}
      <div style={{
        position: 'absolute',
        left: 0, right: 0, bottom: 22,
        textAlign: 'center',
        fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif',
      }}>
        <div style={{
          fontSize: 34, fontWeight: 700, color: labelColor,
          letterSpacing: '-0.02em', transition: 'color 400ms',
        }}>
          {cfg.label}
        </div>
        <div style={{
          fontSize: 12, fontWeight: 500, color: labelColor,
          opacity: 0.6, letterSpacing: '0.18em', textTransform: 'uppercase',
          marginTop: 2, transition: 'color 400ms',
        }}>
          {cfg.sub}
        </div>
      </div>

      {/* Status chip */}
      <div style={{
        position: 'absolute',
        top: 12, right: 12,
        padding: '3px 9px',
        fontSize: 10, letterSpacing: '0.18em',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
        textTransform: 'uppercase',
        background: statusFilled ? C.red : 'transparent',
        color: statusFilled ? '#fff' : C.boxStroke,
        border: `1px solid ${statusFilled ? C.red : C.boxStroke}`,
        borderRadius: 2,
        transition: 'all 300ms',
      }}>
        {statusText}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrows

function Arrows({ t }) {
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      {Object.entries(PANELS).map(([key, cfg]) => {
        const startX = PATH.x;
        const startY = cfg.anchorY;
        const end = panelEdge(cfg);
        const c = arrowControl(cfg);

        const activeAt = getActivateTime(key);
        const since = t - activeAt;
        const deliveryP = clamp(since / SUB.delivery.e, 0, 1);
        const deliveryEased = Easing.easeInOutCubic(deliveryP);
        const reached = since >= SUB.delivery.e;
        const affected = since >= SUB.spread.s;

        const targetColor = affected ? C.red : (reached ? C.redDeep : C.boxStroke);
        const opacity = reached || since > 0 ? 1 : 0.55;
        const sw = reached ? 2.6 : 1.6;

        const fullPath = `M ${startX} ${startY} Q ${c.x} ${c.y} ${end.x} ${end.y}`;

        return (
          <g key={key}>
            <path
              d={fullPath}
              fill="none"
              stroke={C.boxStroke}
              strokeWidth={1.4}
              strokeDasharray="6 6"
              opacity={0.55}
            />
            <ProgressPath
              d={fullPath}
              progress={deliveryEased}
              color={targetColor}
              strokeWidth={sw}
            />
            <Arrowhead
              x={end.x} y={end.y}
              fromX={c.x} fromY={c.y}
              color={reached ? targetColor : C.boxStroke}
              opacity={opacity}
            />
          </g>
        );
      })}
    </svg>
  );
}

function ProgressPath({ d, progress, color, strokeWidth }) {
  const ref = React.useRef(null);
  const [len, setLen] = React.useState(0);
  React.useEffect(() => {
    if (ref.current) setLen(ref.current.getTotalLength());
  }, [d]);
  return (
    <path
      ref={ref}
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={strokeWidth}
      strokeDasharray={len}
      strokeDashoffset={len * (1 - progress)}
      style={{ transition: 'stroke 300ms' }}
    />
  );
}

function Arrowhead({ x, y, fromX, fromY, color, opacity }) {
  const dx = x - fromX, dy = y - fromY;
  const ang = Math.atan2(dy, dx);
  const size = 12;
  const a1 = ang + Math.PI - 0.5;
  const a2 = ang + Math.PI + 0.5;
  return (
    <polygon
      points={`${x},${y} ${x + Math.cos(a1) * size},${y + Math.sin(a1) * size} ${x + Math.cos(a2) * size},${y + Math.sin(a2) * size}`}
      fill={color}
      opacity={opacity}
      style={{ transition: 'fill 300ms' }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DeliveryPill — pill that travels along the arrow into the panel, then
// "rides" the focus zoom (scaling up to match the focused panel size), then
// dissolves at the center of the giant organ view.

function DeliveryPill({ cfg, t, activeAt, focusP }) {
  const since = t - activeAt;
  if (since < 0 || since > SUB.dissolve.e + 0.05) return null;

  const start = { x: PATH.x, y: cfg.anchorY };
  const ctrl  = arrowControl(cfg);
  const edge  = panelEdge(cfg);
  const center = panelCenter(cfg);

  let x, y, rot, scale, opacity, blur, glow;
  const fp = focusP || 0;

  if (since < SUB.delivery.e) {
    // 1) Travel along bezier → panel edge, then short straight to center
    if (since < 0.5) {
      const p = Easing.easeInCubic(since / 0.5);
      x = bezX(p, start.x, ctrl.x, edge.x);
      y = bezX(p, start.y, ctrl.y, edge.y);
      const dx = bezX(Math.min(1, p + 0.02), start.x, ctrl.x, edge.x) - x;
      const dy = bezX(Math.min(1, p + 0.02), start.y, ctrl.y, edge.y) - y;
      rot = Math.atan2(dy, dx) * 180 / Math.PI + 90;
      scale = 0.55 + 0.12 * p;
      opacity = 1;
      blur = 1.2;
      glow = 0.6;
    } else {
      const p = Easing.easeOutCubic((since - 0.5) / 0.2);
      x = edge.x + (center.x - edge.x) * p;
      y = edge.y + (center.y - edge.y) * p;
      rot = -25;
      scale = 0.67 + 0.28 * p;
      opacity = 1;
      blur = 1.2 * (1 - p);
      glow = 0.7;
    }
  } else {
    // 2) After delivery: pill sits at panel center, then rides the focus
    //    zoom toward canvas center while scaling up.
    const fromX = center.x, fromY = center.y;
    const toX = FOCUS_CX, toY = FOCUS_CY;
    x = fromX + (toX - fromX) * fp;
    y = fromY + (toY - fromY) * fp;

    // Scale: at fp=0, ~0.95 (slightly bigger settle); at fp=1, FOCUS_SCALE * 0.95
    const baseScale = 0.95;
    scale = baseScale * (1 + (FOCUS_SCALE - 1) * fp);
    rot = -25;
    opacity = 1;
    blur = 0;
    glow = 0.7;

    // 3) Dissolve overrides during pillDissolve phase
    if (since >= SUB.dissolve.s) {
      const p = clamp((since - SUB.dissolve.s) / (SUB.dissolve.e - SUB.dissolve.s), 0, 1);
      const eased = Easing.easeInCubic(p);
      scale *= 1 + 1.4 * eased;
      opacity = 1 - eased;
      blur = p * 32;
      rot = -25 + 12 * p;
      glow = 0.7 * (1 - p * 0.4);
    } else if (since >= SUB.pillHold.s) {
      // Hold + small bob
      const tb = since - SUB.pillHold.s;
      y += Math.sin(tb * 4) * 2;
      scale *= 1 + 0.04 * Math.sin(tb * 5);
    }
  }

  return (
    <div style={{
      position: 'absolute',
      left: x, top: y,
      width: 80, height: 80,
      transform: `translate(-50%, -50%) rotate(${rot}deg) scale(${scale})`,
      opacity,
      filter: `blur(${blur}px) drop-shadow(0 0 ${12 + blur}px rgba(214,39,30,${glow}))`,
      pointerEvents: 'none',
      zIndex: 60,
      willChange: 'transform, opacity, filter',
    }}>
      <img src={(window.__resources && window.__resources.pill) || "assets/pill.png"} alt=""
        style={{ width: '100%', height: '100%', display: 'block' }}/>
    </div>
  );
}

Object.assign(window, {
  OrganPanel, Arrows, Arrowhead, ProgressPath, DeliveryPill,
  panelCenter, panelEdge, arrowControl,
});
