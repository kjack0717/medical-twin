// Medical Twin — Pill journey through the body.
// Each organ activation now has 5 sub-phases:
//   1) DELIVERY  — pill flies along arrow from path anchor → small panel
//   2) FOCUS-IN  — that panel zooms toward full-canvas size (camera move)
//   3) DISSOLVE  — at full focus, pill blurs + fades into the giant organ
//   4) SPREAD    — red drug effect radiates across the giant organ
//   5) FOCUS-OUT — camera zooms back out, panel returns to its corner (AFFECTED)

const W = 1600;
const H = 1000;

const C = {
  bg:        '#f8f4ec',
  ink:       '#1a1a1a',
  inkSoft:   '#3d3530',
  boxStroke: '#8b3327',
  boxBg:     '#fbf6ee',
  path:      '#3d8bff',
  pathSoft:  '#9ec3ff',
  red:       '#d6271e',
  redDeep:   '#a51a14',
  redSoft:   '#fde2de',
  bodyInk:   '#1a1a1a',
};

const PATH = {
  x: 798,
  top: 305,
  bottom: 935,
  width: 26,
};

const ANCHORS = {
  intestine: 555,
  liver:     635,
  blood:     750,
  joint:     860,
};

const PANEL_W = 380;
const PANEL_H = 380;
const RES = (typeof window !== 'undefined' && window.__resources) || {};
const PANELS = {
  intestine: { x: 70,   y: 80,  label: '장',    sub: 'Intestines',  src: RES.intestine || 'assets/intestine.png', anchorY: ANCHORS.intestine, side: 'left'  },
  liver:     { x: 1150, y: 80,  label: '간',    sub: 'Liver',       src: RES.liver     || 'assets/liver.png',     anchorY: ANCHORS.liver,     side: 'right' },
  blood:     { x: 70,   y: 540, label: '혈액',  sub: 'Bloodstream', src: RES.blood     || 'assets/blood.png',     anchorY: ANCHORS.blood,     side: 'left'  },
  joint:     { x: 1150, y: 540, label: '관절',  sub: 'Joints',      src: RES.joint     || 'assets/joint.png',     anchorY: ANCHORS.joint,     side: 'right' },
};

// Per-organ activation sub-phases (seconds relative to activation start).
const SUB = {
  delivery:  { s: 0.0, e: 0.7 },   // pill flies along arrow into small panel
  focusIn:   { s: 0.7, e: 1.5 },   // panel zooms toward full canvas
  pillHold:  { s: 1.5, e: 1.9 },   // pill held at center of giant organ
  dissolve:  { s: 1.9, e: 3.1 },   // pill blurs + fades
  spread:    { s: 2.5, e: 3.9 },   // red spreads
  focusOut:  { s: 4.0, e: 4.6 },   // panel zooms back to its corner
};
const ORGAN_PHASE_DUR = 4.6;
// Scale factor when fully focused (panel grows from 380px → 380*2.55 ≈ 970px)
const FOCUS_SCALE = 2.55;
// Where the focus view centers on the canvas
const FOCUS_CX = W / 2;
const FOCUS_CY = H * 0.475;

// Global timeline
const T = {
  titleIn:   { s: 0.0, e: 1.0 },
  pillEnter: { s: 1.0, e: 2.4 },
  swallow:   { s: 2.4, e: 3.0 },
  travel1:   { s: 3.0, e: 4.0 },

  intestActivate: 4.0,
  travel2:   { s: 8.6, e: 9.2 },

  liverActivate: 9.2,
  travel3:   { s: 13.8, e: 14.4 },

  bloodActivate: 14.4,
  travel4:   { s: 19.0, e: 19.6 },

  jointActivate: 19.6,

  hold:      { s: 24.2, e: 26.0 },
};

const DURATION = 26.0;

const MOUTH = { x: 740, y: 305 };

function getActivateTime(key) {
  if (key === 'intestine') return T.intestActivate;
  if (key === 'liver')     return T.liverActivate;
  if (key === 'blood')     return T.bloodActivate;
  if (key === 'joint')     return T.jointActivate;
  return Infinity;
}

// Compute current focus state: which organ panel (if any) is in its focus
// phase, and how strong the focus is (0..1). Only one organ can be focused
// at a time given the timeline.
function getCurrentFocus(t) {
  for (const key of Object.keys(PANELS)) {
    const activeAt = getActivateTime(key);
    const since = t - activeAt;
    if (since < SUB.focusIn.s || since > SUB.focusOut.e) continue;
    let p;
    if (since < SUB.focusIn.e) {
      p = Easing.easeInOutCubic((since - SUB.focusIn.s) / (SUB.focusIn.e - SUB.focusIn.s));
    } else if (since < SUB.focusOut.s) {
      p = 1;
    } else {
      p = 1 - Easing.easeInOutCubic((since - SUB.focusOut.s) / (SUB.focusOut.e - SUB.focusOut.s));
    }
    return { key, p };
  }
  return { key: null, p: 0 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Pill on the path.

function pillPosition(t) {
  if (t < T.pillEnter.s) {
    return { visible: false };
  }
  if (t < T.pillEnter.e) {
    const p = Easing.easeOutCubic((t - T.pillEnter.s) / (T.pillEnter.e - T.pillEnter.s));
    return {
      visible: true,
      x: 320 + (MOUTH.x - 320) * p,
      y: 280 + (MOUTH.y - 280) * p,
      opacity: Math.min(1, p * 2),
      rot: -28 + 28 * p,
      scale: 1.0,
    };
  }
  if (t < T.swallow.e) {
    const p = (t - T.swallow.s) / (T.swallow.e - T.swallow.s);
    return {
      visible: true,
      x: MOUTH.x + (PATH.x - MOUTH.x) * Easing.easeInCubic(p),
      y: MOUTH.y + (PATH.top - MOUTH.y) * p,
      opacity: 1,
      rot: 90 * p,
      scale: 1.0 - 0.1 * p,
    };
  }

  const atY = (y, opacity=1) => ({ visible: opacity > 0.01, x: PATH.x, y, opacity, rot: 90, scale: 0.9 });

  if (t < T.travel1.e) {
    const p = Easing.easeInOutCubic((t - T.travel1.s) / (T.travel1.e - T.travel1.s));
    return atY(PATH.top + (ANCHORS.intestine - PATH.top) * p);
  }

  const legs = [
    { activeAt: T.intestActivate, anchorY: ANCHORS.intestine, nextTravel: T.travel2 },
    { activeAt: T.liverActivate,  anchorY: ANCHORS.liver,     nextTravel: T.travel3 },
    { activeAt: T.bloodActivate,  anchorY: ANCHORS.blood,     nextTravel: T.travel4 },
    { activeAt: T.jointActivate,  anchorY: ANCHORS.joint,     nextTravel: null },
  ];

  for (let i = 0; i < legs.length; i++) {
    const L = legs[i];
    if (t < L.activeAt + 0.15) return atY(L.anchorY, 1);
    if (t < L.activeAt + 0.4) {
      const p = (t - (L.activeAt + 0.15)) / 0.25;
      return atY(L.anchorY, 1 - p);
    }
    if (!L.nextTravel) return { visible: false };
    if (t < L.nextTravel.s - 0.2) return { visible: false };
    if (t < L.nextTravel.s) {
      const p = (t - (L.nextTravel.s - 0.2)) / 0.2;
      return atY(L.anchorY, p);
    }
    if (t < L.nextTravel.e) {
      const nextAnchor = legs[i + 1].anchorY;
      const p = Easing.easeInOutCubic((t - L.nextTravel.s) / (L.nextTravel.e - L.nextTravel.s));
      return atY(L.anchorY + (nextAnchor - L.anchorY) * p);
    }
  }

  return { visible: false };
}

// ─────────────────────────────────────────────────────────────────────────────

function MainScene() {
  const t = useTime();
  const focus = getCurrentFocus(t);
  const dim = focus.p; // 0..1

  return (
    <div style={{ position: 'absolute', inset: 0, background: C.bg, overflow: 'hidden' }}>
      <GridBackdrop dim={dim} />
      <Title t={t} dim={dim} />

      {/* Background layer: body, path, arrows. Fades during focus. */}
      <div style={{
        position: 'absolute', inset: 0,
        opacity: 1 - dim * 0.92,
        transition: 'none',
        pointerEvents: 'none',
      }}>
        <BodyAndPath t={t} />
        <Arrows t={t} />
      </div>

      {/* Panels — each receives focusP if it's the active focus target,
          and a dim opacity if some OTHER organ is focused. */}
      {Object.entries(PANELS).map(([key, cfg]) => {
        const isFocused = focus.key === key;
        const otherFocused = focus.key && !isFocused;
        return (
          <OrganPanel
            key={key}
            cfg={cfg}
            t={t}
            activeAt={getActivateTime(key)}
            focusP={isFocused ? focus.p : 0}
            dimOpacity={otherFocused ? focus.p * 0.92 : 0}
          />
        );
      })}

      <Pill t={t} dim={dim} />

      {/* Delivery / dissolve pills, scaled with focus */}
      {Object.entries(PANELS).map(([key, cfg]) => (
        <DeliveryPill
          key={'d-' + key}
          cfg={cfg}
          t={t}
          activeAt={getActivateTime(key)}
          focusP={focus.key === key ? focus.p : 0}
        />
      ))}

      {/* "Now examining" label during focus */}
      {focus.key && focus.p > 0.4 && (
        <FocusLabel cfg={PANELS[focus.key]} focusP={focus.p} />
      )}

      <SequenceHud t={t} />
    </div>
  );
}

function GridBackdrop({ dim }) {
  return (
    <div style={{
      position: 'absolute', inset: 0,
      backgroundImage: 'radial-gradient(rgba(60,40,20,0.10) 1px, transparent 1px)',
      backgroundSize: '24px 24px',
      opacity: 0.5 * (1 - (dim || 0) * 0.7),
    }} />
  );
}

function Title({ t, dim }) {
  const p = clamp((t - T.titleIn.s) / (T.titleIn.e - T.titleIn.s), 0, 1);
  const op = Easing.easeOutCubic(p) * (1 - (dim || 0));
  return (
    <div style={{
      position: 'absolute', top: 30, left: 0, right: 0,
      textAlign: 'center', opacity: op,
      transform: `translateY(${(1-Easing.easeOutCubic(p)) * -8}px)`, pointerEvents: 'none',
      zIndex: 5,
    }}>
      <div style={{
        fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-u, sans-serif',
        fontSize: 13, letterSpacing: '0.4em', color: C.boxStroke,
        textTransform: 'uppercase', fontWeight: 600,
      }}>
        Medical Twin · Drug Pathway
      </div>
    </div>
  );
}

// Big label shown during focus zoom
function FocusLabel({ cfg, focusP }) {
  return (
    <div style={{
      position: 'absolute',
      top: 40, left: 0, right: 0,
      textAlign: 'center',
      opacity: focusP,
      pointerEvents: 'none',
      zIndex: 70,
      fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif',
    }}>
      <div style={{
        fontSize: 11, letterSpacing: '0.4em', color: C.boxStroke,
        textTransform: 'uppercase', fontWeight: 600, marginBottom: 8,
      }}>
        Drug Effect · {cfg.sub}
      </div>
      <div style={{
        fontSize: 30, fontWeight: 700, color: C.redDeep,
        letterSpacing: '0.04em',
      }}>
        {cfg.label}에 약효 확산 중
      </div>
    </div>
  );
}

function SequenceHud({ t }) {
  const stages = [
    { key: 'intake',    label: '복용',  at: T.pillEnter.s },
    { key: 'intestine', label: '장',    at: T.intestActivate + SUB.spread.s },
    { key: 'liver',     label: '간',    at: T.liverActivate + SUB.spread.s },
    { key: 'blood',     label: '혈액',  at: T.bloodActivate + SUB.spread.s },
    { key: 'joint',     label: '관절',  at: T.jointActivate + SUB.spread.s },
  ];

  let activeIdx = -1;
  for (let i = 0; i < stages.length; i++) {
    if (t >= stages[i].at) activeIdx = i;
  }

  return (
    <div style={{
      position: 'absolute', bottom: 28, left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex', alignItems: 'center', gap: 16,
      padding: '12px 22px',
      background: 'rgba(255,255,255,0.88)',
      backdropFilter: 'blur(8px)',
      borderRadius: 999,
      border: `1px solid ${C.boxStroke}22`,
      fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif',
      boxShadow: '0 6px 24px rgba(80,30,10,0.08)',
      zIndex: 80,
    }}>
      {stages.map((s, i) => {
        const isActive = i === activeIdx;
        const isPast = i < activeIdx;
        const color = isActive ? C.red : isPast ? C.redDeep : '#bdb6ac';
        return (
          <React.Fragment key={s.key}>
            {i > 0 && (
              <div style={{
                width: 22, height: 1,
                background: isPast || isActive ? C.redDeep : '#cfc8be',
                transition: 'background 200ms',
              }}/>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 12, height: 12, borderRadius: 6,
                background: isActive || isPast ? color : 'transparent',
                border: `2px solid ${color}`,
                boxShadow: isActive ? `0 0 0 6px ${C.red}22` : 'none',
                transition: 'box-shadow 200ms',
              }}/>
              <div style={{
                fontSize: 15,
                fontWeight: isActive ? 700 : 500,
                color: isActive ? C.red : isPast ? C.redDeep : '#897f72',
              }}>
                {s.label}
              </div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

Object.assign(window, {
  MainScene, getActivateTime, getCurrentFocus,
  PATH, ANCHORS, PANELS, MOUTH, T, SUB, C, DURATION, ORGAN_PHASE_DUR,
  pillPosition, W, H, PANEL_W, PANEL_H,
  FOCUS_SCALE, FOCUS_CX, FOCUS_CY,
});
