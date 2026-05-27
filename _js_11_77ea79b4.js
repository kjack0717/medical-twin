// Body silhouette, the vertical blue path, and the pill itself.

// Drug-front Y on the path — always defined, even when the pill is hidden
// during an organ activation. We compute the furthest anchor reached so far.
function drugFrontY(t) {
  if (t < T.swallow.e) return PATH.top;
  // Interpolate along each leg using the same easing as pillPosition
  const legs = [
    { from: PATH.top,            to: ANCHORS.intestine, travel: T.travel1 },
    { from: ANCHORS.intestine,   to: ANCHORS.liver,     travel: T.travel2 },
    { from: ANCHORS.liver,       to: ANCHORS.blood,     travel: T.travel3 },
    { from: ANCHORS.blood,       to: ANCHORS.joint,     travel: T.travel4 },
  ];
  let y = PATH.top;
  for (const L of legs) {
    if (t < L.travel.s) { y = L.from; break; }
    if (t < L.travel.e) {
      const p = Easing.easeInOutCubic((t - L.travel.s) / (L.travel.e - L.travel.s));
      y = L.from + (L.to - L.from) * p;
      break;
    }
    y = L.to;
  }
  return y;
}

function BodyAndPath({ t }) {
  const pill = pillPosition(t);
  const fillY = drugFrontY(t);

  return (
    <>
      {/* Body silhouette — head profile + shoulders. mix-blend-mode so its
          white background disappears against the page bg. */}
      <div style={{
        position: 'absolute',
        left: PATH.x - 240, top: 70,
        width: 520, height: 540,
        mixBlendMode: 'multiply',
        pointerEvents: 'none',
      }}>
        <img src={(window.__resources && window.__resources.body) || "assets/body.png"} alt=""
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}/>
      </div>

      {/* Path container — drawn behind the body? No, on top so it shows through.
          The body is line-only so the path can sit "inside" it. */}
      <svg
        width={W} height={H}
        viewBox={`0 0 ${W} ${H}`}
        style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      >
        {/* Path outline */}
        <rect
          x={PATH.x - PATH.width/2}
          y={PATH.top}
          width={PATH.width}
          height={PATH.bottom - PATH.top}
          rx={PATH.width/2}
          fill="none"
          stroke={C.path}
          strokeWidth={3}
          opacity={0.85}
        />
        {/* Path fill — grows downward as pill descends */}
        <rect
          x={PATH.x - PATH.width/2 + 4}
          y={PATH.top}
          width={PATH.width - 8}
          height={Math.max(0, fillY - PATH.top)}
          fill={C.pathSoft}
          opacity={0.6}
        />
        {/* Anchor dots at each organ delivery point */}
        {Object.entries(ANCHORS).map(([key, y]) => {
          const reached = t >= getActivateTime(key);
          return (
            <g key={key}>
              <circle cx={PATH.x} cy={y} r={reached ? 9 : 6}
                fill={reached ? C.red : C.path}
                stroke="#fff" strokeWidth={2}
                style={{ transition: 'all 200ms' }}
              />
              {reached && (
                <circle cx={PATH.x} cy={y} r={9}
                  fill="none"
                  stroke={C.red}
                  strokeWidth={2}
                  opacity={0.5 - 0.5 * Math.min(1, (t - getActivateTime(key)) / 1.2)}
                />
              )}
            </g>
          );
        })}
      </svg>

      {/* Mouth indicator — small entry mark at the top of the path,
          glows when the pill enters. */}
      <MouthGlow t={t} />
    </>
  );
}

function MouthGlow({ t }) {
  const active = t >= T.swallow.s && t <= T.swallow.e + 0.4;
  const p = clamp((t - T.swallow.s) / 0.6, 0, 1);
  const size = active ? 26 + 14 * p : 16;
  const op = active ? 0.7 * (1 - p * 0.5) : 0;
  return (
    <div style={{
      position: 'absolute',
      left: MOUTH.x, top: MOUTH.y,
      width: size, height: size,
      borderRadius: '50%',
      background: `radial-gradient(circle, ${C.path}55 0%, transparent 70%)`,
      transform: 'translate(-50%, -50%)',
      opacity: op,
      pointerEvents: 'none',
    }}/>
  );
}

function Pill({ t, dim }) {
  const p = pillPosition(t);
  if (!p.visible) return null;
  const fade = 1 - (dim || 0);
  const pulse = 1 + 0.08 * Math.sin(t * 8);
  return (
    <div style={{
      position: 'absolute',
      left: p.x, top: p.y,
      width: 50, height: 50,
      transform: `translate(-50%, -50%) rotate(${p.rot}deg) scale(${p.scale * pulse})`,
      opacity: p.opacity * fade,
      filter: `drop-shadow(0 4px 8px rgba(214,39,30,0.35))`,
      pointerEvents: 'none',
      willChange: 'transform, opacity',
    }}>
      <img src={(window.__resources && window.__resources.pill) || "assets/pill.png"} alt=""
        style={{ width: '100%', height: '100%', display: 'block' }}/>
    </div>
  );
}

Object.assign(window, { BodyAndPath, Pill, MouthGlow });
