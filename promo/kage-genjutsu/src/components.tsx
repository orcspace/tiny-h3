import React from 'react';
import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame} from 'remotion';
import {CLIPS} from './clips';
import {C, FONT, H, W, aberration, lerp, lin, rand, useMode} from './lib';

/** Per-clip looks: the house grade, a red "ink warrior" (the OUTPUT of a motion transfer), dark mono, blood moon. */
const LOOKS: Record<string, {filter: string; layers: {bg: string; blend: React.CSSProperties['mixBlendMode']; o: number}[]}> = {
  grade: {
    filter: 'contrast(1.2) saturate(0.7) brightness(0.85)',
    layers: [
      {bg: C.deep, blend: 'multiply', o: 0.1},
      {bg: C.crimson, blend: 'soft-light', o: 0.18},
    ],
  },
  ink: {
    filter: 'grayscale(1) contrast(2.4) brightness(1.05)',
    layers: [
      {bg: C.crimson, blend: 'multiply', o: 1},
      {bg: C.hot, blend: 'screen', o: 0.12},
    ],
  },
  mono: {
    filter: 'grayscale(1) contrast(1.5) brightness(0.55)',
    layers: [{bg: C.crimson, blend: 'soft-light', o: 0.2}],
  },
  blood: {
    filter: 'grayscale(1) contrast(1.35) brightness(0.95)',
    layers: [
      {bg: C.crimson, blend: 'multiply', o: 0.95},
      {bg: C.hot, blend: 'soft-light', o: 0.4},
    ],
  },
};

/** A clip from clips.ts with its look; a slot without a clip is flat green. */
export const Slot: React.FC<{id: string; label: string; style?: React.CSSProperties; dim?: number; position?: string}> = ({
  id,
  label,
  style,
  dim = 0,
  position = 'center',
}) => {
  const {keyMode, labels} = useMode();
  const clip = CLIPS[id];
  const look = LOOKS[clip?.look ?? 'grade'];
  return (
    <AbsoluteFill style={{overflow: 'hidden', background: clip ? C.black : C.green, ...style}}>
      {clip ? (
        <>
          <OffthreadVideo
            src={staticFile(clip.src)}
            startFrom={clip.startFrom ?? 0}
            volume={clip.volume ?? 0}
            style={{
              width: '100%',
              height: '100%',
              objectFit: clip.fit ?? 'cover',
              objectPosition: position,
              transform: clip.mirror ? 'scaleX(-1)' : undefined,
              filter: keyMode ? undefined : look.filter,
            }}
          />
          {!keyMode &&
            look.layers.map((l, i) => (
              <AbsoluteFill key={i} style={{background: l.bg, mixBlendMode: l.blend, opacity: l.o}} />
            ))}
          {dim > 0 && <AbsoluteFill style={{background: '#000', opacity: dim}} />}
        </>
      ) : (
        labels &&
        !keyMode && (
          <AbsoluteFill style={{justifyContent: 'flex-start', alignItems: 'center', paddingTop: 110}}>
            <div
              style={{
                fontFamily: FONT.ui,
                fontWeight: 800,
                fontSize: 30,
                color: 'rgba(0,60,0,0.75)',
                letterSpacing: 4,
                textAlign: 'center',
                lineHeight: 1.4,
              }}
            >
              CLIP {id}
              <div style={{fontWeight: 500, fontSize: 22, letterSpacing: 2}}>{label}</div>
            </div>
          </AbsoluteFill>
        )
      )}
    </AbsoluteFill>
  );
};

/** SVG filters used across the film: ragged ink edges and drippy edges. */
export const Defs: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <svg width={0} height={0} style={{position: 'absolute'}}>
      <defs>
        <filter id="ink" x="-10%" y="-10%" width="120%" height="120%">
          <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves={2} seed={Math.floor(f / 2)} />
          <feDisplacementMap in="SourceGraphic" scale={7} />
        </filter>
        <filter id="inkHeavy" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves={3} seed={7} />
          <feDisplacementMap in="SourceGraphic" scale={120} />
        </filter>
        <filter id="drip" x="-10%" y="-10%" width="120%" height="160%">
          <feTurbulence type="fractalNoise" baseFrequency="0.025 0.0015" numOctaves={2} seed={3} />
          <feDisplacementMap in="SourceGraphic" scale={260} yChannelSelector="R" xChannelSelector="A" />
        </filter>
      </defs>
    </svg>
  );
};

/** Animated film grain + vignette + a faint crimson tint on the edges. Skipped in the green-screen master. */
export const Finish: React.FC = () => {
  const f = useCurrentFrame();
  const {keyMode} = useMode();
  if (keyMode) return null;
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <svg width={W} height={H} style={{position: 'absolute', opacity: 0.16, mixBlendMode: 'overlay'}}>
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves={2} seed={f % 30} stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#grain)" />
      </svg>
      <AbsoluteFill
        style={{background: 'radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.5) 85%, rgba(0,0,0,0.85) 100%)'}}
      />
      {/* thin scanlines */}
      <AbsoluteFill
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, rgba(0,0,0,0.12) 0 1px, transparent 1px 4px)',
          opacity: 0.35,
        }}
      />
    </AbsoluteFill>
  );
};

/** Viewfinder HUD: corner brackets, REC, timecode, progress line. */
export const Hud: React.FC = () => {
  const f = useCurrentFrame();
  const sec = Math.floor(f / 30);
  const fr = f % 30;
  const tc = `00:00:${String(sec).padStart(2, '0')}:${String(fr).padStart(2, '0')}`;
  const appear = lin(f, [4, 14], [0, 1]);
  const fadeOut = lin(f, [396, 412], [1, 0]);
  const o = appear * fadeOut;
  const corner = (r: number) => ({
    position: 'absolute' as const,
    width: 54,
    height: 54,
    borderColor: 'rgba(244,241,236,0.75)',
    borderStyle: 'solid',
    borderWidth: 0,
    transform: `rotate(${r}deg)`,
  });
  const small: React.CSSProperties = {
    position: 'absolute',
    fontFamily: FONT.ui,
    fontWeight: 600,
    fontSize: 20,
    letterSpacing: 4,
    color: 'rgba(244,241,236,0.85)',
  };
  return (
    <AbsoluteFill style={{opacity: o, pointerEvents: 'none'}}>
      <div style={{...corner(0), left: 48, top: 44, borderLeftWidth: 3, borderTopWidth: 3}} />
      <div style={{...corner(0), right: 48, top: 44, borderRightWidth: 3, borderTopWidth: 3}} />
      <div style={{...corner(0), left: 48, bottom: 44, borderLeftWidth: 3, borderBottomWidth: 3}} />
      <div style={{...corner(0), right: 48, bottom: 44, borderRightWidth: 3, borderBottomWidth: 3}} />
      <div style={{...small, left: 120, top: 58}}>KAGE GENJUTSU</div>
      <div style={{...small, right: 120, top: 58, display: 'flex', alignItems: 'center', gap: 12}}>
        <span
          style={{width: 14, height: 14, borderRadius: 7, background: C.hot, opacity: Math.floor(f / 12) % 2 ? 0.25 : 1}}
        />
        REC {tc}
      </div>
      <div style={{...small, left: 120, bottom: 58, fontSize: 16}}>SHADOW ▸ MOTION ▸ ILLUSION</div>
      <div style={{...small, right: 120, bottom: 58, fontSize: 16}}>1920×1080 · 30 FPS</div>
      <div style={{position: 'absolute', left: 120, right: 120, bottom: 38, height: 2, background: 'rgba(244,241,236,0.15)'}}>
        <div style={{width: `${(f / 450) * 100}%`, height: '100%', background: C.hot}} />
      </div>
    </AbsoluteFill>
  );
};

/** A word that slams into frame: from huge and blurred to its size, with RGB split on impact. */
export const Slam: React.FC<{
  text: string;
  at: number;
  size: number;
  color?: string;
  style?: React.CSSProperties;
  from?: number;
  shadow?: string;
  ink?: boolean;
}> = ({text, at, size, color = C.white, style, from = 2.6, shadow, ink}) => {
  const f = useCurrentFrame() - at;
  if (f < 0) return null;
  const s = lerp(f, [0, 7], [from, 1], (t) => 1 - (1 - t) ** 4);
  const blur = lin(f, [0, 6], [18, 0]);
  const ab = lin(f, [0, 3, 10], [0, 14, 0]);
  const glitch = f >= 1 && f <= 6;
  const slice = (top: number, h: number, dx: number, color: string) => (
    <span
      aria-hidden
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        color,
        clipPath: `inset(${top}% 0 ${100 - top - h}% 0)`,
        transform: `translateX(${dx}px)`,
      }}
    >
      {text}
    </span>
  );
  return (
    <div
      style={{
        position: 'relative',
        fontFamily: FONT.display,
        fontSize: size,
        lineHeight: 0.9,
        color,
        textTransform: 'uppercase',
        whiteSpace: 'nowrap',
        transform: `scale(${s})`,
        filter: `${blur > 0.2 ? `blur(${blur}px)` : ''} ${ink ? 'url(#ink)' : ''}`.trim() || undefined,
        opacity: lin(f, [0, 2], [0, 1]),
        textShadow: [aberration(ab), shadow].filter((x) => x && x !== 'none').join(', ') || undefined,
        ...style,
      }}
    >
      {text}
      {glitch && (
        <>
          {slice(18 + rand(f) * 20, 9, (rand(f + 1) - 0.5) * 70, color)}
          {slice(55 + rand(f + 2) * 20, 7, (rand(f + 3) - 0.5) * 90, C.hot)}
        </>
      )}
    </div>
  );
};

/** Letters fly in one by one from random places and rotations. */
export const Assemble: React.FC<{text: string; at: number; size: number; color?: string; seed?: number; stagger?: number}> = ({
  text,
  at,
  size,
  color = C.white,
  seed = 1,
  stagger = 2,
}) => {
  const f = useCurrentFrame() - at;
  return (
    <div style={{display: 'flex', fontFamily: FONT.display, fontSize: size, lineHeight: 0.9, color, whiteSpace: 'pre'}}>
      {text.split('').map((ch, i) => {
        const t = f - i * stagger;
        const p = lerp(t, [0, 14], [0, 1], (x) => 1 - (1 - x) ** 3);
        const dx = (rand(seed * 31 + i) - 0.5) * 1600;
        const dy = (rand(seed * 17 + i * 3) - 0.5) * 900;
        const rot = (rand(seed * 7 + i * 5) - 0.5) * 140;
        return (
          <span
            key={i}
            style={{
              display: 'inline-block',
              transform: `translate(${dx * (1 - p)}px, ${dy * (1 - p)}px) rotate(${rot * (1 - p)}deg) scale(${1 + (1 - p) * 1.5})`,
              opacity: lin(t, [0, 4], [0, 1]),
              filter: p < 0.98 ? `blur(${(1 - p) * 14}px)` : undefined,
            }}
          >
            {ch}
          </span>
        );
      })}
    </div>
  );
};

/** Full-frame flash (white or crimson) that decays. */
export const Flash: React.FC<{at: number; len?: number; color?: string; max?: number}> = ({at, len = 6, color = '#fff', max = 1}) => {
  const f = useCurrentFrame() - at;
  if (f < 0 || f >= len) return null;
  return <AbsoluteFill style={{background: color, opacity: max * (1 - f / len) ** 2, pointerEvents: 'none'}} />;
};

/** Radial speed lines rushing outward (anime-style). */
export const SpeedLines: React.FC<{count?: number; intensity?: number}> = ({count = 46, intensity = 1}) => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{pointerEvents: 'none', opacity: intensity}}>
      {Array.from({length: count}).map((_, i) => {
        const ang = rand(i * 3.3) * 360;
        const speed = 60 + rand(i * 9.1) * 90;
        const dist = ((rand(i * 5.7) * 900 + f * speed) % 1300) + 220;
        const len = 160 + rand(i * 2.2) * 420;
        const red = i % 3 === 0;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: W / 2,
              top: H / 2,
              width: len,
              height: red ? 4 : 2,
              transformOrigin: '0 50%',
              transform: `rotate(${ang}deg) translateX(${dist}px)`,
              background: `linear-gradient(90deg, transparent, ${red ? C.hot : 'rgba(255,255,255,0.8)'})`,
            }}
          />
        );
      })}
    </AbsoluteFill>
  );
};

/** Real smoke footage (white on black) laid over the frame with a screen blend. */
export const Smoke: React.FC<{src?: string; opacity?: number; tint?: boolean; style?: React.CSSProperties}> = ({
  src = 'clips/smoke.mp4',
  opacity = 0.5,
  tint = true,
  style,
}) => {
  const {keyMode} = useMode();
  if (keyMode) return null;
  return (
    <AbsoluteFill style={{mixBlendMode: 'screen', opacity, pointerEvents: 'none', ...style}}>
      <OffthreadVideo src={staticFile(src)} muted style={{width: '100%', height: '100%', objectFit: 'cover'}} />
      {tint && <AbsoluteFill style={{background: C.crimson, mixBlendMode: 'multiply', opacity: 0.55}} />}
    </AbsoluteFill>
  );
};

/** Dark graphic background: near-black, drifting smoke, a faint crimson glow low in the frame. */
export const InkBackground: React.FC = () => (
  <AbsoluteFill style={{background: C.black}}>
    <AbsoluteFill style={{background: 'radial-gradient(ellipse at 50% 110%, rgba(139,0,0,0.35) 0%, transparent 60%)'}} />
    <Smoke src="clips/smoke2.mp4" opacity={0.35} />
  </AbsoluteFill>
);

/** Text that rises out from behind an invisible line (mask reveal). */
export const Reveal: React.FC<{at: number; children: React.ReactNode; style?: React.CSSProperties; dur?: number}> = ({
  at,
  children,
  style,
  dur = 10,
}) => {
  const f = useCurrentFrame() - at;
  const p = lerp(f, [0, dur], [0, 1]);
  return (
    <div style={{overflow: 'hidden', ...style}}>
      <div style={{transform: `translateY(${(1 - p) * 110}%)`}}>{children}</div>
    </div>
  );
};

/** Zoom-through: the outgoing scene rushes into the lens, the incoming one lands from big to normal. */
export const zoomOut = (f: number, start: number, len = 8) => {
  const p = lerp(f, [start, start + len], [0, 1], (t) => t * t * t);
  return {transform: `scale(${1 + p * 3})`, filter: p > 0.02 ? `blur(${p * 30}px)` : undefined, opacity: 1 - p * 0.6};
};
export const zoomIn = (f: number, len = 9) => {
  const p = lerp(f, [0, len], [1, 0], (t) => 1 - (1 - t) ** 3);
  return {transform: `scale(${1 + p * 0.6})`, filter: p > 0.02 ? `blur(${p * 26}px)` : undefined};
};
