import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT, H, W, lerp, lin, rand} from './lib';

/** Concentric rings bursting from a point on an impact. */
export const Shockwave: React.FC<{at: number; x?: number; y?: number; color?: string; rings?: number}> = ({
  at,
  x = W / 2,
  y = H / 2,
  color = C.hot,
  rings = 3,
}) => {
  const f = useCurrentFrame() - at;
  if (f < 0 || f > 30) return null;
  return (
    <svg width={W} height={H} style={{position: 'absolute', inset: 0, pointerEvents: 'none'}}>
      {Array.from({length: rings}).map((_, i) => {
        const t = f - i * 3;
        if (t < 0) return null;
        const r = lerp(t, [0, 20], [20, 900 + i * 200], (k) => 1 - (1 - k) ** 3);
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={r}
            fill="none"
            stroke={i === 1 ? '#fff' : color}
            strokeWidth={lin(t, [0, 20], [10, 1])}
            opacity={lin(t, [0, 4, 20], [0, 0.9, 0])}
            style={{filter: `drop-shadow(0 0 12px ${color})`}}
          />
        );
      })}
    </svg>
  );
};

/** A bright diagonal light bar sweeping across the frame (screen blend). */
export const LightSweep: React.FC<{at: number; len?: number; color?: string}> = ({at, len = 14, color = 'rgba(255,255,255,0.55)'}) => {
  const f = useCurrentFrame() - at;
  if (f < 0 || f > len) return null;
  const x = lerp(f, [0, len], [-0.4, 1.2]) * W;
  return (
    <AbsoluteFill style={{mixBlendMode: 'screen', pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          left: x,
          top: -300,
          width: 260,
          height: H + 600,
          transform: 'rotate(18deg)',
          background: `linear-gradient(90deg, transparent, ${color}, rgba(255,31,53,0.45), transparent)`,
          filter: 'blur(18px)',
        }}
      />
    </AbsoluteFill>
  );
};

/** Letters pop up one by one with a springy overshoot. */
export const Stagger: React.FC<{text: string; at: number; style?: React.CSSProperties; step?: number}> = ({text, at, style, step = 1.2}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  return (
    <div style={{display: 'flex', whiteSpace: 'pre', ...style}}>
      {text.split('').map((ch, i) => {
        const p = spring({frame: f - at - i * step, fps, config: {damping: 10, stiffness: 180, mass: 0.6}});
        return (
          <span
            key={i}
            style={{display: 'inline-block', transform: `translateY(${(1 - p) * 60}px) scale(${0.4 + 0.6 * p})`, opacity: Math.min(1, p * 1.5)}}
          >
            {ch}
          </span>
        );
      })}
    </div>
  );
};

/**
 * The Higgsfield-style logo: a wide heavy word cut into horizontal slices that slide in from alternating sides,
 * lock together, then glitch on the beats. The bottom slices carry a crimson-to-black gradient fill.
 */
export const SliceText: React.FC<{
  text: string;
  at: number;
  size: number;
  slices?: number;
  glitchAt?: number[];
  color?: string;
}> = ({text, at, size, slices = 7, glitchAt = [], color = C.white}) => {
  const f = useCurrentFrame() - at;
  const {fps} = useVideoConfig();
  if (f < 0) return null;
  const band = 100 / slices;
  const height = size * 1.05;
  return (
    <div style={{position: 'relative', height, fontFamily: FONT.wide, fontSize: size, lineHeight: 1.05, letterSpacing: -2}}>
      {/* invisible copy to size the box */}
      <div style={{visibility: 'hidden', whiteSpace: 'nowrap'}}>{text}</div>
      {Array.from({length: slices}).map((_, i) => {
        const p = spring({frame: f - i * 1.3, fps, config: {damping: 13, stiffness: 160, mass: 0.7}});
        const dir = i % 2 ? 1 : -1;
        let dx = (1 - p) * dir * (900 + i * 80);
        for (const g of glitchAt) {
          const d = f + at - g;
          if (d >= 0 && d < 5) dx += (rand(i * 13 + g) - 0.5) * 80 * (1 - d / 5);
        }
        const lower = i >= slices - 2;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              inset: 0,
              whiteSpace: 'nowrap',
              clipPath: `inset(${i * band}% 0 ${100 - (i + 1) * band - 0.4}% 0)`,
              transform: `translateX(${dx}px)`,
              opacity: lin(f - i * 1.3, [0, 3], [0, 1]),
              ...(lower
                ? {
                    background: `linear-gradient(180deg, ${C.white} 0%, ${C.white} 62%, ${C.hot} 76%, ${C.deep} 100%)`,
                    WebkitBackgroundClip: 'text',
                    backgroundClip: 'text',
                    color: 'transparent',
                  }
                : {color}),
              filter: p < 0.97 ? `blur(${(1 - p) * 10}px)` : undefined,
            }}
          >
            {text}
          </div>
        );
      })}
    </div>
  );
};
