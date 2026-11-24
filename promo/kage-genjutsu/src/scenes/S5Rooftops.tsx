import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Flash, Slam, Slot} from '../components';
import {C, FONT, H, W, lerp, punch, shake} from '../lib';

// 9.8–12.2 s: diagonal split screen — the ninja on the rooftops / his shadow running ahead. TRANSFER ↔ MOTION.
export const S5Rooftops: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const a = spring({frame: f, fps, config: {damping: 14, stiffness: 120}});
  const b = spring({frame: f - 3, fps, config: {damping: 14, stiffness: 120}});
  const hits = [15, 27, 40, 53];
  const sh = shake(f, hits, 18);
  const z = punch(f, hits, 0.05);
  const line = lerp(f, [4, 14], [0, 1]);
  // the divider runs from (62% , 0) to (38%, 100%)
  const x1 = W * 0.62;
  const x2 = W * 0.38;

  const pill = (text: string, color: string): React.CSSProperties => ({
    position: 'absolute',
    padding: '8px 18px',
    borderRadius: 8,
    background: color,
    fontFamily: FONT.ui,
    fontWeight: 900,
    fontSize: 24,
    letterSpacing: 4,
    color: C.white,
  });

  return (
    <AbsoluteFill style={{background: C.black, transform: `translate(${sh.x}px, ${sh.y}px) rotate(${sh.r}deg) scale(${z})`}}>
      <AbsoluteFill style={{clipPath: 'polygon(0 0, 62% 0, 38% 100%, 0 100%)', transform: `translateY(${(1 - a) * -H}px)`}}>
        <AbsoluteFill style={{transform: `scale(${1.1 + f * 0.0015})`}}>
          <Slot id="05A" label="NINJA ON THE ROOFTOPS" />
        </AbsoluteFill>
      </AbsoluteFill>
      <AbsoluteFill style={{clipPath: 'polygon(62% 0, 100% 0, 100% 100%, 38% 100%)', transform: `translateY(${(1 - b) * H}px)`}}>
        <AbsoluteFill style={{transform: `scale(${1.1 + f * 0.0015})`}}>
          <Slot id="05B" label="HIS SHADOW RUNS AHEAD" />
        </AbsoluteFill>
      </AbsoluteFill>

      <svg width={W} height={H} style={{position: 'absolute', inset: 0}}>
        <line
          x1={x1}
          y1={0}
          x2={x1 + (x2 - x1) * line}
          y2={H * line}
          stroke={C.hot}
          strokeWidth={8}
          style={{filter: `drop-shadow(0 0 16px ${C.hot})`}}
        />
      </svg>

      <div style={{...pill('YOU', 'rgba(0,0,0,0.75)'), left: 130, top: 140, opacity: a}}>YOU</div>
      <div style={{...pill('YOUR SHADOW', C.crimson), right: 130, top: 140, opacity: b}}>YOUR SHADOW</div>

      <div style={{position: 'absolute', left: 120, bottom: 300, transform: `translateX(${(1 - lerp(f, [15, 24], [0, 1])) * -700}px)`}}>
        <Slam text="TRANSFER" at={15} size={210} color={C.white} from={1.6} style={{transformOrigin: '0 50%'}} />
      </div>
      <div style={{position: 'absolute', right: 120, bottom: 110, transform: `translateX(${(1 - lerp(f, [27, 36], [0, 1])) * 700}px)`}}>
        <Slam
          text="MOTION"
          at={27}
          size={210}
          color={C.crimson}
          from={1.6}
          ink
          shadow="0 0 40px rgba(176,0,32,0.6)"
          style={{transformOrigin: '100% 50%'}}
        />
      </div>
      <Flash at={0} len={4} />
    </AbsoluteFill>
  );
};
