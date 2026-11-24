import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Flash, Reveal, Slam, Slot, Smoke} from '../components';
import {C, FONT, H, W, lerp, lin, punch, shake} from '../lib';

// 0.0–2.4 s: black, drum hit, the shadow tears off the wall, KAGE slams in. Leaves with a whip pan.
export const S1Shadow: React.FC = () => {
  const f = useCurrentFrame();
  const hits = [4, 26, 51];
  const sh = shake(f, hits, 30);
  const zoom = punch(f, hits, 0.1) * lin(f, [0, 72], [1.12, 1.0]);
  const whip = lerp(f, [62, 72], [0, 1], (t) => t * t);
  const open = lin(f, [0, 4], [0, 1]);
  const line = lerp(f, [10, 24], [0, 1]);

  return (
    <AbsoluteFill
      style={{
        background: C.black,
        transform: `translateX(${-whip * W * 0.7}px)`,
        filter: whip > 0.02 ? `blur(${whip * 34}px)` : undefined,
      }}
    >
      <AbsoluteFill
        style={{opacity: open, transform: `translate(${sh.x}px, ${sh.y}px) rotate(${sh.r}deg) scale(${zoom})`}}
      >
        <Slot id="01" label="SHADOW TEARS OFF THE WALL" />
        <Smoke opacity={0.4} />
        {/* darken bottom for the type */}
        <AbsoluteFill style={{background: 'linear-gradient(180deg, transparent 40%, rgba(0,0,0,0.55) 100%)'}} />
      </AbsoluteFill>

      {/* KAGE */}
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', transform: `translate(${sh.x * 0.5}px, ${sh.y * 0.5}px)`}}>
        <Slam
          text="KAGE"
          at={6}
          size={430}
          color={C.crimson}
          from={3.2}
          ink
          shadow={`0 0 60px rgba(176,0,32,0.65), 0 12px 0 ${C.black}`}
          style={{letterSpacing: 6}}
        />
        <Reveal at={16} style={{position: 'absolute', top: H / 2 + 200}}>
          <div style={{fontFamily: FONT.ui, fontWeight: 700, fontSize: 30, letterSpacing: lerp(f, [16, 44], [40, 16]), color: C.white}}>
            THE SHADOW ARTS
          </div>
        </Reveal>
      </AbsoluteFill>

      {/* a red blade of light crossing the frame */}
      <div
        style={{
          position: 'absolute',
          top: H * 0.5 - 2,
          left: 0,
          height: 4,
          width: W * line,
          background: `linear-gradient(90deg, transparent, ${C.hot})`,
          boxShadow: `0 0 24px ${C.hot}`,
          opacity: lin(f, [24, 34], [1, 0]),
        }}
      />

      <Flash at={4} len={5} />
      <Flash at={26} len={6} color={C.crimson} max={0.45} />
    </AbsoluteFill>
  );
};
