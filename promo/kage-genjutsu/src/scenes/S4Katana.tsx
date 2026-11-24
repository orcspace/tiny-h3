import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Flash, Slam, Slot} from '../components';
import {LightSweep, Shockwave, Stagger} from '../motion';
import {C, FONT, W, lerp, lin, shake} from '../lib';

const CUTS = [
  {id: '04A', label: 'KATANA DRAWN', word: 'KEEP', at: 0},
  {id: '04B', label: 'SLASH THROUGH PETALS', word: 'EVERY', at: 24},
  {id: '04C', label: 'INK BURST', word: 'MOVE', at: 48},
];

// 7.4–9.9 s: three hard cuts on the beat, each with a flash, a punch-in, shake and one huge word.
export const S4Katana: React.FC = () => {
  const f = useCurrentFrame();
  const idx = f >= 48 ? 2 : f >= 24 ? 1 : 0;
  const cut = CUTS[idx];
  const lf = f - cut.at;
  const sh = shake(f, [0, 9, 24, 35, 48, 61], 30, 8);
  const zoom = lerp(lf, [0, 10], [1.25, 1.02]);
  const bar = lerp(lf, [2, 8], [0, 1]);
  const marquee = (f * 38) % 1800;

  return (
    <AbsoluteFill style={{background: C.black}}>
      <AbsoluteFill style={{transform: `translate(${sh.x}px, ${sh.y}px) rotate(${sh.r}deg) scale(${zoom})`}}>
        <Slot id={cut.id} label={cut.label} />
      </AbsoluteFill>

      {/* outlined kinetic marquee behind the word */}
      <div
        style={{
          position: 'absolute',
          top: 120,
          left: -marquee,
          whiteSpace: 'nowrap',
          fontFamily: FONT.display,
          fontSize: 200,
          color: 'transparent',
          WebkitTextStroke: `3px ${C.crimson}`,
          opacity: 0.85,
        }}
      >
        {'KEEP EVERY MOVE — KEEP EVERY MOVE — KEEP EVERY MOVE — KEEP EVERY MOVE — '}
      </div>

      <AbsoluteFill style={{background: 'linear-gradient(0deg, rgba(0,0,0,0.75) 0%, transparent 55%)'}} />

      {/* word on a crimson bar that wipes in */}
      <div style={{position: 'absolute', left: 120, bottom: 130, transform: `translate(${sh.x * 0.6}px, ${sh.y * 0.6}px)`}}>
        <div
          style={{
            position: 'absolute',
            left: -30,
            top: 40,
            height: 250,
            width: (cut.word.length * 150 + 60) * bar,
            background: C.crimson,
            transform: 'skewX(-12deg)',
            boxShadow: `0 0 50px rgba(176,0,32,0.7)`,
          }}
        />
        <Slam key={cut.word} text={cut.word} at={cut.at + 1} size={330} color={C.white} style={{position: 'relative', transformOrigin: '0 60%'}} />
      </div>

      <div
        style={{
          position: 'absolute',
          right: 130,
          bottom: 150,
          fontFamily: FONT.ui,
          fontWeight: 800,
          fontSize: 40,
          color: C.white,
          letterSpacing: 6,
          opacity: lin(lf, [3, 7], [0, 1]),
        }}
      >
        0{idx + 1} / 03
      </div>

      {CUTS.map((c) => (
        <Shockwave key={'s' + c.id} at={c.at + 2} x={420} y={1080 - 260} rings={2} />
      ))}
      {CUTS.map((c) => (
        <Flash key={c.id} at={c.at} len={4} color={c.at === 24 ? C.hot : '#fff'} max={0.9} />
      ))}
      {/* white frame into the next scene */}
      {f >= 72 && <AbsoluteFill style={{background: '#fff'}} />}
      <div style={{position: 'absolute', width: W, height: 0}} />
    </AbsoluteFill>
  );
};
