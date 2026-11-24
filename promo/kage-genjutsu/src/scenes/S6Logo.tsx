import React from 'react';
import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame} from 'remotion';
import {Assemble, Flash, Reveal, Slot, Smoke} from '../components';
import {C, FONT, lerp, lin, shake, useMode} from '../lib';

// 11.9–15.0 s: black ink blooms over the last shot and swallows it, the blood moon rises out of the dark,
// the logo assembles letter by letter.
export const S6Logo: React.FC = () => {
  const f = useCurrentFrame();
  const {keyMode} = useMode();
  const sh = shake(f, [18, 30, 81], 18);
  const pulse = 0.55 + 0.25 * Math.sin(f / 5);
  const push = lin(f, [16, 94], [1.0, 1.08]);
  const line = lerp(f, [56, 70], [0, 1]);
  const swallow = lin(f, [4, 16], [0, 1]); // the frame goes to ink-black
  const emerge = lin(f, [16, 30], [0, 1]); // the moon shot comes out of it

  return (
    <AbsoluteFill>
      {/* ink blooming over the previous scene (multiply: white water disappears, black ink stays) */}
      {!keyMode && f < 22 && (
        <AbsoluteFill style={{mixBlendMode: 'multiply', transform: `scale(${2.2 + f * 0.05})`}}>
          <OffthreadVideo src={staticFile('clips/ink.mp4')} muted style={{width: '100%', height: '100%', objectFit: 'cover'}} />
        </AbsoluteFill>
      )}
      <AbsoluteFill style={{background: '#000', opacity: f < 16 ? swallow : 1 - emerge}} />

      {f >= 16 && (
        <AbsoluteFill style={{opacity: emerge, transform: `translate(${sh.x}px, ${sh.y}px) scale(${push})`}}>
          <Slot id="06" label="INK IN WATER · BLOOD MOON" dim={0.25} />
          <Smoke opacity={0.35} />
          <AbsoluteFill
            style={{
              background: `radial-gradient(ellipse at 50% 50%, rgba(0,0,0,0.55) 0%, transparent 55%)`,
            }}
          />
          <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column'}}>
            <div style={{textShadow: `0 0 50px rgba(255,31,53,${pulse}), 0 10px 0 ${C.blood}`}}>
              <Assemble text="KAGE GENJUTSU" at={30} size={220} seed={5} stagger={1.5} />
            </div>
            <div
              style={{
                marginTop: 26,
                height: 5,
                width: 900 * line,
                background: `linear-gradient(90deg, transparent, ${C.hot}, transparent)`,
                boxShadow: `0 0 20px ${C.hot}`,
              }}
            />
            <Reveal at={60} style={{marginTop: 22}}>
              <div style={{fontFamily: FONT.ui, fontWeight: 700, fontSize: 40, color: C.white, letterSpacing: lerp(f, [60, 86], [40, 16])}}>
                BECOME THE ILLUSION
              </div>
            </Reveal>
          </AbsoluteFill>
        </AbsoluteFill>
      )}
      <Flash at={81} len={8} color="#fff" max={0.3} />
    </AbsoluteFill>
  );
};
