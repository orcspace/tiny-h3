import React from 'react';
import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame} from 'remotion';
import {Flash, Reveal, Slot, Smoke} from '../components';
import {LightSweep, Shockwave, SliceText, Stagger} from '../motion';
import {C, FONT, lerp, lin, shake, useMode} from '../lib';

// 11.9–15.0 s: black ink blooms over the last shot and swallows it, the blood moon rises out of the dark,
// the logo assembles letter by letter.
export const S6Logo: React.FC = () => {
  const f = useCurrentFrame();
  const {keyMode} = useMode();
  const sh = shake(f, [18, 30, 81], 18);
  const pulse = 0.55 + 0.25 * Math.sin(f / 5);
  const push = lin(f, [16, 94], [1.0, 1.08]);
  const line = lerp(f, [60, 74], [0, 1]);
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
            <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', transform: `scale(${1 + Math.sin(f / 9) * 0.008})`}}>
              {/* small brand line above, like the "Higgsfield" mark */}
              <div style={{display: 'flex', alignItems: 'center', gap: 14, marginBottom: 6}}>
                <div
                  style={{
                    width: 34,
                    height: 34,
                    borderRadius: 17,
                    border: `5px solid ${C.hot}`,
                    transform: `scale(${lerp(f, [30, 38], [0, 1], (t) => 1 - (1 - t) ** 3)}) rotate(${f * 6}deg)`,
                    borderRightColor: 'transparent',
                  }}
                />
                <Stagger text="KAGE" at={32} style={{fontFamily: FONT.ui, fontWeight: 800, fontSize: 52, color: C.white, letterSpacing: 4}} />
              </div>
              <div style={{filter: `drop-shadow(0 0 ${30 * pulse}px rgba(255,31,53,0.55))`}}>
                <SliceText text="GENJUTSU" at={36} size={190} slices={7} glitchAt={[62, 81, 88]} />
              </div>
              <div style={{alignSelf: 'flex-end', marginTop: 4, marginRight: 8}}>
                <Reveal at={56}>
                  <div style={{fontFamily: FONT.wide, fontStyle: 'italic', fontSize: 50, color: C.white, transform: 'skewX(-10deg)'}}>
                    TRY IT <span style={{color: C.hot}}>NOW</span>
                  </div>
                </Reveal>
              </div>
              <div
                style={{
                  marginTop: 22,
                  height: 4,
                  width: 1000 * line,
                  background: `linear-gradient(90deg, transparent, ${C.hot}, transparent)`,
                  boxShadow: `0 0 20px ${C.hot}`,
                }}
              />
              <Reveal at={66} style={{marginTop: 18}}>
                <div style={{fontFamily: FONT.ui, fontWeight: 700, fontSize: 30, color: C.white, letterSpacing: lerp(f, [66, 90], [36, 14])}}>
                  BECOME THE ILLUSION
                </div>
              </Reveal>
            </div>
          </AbsoluteFill>
        </AbsoluteFill>
      )}
      <Shockwave at={44} />
      <LightSweep at={70} />
      <Flash at={44} len={5} color="#fff" max={0.35} />
      <Flash at={81} len={8} color="#fff" max={0.3} />
    </AbsoluteFill>
  );
};
