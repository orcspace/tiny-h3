import React from 'react';
import {AbsoluteFill, Easing, useCurrentFrame} from 'remotion';
import {Slam, Slot, SpeedLines, zoomOut} from '../components';
import {LightSweep, Shockwave, Stagger} from '../motion';
import {C, FONT, W, lerp, lin, punch, shake} from '../lib';

// 2.1–4.5 s: whip in, speed-ramped sprint, EVERY / SHADOW / HIDES A STORY on the beats.
export const S2Sprint: React.FC = () => {
  const f = useCurrentFrame();
  const hits = [15, 28, 41, 54];
  const sh = shake(f, hits, 22);
  const whipIn = lerp(f, [0, 10], [1, 0], Easing.out(Easing.cubic));
  const ramp = lin(f, [0, 72], [0, 1]) ** 2.2; // slow, then rushing
  const zoom = (1.04 + ramp * 0.32) * punch(f, hits, 0.08);
  const under = lerp(f, [44, 52], [0, 1]);

  return (
    <AbsoluteFill style={{background: C.black, ...zoomOut(f, 62)}}>
    <AbsoluteFill
      style={{
        background: C.black,
        transform: `translateX(${whipIn * W * 0.7}px)`,
        filter: whipIn > 0.02 ? `blur(${whipIn * 34}px)` : undefined,
      }}
    >
      <AbsoluteFill style={{transform: `translate(${sh.x}px, ${sh.y}px) rotate(${sh.r}deg) scale(${zoom})`}}>
        <Slot id="02" label="SPRINT THROUGH THE RED TUNNEL" />
      </AbsoluteFill>
      <SpeedLines intensity={lin(f, [8, 30, 72], [0, 0.7, 1])} />
      <AbsoluteFill style={{background: 'linear-gradient(90deg, rgba(0,0,0,0.7) 0%, transparent 55%)'}} />

      <div style={{position: 'absolute', left: 130, top: 250, transform: `translate(${sh.x * 0.4}px, ${sh.y * 0.4}px)`}}>
        <Slam text="EVERY" at={15} size={250} color={C.white} style={{transformOrigin: '0 50%'}} />
        <Slam
          text="SHADOW"
          at={28}
          size={250}
          color={C.crimson}
          ink
          shadow="0 0 40px rgba(176,0,32,0.6)"
          style={{transformOrigin: '0 50%'}}
        />
        {f >= 41 && (
          <div style={{marginTop: 26, opacity: lin(f, [41, 45], [0, 1])}}>
            <div
              style={{
                fontFamily: FONT.ui,
                fontWeight: 800,
                fontSize: 46,
                letterSpacing: lerp(f, [41, 60], [30, 12]),
                color: C.white,
              }}
            >
              <Stagger text="HIDES A STORY" at={41} />
            </div>
            <div style={{height: 6, width: 520 * under, background: C.hot, marginTop: 12, boxShadow: `0 0 18px ${C.hot}`}} />
          </div>
        )}
      </div>
      <Shockwave at={28} x={520} y={420} />
      <LightSweep at={50} />
    </AbsoluteFill>
    </AbsoluteFill>
  );
};
