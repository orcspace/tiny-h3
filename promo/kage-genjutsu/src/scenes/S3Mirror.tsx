import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Flash, InkBackground, Slam, Slot, zoomIn} from '../components';
import {LightSweep, Shockwave, Stagger} from '../motion';
import {C, FONT, H, W, aberration, lerp, lin, punch, shake} from '../lib';

const CARD_W = 760;
const CARD_H = 500;

const Card: React.FC<{
  id: string;
  label: string;
  tag: string;
  tagColor: string;
  x: number;
  y: number;
  rot: number;
  enter: number;
  fromX: number;
}> = ({id, label, tag, tagColor, x, y, rot, enter, fromX}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = spring({frame: f - enter, fps, config: {damping: 11, stiffness: 140, mass: 0.8}});
  const float = Math.sin((f + enter * 7) / 14) * 6;
  return (
    <div
      style={{
        position: 'absolute',
        left: x - CARD_W / 2,
        top: y - CARD_H / 2,
        width: CARD_W,
        height: CARD_H,
        borderRadius: 30,
        overflow: 'hidden',
        transform: `perspective(1400px) translateX(${(1 - p) * fromX}px) translateY(${float}px) rotateY(${(1 - p) * Math.sign(fromX) * -75 + Math.sin((f + enter) / 18) * 6}deg) rotate(${rot * p + (1 - p) * rot * 3}deg)`,
        boxShadow: `0 40px 80px rgba(0,0,0,0.7), 0 0 0 2px rgba(244,241,236,0.18), 0 0 60px rgba(176,0,32,0.35)`,
        opacity: lin(f - enter, [0, 3], [0, 1]),
      }}
    >
      <Slot id={id} label={label} position="center 30%" />
      <div
        style={{
          position: 'absolute',
          left: 26,
          bottom: 24,
          padding: '8px 18px',
          borderRadius: 8,
          background: tagColor,
          fontFamily: FONT.ui,
          fontWeight: 900,
          fontSize: 26,
          letterSpacing: 4,
          color: C.white,
        }}
      >
        {tag}
      </div>
    </div>
  );
};

// 4.4–7.4 s: INPUT / OUTPUT cards, punches on the beat, MOTION sliced by a katana into ILLUSION.
export const S3Mirror: React.FC = () => {
  const f = useCurrentFrame();
  const SLASH = 61;
  const hits = [22, 35, 48, SLASH];
  const sh = shake(f, [SLASH, 74], 26);
  const groupScale = punch(f, hits, 0.06);
  const slashDraw = lerp(f, [SLASH - 3, SLASH + 1], [0, 1], (t) => t);
  const slashFade = lin(f, [SLASH + 2, SLASH + 12], [1, 0]);
  const split = lerp(f, [SLASH, SLASH + 10], [0, 1]);
  const textTop = H - 250;
  const ab = lin(f, [SLASH, SLASH + 2, SLASH + 10], [0, 16, 0]);

  const motion = (clip: string, dx: number, dy: number) => (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        justifyContent: 'center',
        clipPath: clip,
        transform: `translate(${dx * split}px, ${dy * split}px)`,
        opacity: f < SLASH ? 1 : 1 - split,
      }}
    >
      <div
        style={{
          fontFamily: FONT.display,
          fontSize: 250,
          lineHeight: 1,
          color: C.white,
          letterSpacing: 8,
          textShadow: aberration(ab),
        }}
      >
        MOTION
      </div>
    </div>
  );

  return (
    <AbsoluteFill style={{background: C.black, ...zoomIn(f)}}>
    <AbsoluteFill style={{transform: `translate(${sh.x}px, ${sh.y}px) rotate(${sh.r}deg)`}}>
      <InkBackground />
      <AbsoluteFill style={{transform: `scale(${groupScale})`}}>
        <Card id="03A" label="REAL FIGHTER · KATA" tag="INPUT" tagColor="rgba(0,0,0,0.75)" x={520} y={400} rot={-7} enter={0} fromX={-1300} />
        <Card id="03B" label="INK WARRIOR · SAME MOVES" tag="OUTPUT" tagColor={C.crimson} x={1400} y={420} rot={6} enter={5} fromX={1300} />
        {/* arrow between the cards */}
        <div
          style={{
            position: 'absolute',
            left: W / 2 - 50,
            top: 380,
            fontFamily: FONT.display,
            fontSize: 110,
            color: C.hot,
            opacity: lin(f, [12, 16], [0, 1]),
            transform: `translateX(${Math.sin(f / 4) * 10}px)`,
            textShadow: `0 0 30px ${C.hot}`,
          }}
        >
          ▶
        </div>
      </AbsoluteFill>

      {/* MOTION, then cut in two */}
      <div style={{position: 'absolute', left: 0, right: 0, top: textTop, height: 250}}>
        {f >= 9 && f < SLASH && (
          <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
            <Slam text="MOTION" at={9} size={250} style={{letterSpacing: 8}} />
          </AbsoluteFill>
        )}
        {f >= SLASH && (
          <>
            {motion('polygon(0 0, 100% 0, 100% 40%, 0 62%)', -90, -60)}
            {motion('polygon(0 62%, 100% 40%, 100% 100%, 0 100%)', 90, 60)}
            <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
              <Slam
                text="ILLUSION"
                at={SLASH + 2}
                size={250}
                color={C.crimson}
                from={0.4}
                ink
                shadow={`0 0 50px rgba(255,31,53,0.6)`}
                style={{letterSpacing: 8}}
              />
            </AbsoluteFill>
          </>
        )}
      </div>

      {/* the katana slash */}
      <svg width={W} height={H} style={{position: 'absolute', inset: 0, opacity: slashFade}}>
        <defs>
          <linearGradient id="blade" x1="0" x2="1">
            <stop offset="0" stopColor={C.hot} stopOpacity="0" />
            <stop offset="0.5" stopColor="#fff" />
            <stop offset="1" stopColor={C.hot} stopOpacity="0" />
          </linearGradient>
        </defs>
        <line
          x1={80}
          y1={textTop + 250 * 0.62 + 30}
          x2={80 + (W - 160) * slashDraw}
          y2={textTop + 250 * 0.62 + 30 - (250 * 0.22 + 60) * slashDraw}
          stroke="url(#blade)"
          strokeWidth={9}
          strokeLinecap="round"
          style={{filter: `drop-shadow(0 0 18px ${C.hot})`}}
        />
      </svg>
      <Shockwave at={SLASH + 1} x={W / 2} y={H - 125} />
      <LightSweep at={76} />
      <Flash at={SLASH} len={6} color="#fff" max={0.5} />
    </AbsoluteFill>
    </AbsoluteFill>
  );
};
