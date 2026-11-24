import React from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile} from 'remotion';
import {MUSIC, MUSIC_VOLUME, SFX} from './clips';
import {SoundDesign} from './Sound';
import {Defs, Finish, Hud} from './components';
import {C, ModeContext} from './lib';
import {S1Shadow} from './scenes/S1Shadow';
import {S2Sprint} from './scenes/S2Sprint';
import {S3Mirror} from './scenes/S3Mirror';
import {S4Katana} from './scenes/S4Katana';
import {S5Rooftops} from './scenes/S5Rooftops';
import {S6Logo} from './scenes/S6Logo';

export type Props = {keyMode: boolean; labels: boolean};

// Scene timing in frames (30 fps). Overlaps are the transitions: whip pan 62–72, ink flood 356–372.
export const SCENES = [
  {from: 0, len: 72, C: S1Shadow},
  {from: 62, len: 72, C: S2Sprint},
  {from: 132, len: 92, C: S3Mirror},
  {from: 222, len: 74, C: S4Katana},
  {from: 294, len: 72, C: S5Rooftops},
  {from: 356, len: 94, C: S6Logo},
];

export const KageGenjutsu: React.FC<Props> = ({keyMode, labels}) => (
  <ModeContext.Provider value={{keyMode, labels}}>
    <AbsoluteFill style={{background: C.black}}>
      <Defs />
      {SCENES.map(({from, len, C: Scene}, i) => (
        <Sequence key={i} from={from} durationInFrames={len} layout="absolute-fill">
          <Scene />
        </Sequence>
      ))}
      <Finish />
      <Hud />
      {SFX && <SoundDesign />}
      {MUSIC && <Audio src={staticFile(MUSIC)} volume={MUSIC_VOLUME} />}
    </AbsoluteFill>
  </ModeContext.Provider>
);
