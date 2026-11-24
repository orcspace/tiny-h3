import React from 'react';
import {Audio, Sequence, staticFile} from 'remotion';
import {beat} from './lib';

type Cue = {at: number; sfx: string; volume?: number};

// Drums on every beat of the 140 BPM grid while the edit runs, a gap before the logo, then the final boom.
const drums: Cue[] = [];
for (let n = 2; n <= 27; n++) {
  drums.push({at: beat(n), sfx: 'taiko', volume: n % 4 === 0 ? 0.95 : 0.6});
  if (n >= 4) drums.push({at: Math.round((beat(n) + beat(n + 1)) / 2), sfx: 'tick', volume: 0.35});
}

const CUES: Cue[] = [
  {at: 0, sfx: 'sub', volume: 0.9},
  {at: 4, sfx: 'boom', volume: 1},
  {at: 58, sfx: 'whoosh', volume: 0.9}, // whip pan
  {at: 124, sfx: 'whoosh', volume: 0.9}, // zoom-through
  {at: 132, sfx: 'boom', volume: 0.6},
  {at: 190, sfx: 'slash', volume: 1}, // katana cuts MOTION
  {at: 193, sfx: 'boom', volume: 0.8},
  {at: 222, sfx: 'slash', volume: 0.9},
  {at: 246, sfx: 'slash', volume: 0.9},
  {at: 270, sfx: 'slash', volume: 0.9},
  {at: 292, sfx: 'whoosh', volume: 0.8},
  {at: 308, sfx: 'riser', volume: 0.7},
  {at: 356, sfx: 'boom', volume: 0.9}, // ink swallows the frame
  {at: 386, sfx: 'sub', volume: 0.8}, // logo assembles
  {at: 437, sfx: 'boom', volume: 1},
  ...drums,
];

export const SoundDesign: React.FC = () => (
  <>
    {CUES.map((c, i) => (
      <Sequence key={i} from={c.at} layout="none">
        <Audio src={staticFile(`sfx/${c.sfx}.wav`)} volume={c.volume ?? 1} />
      </Sequence>
    ))}
  </>
);
