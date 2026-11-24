import '@fontsource/anton/400.css';
import '@fontsource/archivo-black/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/inter/800.css';
import '@fontsource/inter/900.css';
import React from 'react';
import {Composition, continueRender, delayRender} from 'remotion';
import {KageGenjutsu} from './KageGenjutsu';
import {DURATION, FPS, H, W} from './lib';

// Wait for the fonts so no frame is rendered with a fallback face.
const handle = delayRender('fonts');
Promise.all(
  ['400 100px Anton', '400 100px "Archivo Black"', '500 40px Inter', '700 40px Inter', '800 40px Inter', '900 40px Inter'].map((f) => document.fonts.load(f)),
).then(() => continueRender(handle), () => continueRender(handle));

export const RemotionRoot: React.FC = () => (
  <>
    {/* full look: grade, grain, vignette; green slots show their names */}
    <Composition
      id="KageGenjutsu"
      component={KageGenjutsu}
      durationInFrames={DURATION}
      fps={FPS}
      width={W}
      height={H}
      defaultProps={{keyMode: false, labels: true}}
    />
    {/* clean #00FF00 slots, no grain over them: for chroma key in Premiere / CapCut / DaVinci */}
    <Composition
      id="KageGenjutsuKey"
      component={KageGenjutsu}
      durationInFrames={DURATION}
      fps={FPS}
      width={W}
      height={H}
      defaultProps={{keyMode: true, labels: false}}
    />
  </>
);
