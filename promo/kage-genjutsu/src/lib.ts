import {createContext, useContext} from 'react';
import {Easing, interpolate} from 'remotion';

export const FPS = 30;
export const W = 1920;
export const H = 1080;
export const DURATION = 450; // 15 s

// The edit is cut to a 140 BPM track: beat(n) is the frame of the n-th beat.
export const BPM = 140;
export const beat = (n: number) => Math.round((n * FPS * 60) / BPM);

export const C = {
  black: '#060505',
  ink: '#0e0909',
  blood: '#3d0006',
  deep: '#8b0000',
  crimson: '#b00020',
  hot: '#ff1f35',
  white: '#f4f1ec',
  green: '#00ff00',
};

export const FONT = {
  display: 'Anton, "Bebas Neue", Impact, sans-serif',
  wide: '"Archivo Black", "Arial Black", sans-serif',
  ui: 'Inter, "Helvetica Neue", Arial, sans-serif',
};

/** Full look (grain, grade, vignette) or a clean green-screen master for keying in an editor. */
export const ModeContext = createContext<{keyMode: boolean; labels: boolean}>({keyMode: false, labels: true});
export const useMode = () => useContext(ModeContext);

/** Deterministic pseudo random in [0, 1). */
export const rand = (seed: number) => {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
};

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

export const lerp = (f: number, input: number[], output: number[], ease = Easing.bezier(0.22, 1, 0.36, 1)) =>
  interpolate(f, input, output, {...clamp, easing: ease});

export const lin = (f: number, input: number[], output: number[]) => interpolate(f, input, output, clamp);

/** Camera shake that decays after every hit frame. */
export function shake(frame: number, hits: number[], amp = 26, len = 9) {
  let x = 0;
  let y = 0;
  let r = 0;
  for (const h of hits) {
    const d = frame - h;
    if (d >= 0 && d < len) {
      const k = (1 - d / len) ** 2;
      x += (rand(frame * 3.1 + h) - 0.5) * 2 * amp * k;
      y += (rand(frame * 7.7 + h * 1.3) - 0.5) * 2 * amp * k;
      r += (rand(frame * 1.9 + h * 2.1) - 0.5) * 2 * 1.4 * k;
    }
  }
  return {x, y, r};
}

/** Punch zoom: a quick scale-up on every hit that eases back. */
export function punch(frame: number, hits: number[], amount = 0.12, len = 10) {
  let s = 1;
  for (const h of hits) {
    const d = frame - h;
    if (d >= 0 && d < len) s += amount * (1 - d / len) ** 3;
  }
  return s;
}

/** 0..1 impulse on hit frames (for flashes and chromatic aberration). */
export function impulse(frame: number, hits: number[], len = 6) {
  let v = 0;
  for (const h of hits) {
    const d = frame - h;
    if (d >= 0 && d < len) v = Math.max(v, (1 - d / len) ** 2);
  }
  return v;
}

export const aberration = (px: number) =>
  px < 0.3 ? 'none' : `${px}px 0 rgba(255,20,50,0.85), ${-px}px 0 rgba(0,230,255,0.6)`;
