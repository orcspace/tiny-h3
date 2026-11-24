/**
 * Clips used in the film (public/clips/). Set a slot to `null` to get pure green (#00FF00) there instead,
 * or replace a file with your own TinyH3 generation (same name, 30 fps).
 *
 * look: 'grade' (house grade) | 'ink' (red ink warrior, for the OUTPUT card) | 'mono' (dark b/w) | 'blood' (red moon)
 * startFrom: first frame of the clip (30 fps); volume: 0..1 (stock clips are silent).
 *
 * Stock footage: Mixkit (mixkit.co), Mixkit License — free for commercial and personal use, no attribution needed.
 * Check each clip page if you publish commercially: a few Mixkit clips carry a personal-use "Restricted" license.
 */
export type Clip = {
  src: string;
  startFrom?: number;
  volume?: number;
  look?: 'grade' | 'ink' | 'mono' | 'blood';
  mirror?: boolean;
  fit?: 'cover' | 'contain';
} | null;

export const CLIPS: Record<string, Clip> = {
  '01': {src: 'clips/01.mp4', look: 'mono'}, // smoky silhouette of a man        (Mixkit #434)
  '02': {src: 'clips/02.mp4'}, // red neon signs at night                    (Mixkit #4447)
  '03A': {src: 'clips/03.mp4', fit: 'contain'}, // INPUT: karate kick on black               (Mixkit #49629)
  '03B': {src: 'clips/03.mp4', look: 'ink', mirror: true, fit: 'contain'}, // OUTPUT: same moves as a red ink warrior
  '04A': {src: 'clips/04a.mp4', look: 'mono'}, // katana close-up             (Mixkit #13627)
  '04B': {src: 'clips/04b.mp4'}, // karate kata on black       (Mixkit #49635)
  '04C': {src: 'clips/04c.mp4'}, // red smoke on black                       (Mixkit #45597)
  '05A': {src: 'clips/05a.mp4'}, // city at night from above                 (Mixkit #49878)
  '05B': {src: 'clips/05b.mp4', look: 'mono'}, // Tokyo skyline at dusk       (Mixkit #4308)
  '06': {src: 'clips/06.mp4', look: 'blood'}, // full moon in clouds → blood moon (Mixkit #45584)
};

/** Optional music track (140 BPM fits the cuts), e.g. 'music.mp3' in public/. Plays under the built-in sound design. */
export const MUSIC: string | null = null;
export const MUSIC_VOLUME = 0.8;
/** Built-in sound design (drums on the beat, booms, slashes, whooshes from public/sfx). */
export const SFX = true;
