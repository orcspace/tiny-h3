# Kage Genjutsu: промо-ролик (15 с, Remotion)

```bash
npm install
npm run studio          # редактор в браузере: смотреть и править покадрово
npm run render          # out/kage-genjutsu.mp4 (1920×1080, 30 fps, со звуком)
```

На Linux без Chrome: `npx remotion render KageGenjutsu out/kage-genjutsu.mp4 --browser-executable=/путь/к/chrome`.

- Клипы: `public/clips/` и `src/clips.ts`. Свой клип: положите файл (30 fps) с тем же именем или пропишите путь. `null` даёт зелёный слот.
- Звук: `public/sfx/` (синтезированные удары, свисты, катана), таймкоды в `src/Sound.tsx`. Музыка: `MUSIC` в `src/clips.ts`.
- Сцены: `src/scenes/S1…S6`, тайминги в `src/KageGenjutsu.tsx`, бит 140 BPM в `src/lib.ts`.
- Чистая версия с зелёными слотами без зерна (для кеинга в монтажке): `npm run render:key`.

Стоковые клипы: [Mixkit](https://mixkit.co) (#434, #4447, #49629, #13627, #49635, #45597, #49878, #4308, #45584, #8509, #20465, #8477), Mixkit License. Часть клипов Mixkit идёт под личной (Restricted) лицензией: перед коммерческой публикацией проверьте страницу клипа или замените своими генерациями TinyH3.
Remotion бесплатен для частных лиц и компаний до 3 человек, иначе нужна лицензия (remotion.dev/license).
