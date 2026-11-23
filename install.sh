#!/bin/sh
# TinyH3 Video: установка на Linux / macOS. Скачивает uv и запускает установщик.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${TMPDIR:-/tmp}/tinyh3-bootstrap"
mkdir -p "$WORK"
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [ -x "$WORK/uv/uv" ]; then
  UV="$WORK/uv/uv"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) T=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) T=aarch64-unknown-linux-gnu ;;
    Darwin-arm64) T=aarch64-apple-darwin ;;
    Darwin-x86_64) T=x86_64-apple-darwin ;;
    *) echo "Платформа не поддерживается: $(uname -s) $(uname -m)"; exit 1 ;;
  esac
  BASE=https://github.com/astral-sh/uv/releases/latest/download
  echo "  Скачиваю uv..."
  curl -fsSL "$BASE/uv-$T.tar.gz" -o "$WORK/uv.tar.gz"
  WANT="$(curl -fsSL "$BASE/uv-$T.tar.gz.sha256" | cut -d' ' -f1)"
  if command -v sha256sum >/dev/null 2>&1; then HAVE="$(sha256sum "$WORK/uv.tar.gz" | cut -d' ' -f1)"; else HAVE="$(shasum -a 256 "$WORK/uv.tar.gz" | cut -d' ' -f1)"; fi
  [ "$WANT" = "$HAVE" ] || { echo "Контрольная сумма uv не совпала"; exit 1; }
  mkdir -p "$WORK/uv"
  tar -xzf "$WORK/uv.tar.gz" -C "$WORK/uv" --strip-components=1
  UV="$WORK/uv/uv"
fi
export UV_CACHE_DIR="$WORK/cache" UV_NO_PROGRESS=1 PYTHONPATH="$ROOT" PYTHONUTF8=1
"$UV" python install 3.12
exec "$UV" run --no-project --python 3.12 python -m tinyh3 install --uv "$UV" --source "$ROOT" "$@"
