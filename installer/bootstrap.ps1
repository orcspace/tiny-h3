# TinyH3 Video: bootstrap. Downloads uv (checks sha256), makes Python 3.12 available and starts the installer.
# Everything the installer asks (folders, profile) happens in tinyh3/installer.py.
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $env:TEMP 'tinyh3-bootstrap'
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Get-Uv {
    $local = Get-Command uv -ErrorAction SilentlyContinue
    if ($local) { return $local.Source }
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
    $name = "uv-$arch-pc-windows-msvc.zip"
    $base = 'https://github.com/astral-sh/uv/releases/latest/download'
    $zip = Join-Path $work $name
    $dir = Join-Path $work 'uv'
    $exe = Join-Path $dir 'uv.exe'
    if (Test-Path $exe) { return $exe }
    Write-Host '  Скачиваю uv (менеджер Python и пакетов)...'
    Invoke-WebRequest "$base/$name" -OutFile $zip -UseBasicParsing
    $sum = (Invoke-WebRequest "$base/$name.sha256" -UseBasicParsing).Content
    $want = ([Text.Encoding]::UTF8.GetString([Text.Encoding]::UTF8.GetBytes($sum)).Trim() -split '\s+')[0].ToLower()
    $have = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($want -ne $have) { throw "Контрольная сумма uv не совпала ($have)" }
    Expand-Archive -Path $zip -DestinationPath $dir -Force
    if (-not (Test-Path $exe)) { $exe = (Get-ChildItem $dir -Recurse -Filter uv.exe | Select-Object -First 1).FullName }
    return $exe
}

try {
    $uv = Get-Uv
    Write-Host "  uv: $uv"
    $env:UV_CACHE_DIR = Join-Path $work 'cache'
    $env:UV_NO_PROGRESS = '1'
    $env:PYTHONPATH = $root
    $env:PYTHONUTF8 = '1'
    & $uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось получить Python 3.12' }
    & $uv run --no-project --python 3.12 python -m tinyh3 install --uv $uv --source $root @Rest
    exit $LASTEXITCODE
}
catch {
    Write-Host ''
    Write-Host "  ОШИБКА: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host '  Проверьте подключение к интернету (нужен доступ к github.com) и запустите install.bat снова.'
    exit 1
}
