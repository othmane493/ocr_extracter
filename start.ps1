# Démarrage Windows : Python, pip (min), Tesseract (+ langues fra/ara), requirements.txt, app.py
# Exécution : PowerShell -ExecutionPolicy Bypass -File .\start.ps1
# Ou double-clic sur start.bat
# Sous Windows, les dictionnaires fra/ara s’ajoutent via l’installateur (pas d’équivalent fiable en ligne de commande ici).

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$MinPipMajor = if ($env:MIN_PIP_MAJOR) { [int]$env:MIN_PIP_MAJOR } else { 23 }
$MinPipMinor = if ($env:MIN_PIP_MINOR) { [int]$env:MIN_PIP_MINOR } else { 0 }

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host $Message -ForegroundColor Cyan
}

function Get-PythonLaunchInfo {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Exe = "py"; ArgsPrefix = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Exe = "python"; ArgsPrefix = @() }
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        return @{ Exe = "python3"; ArgsPrefix = @() }
    }
    return $null
}

function Invoke-Py {
    param(
        [Parameter(Mandatory)] [hashtable] $Launch,
        [Parameter(Mandatory)] [string[]] $Args
    )
    $all = $Launch.ArgsPrefix + $Args
    & $Launch.Exe @all
    if ($LASTEXITCODE -ne 0) {
        throw "Échec (code $LASTEXITCODE): $($Launch.Exe) $($all -join ' ')"
    }
}

function Ensure-TesseractWindows {
    $tessCmd = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($tessCmd) {
        return $tessCmd.Source
    }
    $candidates = @(
        Join-Path ${env:ProgramFiles} "Tesseract-OCR\tesseract.exe"
        Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR\tesseract.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p) {
            $binDir = Split-Path -Parent $p
            $env:Path = "$binDir;$env:Path"
            Write-Host "Tesseract trouvé : $p (répertoire ajouté au PATH de cette session)" -ForegroundColor DarkGray
            return $p
        }
    }
    return $null
}

function Test-TesseractFraAra {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $lines = @(& tesseract --list-langs 2>&1)
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) {
        return @{ Ok = $false; HasFra = $false; HasAra = $false }
    }
    $codes = $lines | Select-Object -Skip 1 | ForEach-Object { "$_".Trim().ToLowerInvariant() } | Where-Object { $_ }
    $hasFra = $codes -contains "fra"
    $hasAra = $codes -contains "ara"
    return @{ Ok = ($hasFra -and $hasAra); HasFra = $hasFra; HasAra = $hasAra }
}

Write-Host "==========================================" -ForegroundColor White
Write-Host "API d'extraction de documents marocains (Windows)" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor White

Write-Step "1. Python"
$launch = Get-PythonLaunchInfo
if (-not $launch) {
    Write-Host ""
    Write-Host "Erreur : Python 3 introuvable (commandes 'py', 'python' ou 'python3')." -ForegroundColor Red
    Write-Host "Installez Python depuis https://www.python.org/downloads/ (cocher 'Add to PATH')." -ForegroundColor Yellow
    exit 1
}
Write-Host "Interpréteur : $($launch.Exe) $($launch.ArgsPrefix -join ' ')"
& $launch.Exe @($launch.ArgsPrefix + @("--version"))
if ($LASTEXITCODE -ne 0) { exit 1 }

$pipVersionScript = @"
import re, subprocess, sys
need = ($MinPipMajor, $MinPipMinor)
r = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True)
out = (r.stdout or '') + (r.stderr or '')
m = re.search(r'pip (\d+)\.(\d+)', out)
if not m:
    print('Impossible de lire la version de pip.', file=sys.stderr)
    sys.exit(1)
maj, minv = int(m.group(1)), int(m.group(2))
if (maj, minv) < need:
    print(f'pip {maj}.{minv} < requis {need[0]}.{need[1]}', file=sys.stderr)
    sys.exit(2)
sys.exit(0)
"@

Write-Step "2. pip (minimum $MinPipMajor.$MinPipMinor)"
& $launch.Exe @($launch.ArgsPrefix + @("-c", $pipVersionScript))
$pipEc = $LASTEXITCODE
if ($pipEc -eq 2) {
    Write-Host "Mise à jour de pip..."
    & $launch.Exe @($launch.ArgsPrefix + @("-m", "pip", "install", "--upgrade", "pip>=${MinPipMajor}.${MinPipMinor}"))
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $launch.Exe @($launch.ArgsPrefix + @("-c", $pipVersionScript))
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Erreur : pip reste insuffisant après mise à jour." -ForegroundColor Red
        exit 1
    }
} elseif ($pipEc -ne 0) {
    Write-Host "Erreur : impossible de vérifier pip." -ForegroundColor Red
    exit 1
}
& $launch.Exe @($launch.ArgsPrefix + @("-m", "pip", "--version"))

Write-Step "3. Tesseract OCR"
$tessPath = Ensure-TesseractWindows
if (-not $tessPath) {
    Write-Host ""
    Write-Host "Tesseract OCR est introuvable sur ce système." -ForegroundColor Red
    Write-Host ""
    Write-Host "Installez d'abord Tesseract pour Windows (tesseract.exe), par exemple :" -ForegroundColor Yellow
    Write-Host "  https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Pendant l'installation :" -ForegroundColor Yellow
    Write-Host "  - À l'étape « Additional language data » (ou équivalent), cochez French et Arabic" -ForegroundColor Yellow
    Write-Host "    (codes Tesseract : fra et ara — requis par l'API)" -ForegroundColor Yellow
    Write-Host "  - Cochez l'option pour ajouter Tesseract au PATH (recommandé)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Relancez ce script une fois Tesseract installé." -ForegroundColor Yellow
    exit 1
}
$tessRun = Get-Command tesseract -ErrorAction SilentlyContinue
if ($tessRun) {
    & tesseract --version 2>&1 | Select-Object -First 1
} else {
    & $tessPath --version 2>&1 | Select-Object -First 1
}

$langCheck = Test-TesseractFraAra
if (-not $langCheck.Ok) {
    Write-Host ""
    Write-Host "Les données de langue Tesseract **fra** (français) et **ara** (arabe) sont obligatoires." -ForegroundColor Red
    if (-not $langCheck.HasFra) { Write-Host "  — manquant : fra" -ForegroundColor Yellow }
    if (-not $langCheck.HasAra) { Write-Host "  — manquant : ara" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "Sous Windows, réinstallez ou modifiez l'installation Tesseract et ajoutez les langues à l'assistant," -ForegroundColor Yellow
    Write-Host "ou copiez les fichiers .traineddata fra et ara dans le dossier tessdata de Tesseract." -ForegroundColor Yellow
    Write-Host "Référence : https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
Write-Host "   Langues Tesseract (fra, ara) : OK" -ForegroundColor DarkGray

Write-Step "4. Dépendances Python (requirements.txt)"
Write-Host "Cela peut prendre plusieurs minutes (notamment PaddleOCR)..."
Invoke-Py -Launch $launch -Args @("-m", "pip", "install", "-r", "requirements.txt")

Write-Step "5. Dossiers locaux"
New-Item -ItemType Directory -Force -Path "uploads", "images" | Out-Null

Write-Step "6. Lancement de l'API sur http://0.0.0.0:5000"
Write-Host "Ctrl+C pour arrêter" -ForegroundColor DarkGray
Write-Host "==========================================" -ForegroundColor White
& $launch.Exe @($launch.ArgsPrefix + @("app.py"))
exit $LASTEXITCODE
