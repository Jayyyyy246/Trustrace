<#
    Batch Windows.Media.Ocr runner for TRUSTTRACE Phase 5
    Takes a manifest CSV path and outputs individual OCR JSONs to data/ocr/
#>
param (
    [Parameter(Mandatory=$true)]
    [string]$ManifestPath,
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,
    [int]$MaxItems = 0
)

$ErrorActionPreference = "Continue"

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { 
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' 
})[0]

function Await-Operation($WinRtOp, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtOp))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $engine) {
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new("en-US"))
}
if (-not $engine) {
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new("en-GB"))
}

if (-not $engine) {
    Write-Error "Windows.Media.Ocr engine unavailable."
    exit 1
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$rows = Import-Csv -Path $ManifestPath
if ($MaxItems -gt 0) {
    $rows = $rows | Select-Object -First $MaxItems
}

$total = $rows.Count
Write-Host "Running batch OCR on $total images..."

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$processed = 0

foreach ($r in $rows) {
    $sampleId = $r.sample_id
    $imgPath = $r.image_path
    $outJsonPath = Join-Path $OutputDir "$($sampleId)_ocr.json"

    if (Test-Path $outJsonPath) {
        $processed++
        continue
    }

    try {
        $resolved = (Resolve-Path $imgPath).Path
        $file = Await-Operation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolved)) ([Windows.Storage.StorageFile])
        $stream = Await-Operation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $decoder = Await-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Await-Operation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $ocrResult = Await-Operation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

        $linesData = @()
        if ($ocrResult.Lines) {
            foreach ($line in $ocrResult.Lines) {
                $wordsData = @()
                foreach ($word in $line.Words) {
                    $wordsData += @{
                        text = $word.Text
                        bbox = @([int]$word.BoundingRect.X, [int]$word.BoundingRect.Y, [int]$word.BoundingRect.Width, [int]$word.BoundingRect.Height)
                    }
                }
                $linesData += @{
                    text = $line.Text
                    words = $wordsData
                }
            }
        }

        $resObj = @{
            sample_id = $sampleId
            status = "AVAILABLE"
            engine = "Windows.Media.Ocr"
            language = $engine.RecognizerLanguage.LanguageTag
            text = if ($ocrResult.Text) { $ocrResult.Text } else { "" }
            lines = $linesData
            word_count = ($linesData | ForEach-Object { $_.words.Count } | Measure-Object -Sum).Sum
            line_count = $linesData.Count
        }

        $resObj | ConvertTo-Json -Depth 6 -Compress | Set-Content -Path $outJsonPath -Encoding UTF8
    } catch {
        $errObj = @{
            sample_id = $sampleId
            status = "FAILED"
            engine = "Windows.Media.Ocr"
            error = $_.Exception.Message
            text = ""
            lines = @()
            word_count = 0
            line_count = 0
        }
        $errObj | ConvertTo-Json -Compress | Set-Content -Path $outJsonPath -Encoding UTF8
    }

    $processed++
    if ($processed % 50 -eq 0 -or $processed -eq $total) {
        $elapsedSec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        $rate = [math]::Round($processed / [math]::Max($elapsedSec, 0.001), 1)
        Write-Host "Processed $processed / $total images ($rate img/sec, ${elapsedSec}s elapsed)"
    }
}

Write-Host "Batch OCR complete in $($sw.Elapsed.TotalSeconds) seconds."
