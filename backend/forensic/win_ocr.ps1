<#
    TRUSTTRACE Windows Native Media OCR Engine
    Invokes Windows.Media.Ocr.OcrEngine via WinRT without requiring external Tesseract binaries.
#>
param (
    [Parameter(Mandatory=$true)]
    [string]$ImagePath
)

$ErrorActionPreference = "Stop"

try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    
    # Generic async task helper for WinRT operations
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

    $resolved = (Resolve-Path $ImagePath).Path
    $fileOp = [Windows.Storage.StorageFile]::GetFileFromPathAsync($resolved)
    $file = Await-Operation $fileOp ([Windows.Storage.StorageFile])

    $streamOp = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read)
    $stream = Await-Operation $streamOp ([Windows.Storage.Streams.IRandomAccessStream])

    $decoderOp = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)
    $decoder = Await-Operation $decoderOp ([Windows.Graphics.Imaging.BitmapDecoder])

    $bitmapOp = $decoder.GetSoftwareBitmapAsync()
    $bitmap = Await-Operation $bitmapOp ([Windows.Graphics.Imaging.SoftwareBitmap])

    # Attempt user profile language recognizer, then fallback to en-US / en-GB
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if (-not $engine) {
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new("en-US"))
    }
    if (-not $engine) {
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new("en-GB"))
    }

    if (-not $engine) {
        $output = @{
            status = "NOT_AVAILABLE"
            engine = "Windows.Media.Ocr"
            error = "No supported OCR language packs found installed in Windows."
        }
        $output | ConvertTo-Json -Compress
        exit 0
    }

    $ocrOp = $engine.RecognizeAsync($bitmap)
    $ocrResult = Await-Operation $ocrOp ([Windows.Media.Ocr.OcrResult])

    $regions = @()
    if ($ocrResult.Lines) {
        foreach ($line in $ocrResult.Lines) {
            foreach ($word in $line.Words) {
                $regions += @{
                    text = $word.Text
                    bbox = @([int]$word.BoundingRect.X, [int]$word.BoundingRect.Y, [int]$word.BoundingRect.Width, [int]$word.BoundingRect.Height)
                    confidence = $null
                }
            }
        }
    }

    $rawText = if ($ocrResult.Text) { $ocrResult.Text } else { "" }

    $output = @{
        status = "AVAILABLE"
        engine = "Windows.Media.Ocr"
        text = $rawText
        confidence = $null
        regions = $regions
        language = $engine.RecognizerLanguage.LanguageTag
    }

    $output | ConvertTo-Json -Depth 5 -Compress
} catch {
    $err = @{
        status = "FAILED"
        engine = "Windows.Media.Ocr"
        error = $_.Exception.Message
    }
    $err | ConvertTo-Json -Compress
}
