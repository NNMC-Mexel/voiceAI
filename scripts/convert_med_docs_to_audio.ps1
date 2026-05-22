param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,
    [switch]$Overwrite,
    [int]$MaxFiles = 0,
    [string[]]$OnlyFiles = @(),
    [ValidateSet('auto', 'sapi', 'silero')]
    [string]$TtsMode = 'auto',
    [string]$SileroUrl = 'http://127.0.0.1:5500'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Normalize-Text {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ''
    }

    $normalized = $Text -replace "`u0007", "`n"
    $normalized = $normalized -replace "`u000B", "`n"
    $normalized = $normalized -replace '[`u0000-`u0008`u000C`u000E-`u001F]', ' '
    $normalized = $normalized -replace "`r`n?", "`n"
    $normalized = $normalized -replace '[ \t]+', ' '
    $normalized = $normalized -replace ' *`n *', "`n"
    $normalized = $normalized -replace '`n{3,}', "`n`n"

    return $normalized.Trim()
}

function Filter-PatientContent {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ''
    }

    $value = $Text

    $startMatch = [System.Text.RegularExpressions.Regex]::Match(
        $value,
        '(?i)(?:ФИО\s*пациента\b|ФИО\b|Пациент(?:ка)?\b)\s*(?:[:\-]\s*)?'
    )
    if ($startMatch.Success -and $startMatch.Index -gt 0) {
        $value = $value.Substring($startMatch.Index)
    }

    $cutPhrases = @(
        '(?i)Пациенту\s+представлена\s+информация\s+относительно\s+лекарственных\s+препаратов',
        '(?i)С\s+результатами\s+исследования\s+и\s+рекомендациями\s+ознакомлен',
        '(?i)Претензий\s+к\s+мед\s*персоналу'
    )

    foreach ($phrase in $cutPhrases) {
        $m = [System.Text.RegularExpressions.Regex]::Match($value, $phrase)
        if ($m.Success) {
            $value = $value.Substring(0, $m.Index).Trim()
            break
        }
    }

    $lines = @($value -split "`n")
    $firstPatientLine = -1
    for ($idx = 0; $idx -lt $lines.Count; $idx++) {
        $probe = ($lines[$idx] -replace '\s+', ' ').Trim()
        if ($probe -match '(?i)\bФИО\b' -or ($probe -match '(?i)\bПациент' -and $probe -notmatch '(?i)Пациенту представлена информация')) {
            $firstPatientLine = $idx
            break
        }
    }
    if ($firstPatientLine -gt 0) {
        $lines = $lines[$firstPatientLine..($lines.Count - 1)]
    }

    $out = New-Object 'System.Collections.Generic.List[string]'
    $dropPatterns = @(
        '(?i)^\s*ТОО\b',
        '(?i)^\s*ООО\b',
        '(?i)^\s*АО\b',
        '(?i)MEXEL\s+HEALTH',
        '(?i)^\s*Врач\b',
        '(?i)^\s*ФИО\s*врач',
        '(?i)^\s*(Сосудистый\s+хирург|Хирург|Кардиолог|Гинеколог|Отоларинголог|Невролог|Терапевт)\b.*$',
        '(?i)подпись',
        '(?i)^\s*С результатами',
        '(?i)^\s*Претензий к мед',
        '(?i)^\s*Пациенту представлена информация'
    )

    $lineIndex = 0
    foreach ($lineRaw in $lines) {
        $lineIndex++
        $line = $lineRaw.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $drop = $false
        foreach ($pat in $dropPatterns) {
            if ($line -match $pat) {
                $drop = $true
                break
            }
        }

        if (-not $drop -and $lineIndex -le 5 -and $line -match '(?i)^\s*(Осмотр|Прием|Приём)\b') {
            $drop = $true
        }

        if (-not $drop -and $line -match '(?i)Врач') {
            $drop = $true
        }

        $singleLetterTokens = @([System.Text.RegularExpressions.Regex]::Matches($line, '(?<!\p{L})\p{L}(?!\p{L})')).Count
        $allLetterTokens = @([System.Text.RegularExpressions.Regex]::Matches($line, '\p{L}+')).Count
        if ($allLetterTokens -gt 0 -and ($singleLetterTokens / $allLetterTokens) -ge 0.6) {
            $drop = $true
        }

        if (-not $drop) {
            $clean = $line
            $clean = $clean -replace '\.\s*\.', ' '
            $clean = $clean -replace '^(?:\p{L}\s+){2,6}(?=\p{Lu})', ''
            $clean = $clean -replace '\s{2,}', ' '
            $clean = $clean.Trim()
            if (-not [string]::IsNullOrWhiteSpace($clean)) {
                $out.Add($clean)
            }
        }
    }

    return ($out -join "`n").Trim()
}

function Sanitize-PathPart {
    param([string]$Name)

    $value = $Name
    if ([string]::IsNullOrWhiteSpace($value)) {
        return ''
    }

    $value = $value -replace '[<>:"/\\|?*]', '_'
    $value = $value -replace '\s+', ' '
    $value = $value.Trim(' ', '.')

    if ($value.Length -gt 120) {
        $value = $value.Substring(0, 120).Trim(' ', '.')
    }

    return $value
}

function Get-SpecialtyName {
    param(
        [string]$FilePath,
        [string]$RootResolved
    )

    $relative = $FilePath.Substring($RootResolved.Length).TrimStart('\\')
    $parts = $relative -split '\\'

    if ($parts.Count -gt 1 -and -not [string]::IsNullOrWhiteSpace($parts[0])) {
        return $parts[0]
    }

    return [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
}

function Split-PatientRecords {
    param([string]$Text)

    $startPattern = '(?i)(?:ФИО(?:\s*пациента)?\b|Ф\.И\.О\.?|Пациент(?:ка)?\b)\s*(?!врач)(?:[:\-]\s*)?'
    $matches = [System.Text.RegularExpressions.Regex]::Matches($Text, $startPattern)

    if ($matches.Count -le 1) {
        return @($Text)
    }

    $records = New-Object 'System.Collections.Generic.List[string]'

    for ($i = 0; $i -lt $matches.Count; $i++) {
        if ($i -eq 0) {
            $start = 0
        } else {
            $start = $matches[$i].Index
        }

        if ($i -lt ($matches.Count - 1)) {
            $finish = $matches[$i + 1].Index
        } else {
            $finish = $Text.Length
        }

        if ($finish -le $start) {
            continue
        }

        $chunk = $Text.Substring($start, $finish - $start).Trim()
        if ($chunk.Length -ge 120) {
            $records.Add($chunk)
        }
    }

    if ($records.Count -eq 0) {
        return @($Text)
    }

    return $records.ToArray()
}

function Get-PatientName {
    param(
        [string]$BlockText,
        [string]$Fallback,
        [string]$SourceHint = ''
    )

    $patterns = @(
        '(?im)\bФИО\s*пациента\b\s*(?:[:\-]\s*)?(.+)$',
        '(?im)\bФИО\b\s*(?:[:\-]\s*)?(.+)$',
        '(?im)\bПациент(?:ка)?\b\s*(?:[:\-]\s*)?(.+)$'
    )

    $stopWords = @(
        'фио', 'пациента', 'пациент', 'пациентка', 'врач', 'дата', 'рождения',
        'не', 'указано', 'телефон', 'пол', 'возраст'
    )

    $searchArea = if ($BlockText.Length -gt 1200) { $BlockText.Substring(0, 1200) } else { $BlockText }

    foreach ($pattern in $patterns) {
        $match = [System.Text.RegularExpressions.Regex]::Match($searchArea, $pattern)
        if (-not $match.Success) {
            continue
        }

        $line = $match.Groups[1].Value
        $line = $line -replace '[`u0000-`u001F]', ' '
        $line = $line -replace '\s+', ' '
        if ($line.Trim().ToLowerInvariant().StartsWith('врач')) {
            continue
        }

        $words = [System.Text.RegularExpressions.Regex]::Matches($line, '\p{L}[\p{L}\-]*') |
            ForEach-Object { $_.Value }

        $filtered = @()
        foreach ($w in $words) {
            $lw = $w.ToLowerInvariant()
            if ($w.Length -lt 2) {
                continue
            }
            if ($w -notmatch '^\p{Lu}') {
                continue
            }
            if ($stopWords -contains $lw) {
                continue
            }
            $filtered += $w
        }

        if ($filtered.Count -ge 2) {
            $take = [Math]::Min(4, $filtered.Count)
            return (($filtered[0..($take - 1)] -join ' ').Trim(' ', ',', '.'))
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($SourceHint)) {
        $sourceMatch = [System.Text.RegularExpressions.Regex]::Match(
            $SourceHint,
            '(?i)\b(\p{Lu}[\p{L}\-]+(?:\s+\p{Lu}[\p{L}\-]+){2,3})\b'
        )
        if ($sourceMatch.Success) {
            return $sourceMatch.Groups[1].Value.Trim(' ', ',', '.')
        }
    }

    $fallbackMatch = [System.Text.RegularExpressions.Regex]::Match(
        $BlockText,
        '(?im)\b(\p{Lu}[\p{L}\-]+(?:\s+\p{Lu}[\p{L}\-]+){2,3})\b'
    )

    if ($fallbackMatch.Success) {
        return $fallbackMatch.Groups[1].Value.Trim()
    }

    return $Fallback
}

function Split-TextChunks {
    param(
        [string]$InputText,
        [int]$MaxChars = 2600
    )

    $text = ($InputText -replace '\s+', ' ').Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return @()
    }

    $sentences = [System.Text.RegularExpressions.Regex]::Split($text, '(?<=[\.!\?;:])\s+')
    $chunks = New-Object 'System.Collections.Generic.List[string]'
    $builder = New-Object System.Text.StringBuilder

    foreach ($sentenceRaw in $sentences) {
        $sentence = $sentenceRaw.Trim()
        if ([string]::IsNullOrWhiteSpace($sentence)) {
            continue
        }

        if ($sentence.Length -gt $MaxChars) {
            $words = $sentence -split '\s+'
            foreach ($word in $words) {
                $part = if ($builder.Length -eq 0) { $word } else { " $word" }
                if (($builder.Length + $part.Length) -gt $MaxChars -and $builder.Length -gt 0) {
                    $chunks.Add($builder.ToString().Trim())
                    $builder.Clear() | Out-Null
                    $part = $word
                }
                [void]$builder.Append($part)
            }
            continue
        }

        $addition = if ($builder.Length -eq 0) { $sentence } else { " $sentence" }
        if (($builder.Length + $addition.Length) -gt $MaxChars -and $builder.Length -gt 0) {
            $chunks.Add($builder.ToString().Trim())
            $builder.Clear() | Out-Null
            $addition = $sentence
        }

        [void]$builder.Append($addition)
    }

    if ($builder.Length -gt 0) {
        $chunks.Add($builder.ToString().Trim())
    }

    return $chunks.ToArray()
}

function Prepare-TtsText {
    param([string]$InputText)

    if ([string]::IsNullOrWhiteSpace($InputText)) {
        return ''
    }

    $text = $InputText
    $text = $text -replace '[\u2013\u2014]', '-'
    $text = $text -replace '_{2,}', ' '
    $text = $text -replace '[ ]{2,}', ' '
    $text = $text -replace '(?m)^\s*[-=]{2,}\s*$', ''
    $text = $text -replace "(?m)^\s*\d+\s*$", ''
    $text = $text -replace "`r`n?", "`n"
    $text = $text -replace "`n{3,}", "`n`n"

    return $text.Trim()
}

function Write-WavFromText {
    param(
        [System.Speech.Synthesis.SpeechSynthesizer]$Synth,
        [string]$Text,
        [string]$OutputPath
    )

    $ttsText = Prepare-TtsText -InputText $Text
    $chunks = @(Split-TextChunks -InputText $ttsText)
    if ($chunks.Count -eq 0) {
        return
    }

    $Synth.SetOutputToWaveFile($OutputPath)
    try {
        foreach ($chunk in $chunks) {
            $Synth.Speak($chunk)
        }
    }
    finally {
        $Synth.SetOutputToNull()
    }
}

function Write-WavFromSilero {
    param(
        [string]$Text,
        [string]$OutputPath,
        [string]$SileroBaseUrl
    )

    $ttsText = Prepare-TtsText -InputText $Text
    if ([string]::IsNullOrWhiteSpace($ttsText)) {
        return
    }

    $body = @{ text = $ttsText } | ConvertTo-Json -Depth 4
    $resp = Invoke-RestMethod -Method Post -Uri ($SileroBaseUrl.TrimEnd('/') + '/tts') -Body $body -ContentType 'application/json; charset=utf-8' -TimeoutSec 180

    if ($null -eq $resp -or [string]::IsNullOrWhiteSpace($resp.audio_base64)) {
        throw 'Silero TTS returned empty audio payload.'
    }

    $bytes = [Convert]::FromBase64String([string]$resp.audio_base64)
    [System.IO.File]::WriteAllBytes($OutputPath, $bytes)
}

function New-SapiSynth {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer

    $installedVoices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
    if ($installedVoices -contains 'Microsoft Irina Desktop') {
        $s.SelectVoice('Microsoft Irina Desktop')
    }
    else {
        $ruVoice = $s.GetInstalledVoices() |
            Where-Object { $_.VoiceInfo.Culture.Name -eq 'ru-RU' } |
            Select-Object -First 1

        if ($null -ne $ruVoice) {
            $s.SelectVoice($ruVoice.VoiceInfo.Name)
        }
    }

    $s.Rate = -1
    return $s
}

$rootResolved = (Resolve-Path -LiteralPath $RootPath).Path

Add-Type -AssemblyName System.Speech

$word = $null
$synth = $null
$effectiveTtsMode = 'sapi'
$report = New-Object 'System.Collections.Generic.List[object]'

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    if ($TtsMode -in @('auto', 'silero')) {
        try {
            $null = Invoke-RestMethod -Method Get -Uri ($SileroUrl.TrimEnd('/') + '/health') -TimeoutSec 3
            $effectiveTtsMode = 'silero'
            Write-Host "TTS mode: silero ($SileroUrl)"
        }
        catch {
            if ($TtsMode -eq 'silero') {
                throw "Silero TTS requested, but server is not reachable at $SileroUrl"
            }
        }
    }

    if ($effectiveTtsMode -eq 'sapi') {
        $synth = New-SapiSynth
        Write-Host 'TTS mode: sapi (fallback)'
    }

    $sourceFiles = Get-ChildItem -LiteralPath $rootResolved -Recurse -File |
        Where-Object {
            $ext = $_.Extension.ToLowerInvariant()
            $ext -in @('.doc', '.docx', '.pdf') -and $_.BaseName -ne 'source_original'
        } |
        Sort-Object FullName

    $onlyFilesResolved = @($OnlyFiles | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object {
        (Resolve-Path -LiteralPath $_).Path
    })

    if ($onlyFilesResolved.Count -gt 0) {
        $sourceFiles = $sourceFiles | Where-Object { $onlyFilesResolved -contains $_.FullName }
    }

    if ($MaxFiles -gt 0) {
        $sourceFiles = $sourceFiles | Select-Object -First $MaxFiles
    }

    $sourceFiles = @($sourceFiles)

    $total = $sourceFiles.Count
    $current = 0

    foreach ($file in $sourceFiles) {
        $current++
        Write-Host "[$current/$total] $($file.FullName)"

        $text = ''
        $doc = $null

        try {
            $doc = $word.Documents.Open($file.FullName, $false, $true)
            $text = $doc.Content.Text
        }
        catch {
            $report.Add([pscustomobject]@{
                File = $file.FullName
                Status = 'ERROR_OPEN'
                Message = $_.Exception.Message
                Patients = 0
            })
            continue
        }
        finally {
            if ($null -ne $doc) {
                try { $doc.Close($false) | Out-Null } catch {}
                try { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null } catch {}
            }
        }

        $cleanText = Normalize-Text -Text $text
        if ([string]::IsNullOrWhiteSpace($cleanText)) {
            $report.Add([pscustomobject]@{
                File = $file.FullName
                Status = 'EMPTY_TEXT'
                Message = 'No textual content extracted'
                Patients = 0
            })
            continue
        }

        $records = @(Split-PatientRecords -Text $cleanText)
        $specialty = Sanitize-PathPart -Name (Get-SpecialtyName -FilePath $file.FullName -RootResolved $rootResolved)
        if ([string]::IsNullOrWhiteSpace($specialty)) {
            $specialty = 'Неразобранное'
        }

        $specialtyDir = Join-Path $rootResolved $specialty
        if (-not (Test-Path -LiteralPath $specialtyDir)) {
            New-Item -ItemType Directory -Path $specialtyDir | Out-Null
        }

        $patientsProcessed = 0
        $stem = Sanitize-PathPart -Name $file.BaseName
        if ([string]::IsNullOrWhiteSpace($stem)) {
            $stem = 'source'
        }

        for ($i = 0; $i -lt $records.Count; $i++) {
            $record = $records[$i].Trim()
            $record = Filter-PatientContent -Text $record
            if ($record.Length -lt 120) {
                continue
            }

            $fallback = 'Пациент_{0:D3}' -f ($i + 1)
            $patientName = Sanitize-PathPart -Name (Get-PatientName -BlockText $record -Fallback $fallback -SourceHint $file.BaseName)
            if ([string]::IsNullOrWhiteSpace($patientName)) {
                $patientName = $fallback
            }

            $patientDir = Join-Path $specialtyDir $patientName
            if (-not (Test-Path -LiteralPath $patientDir)) {
                New-Item -ItemType Directory -Path $patientDir | Out-Null
            }

            $sourceCopyName = 'source_original' + $file.Extension.ToLowerInvariant()
            $sourceCopyPath = Join-Path $patientDir $sourceCopyName
            if ($Overwrite -or -not (Test-Path -LiteralPath $sourceCopyPath)) {
                Copy-Item -LiteralPath $file.FullName -Destination $sourceCopyPath -Force
            }

            $baseName = '{0}__{1:D2}' -f $stem, ($i + 1)
            $txtPath = Join-Path $patientDir ($baseName + '_text.txt')
            $wavPath = Join-Path $patientDir ($baseName + '_audio.wav')

            if ((-not $Overwrite) -and (Test-Path -LiteralPath $txtPath) -and (Test-Path -LiteralPath $wavPath)) {
                $patientsProcessed++
                continue
            }

            Set-Content -LiteralPath $txtPath -Value $record -Encoding UTF8
            if ($effectiveTtsMode -eq 'silero') {
                try {
                    Write-WavFromSilero -Text $record -OutputPath $wavPath -SileroBaseUrl $SileroUrl
                }
                catch {
                    if ($null -eq $synth) {
                        $synth = New-SapiSynth
                        Write-Host 'Silero failed for one item, switched to local SAPI fallback for this run.'
                    }
                    Write-Warning ("Silero TTS failed for: " + $file.FullName + " | " + $_.Exception.Message)
                    Write-WavFromText -Synth $synth -Text $record -OutputPath $wavPath
                }
            }
            else {
                Write-WavFromText -Synth $synth -Text $record -OutputPath $wavPath
            }
            $patientsProcessed++
        }

        $status = if ($patientsProcessed -gt 0) { 'OK' } else { 'NO_PATIENT_BLOCKS' }
        $report.Add([pscustomobject]@{
            File = $file.FullName
            Status = $status
            Message = ''
            Patients = $patientsProcessed
        })
    }

    $reportPath = Join-Path $rootResolved '_audio_sort_report.csv'
    $report | Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding UTF8

    $okCount = @($report | Where-Object { $_.Status -eq 'OK' }).Count
    $errorCount = @($report | Where-Object { $_.Status -like 'ERROR*' }).Count
    $emptyCount = @($report | Where-Object { $_.Status -eq 'EMPTY_TEXT' }).Count
    $noBlocksCount = @($report | Where-Object { $_.Status -eq 'NO_PATIENT_BLOCKS' }).Count

    $summary = [pscustomobject]@{
        SourceFiles = $total
        OK = $okCount
        Errors = $errorCount
        Empty = $emptyCount
        NoPatientBlocks = $noBlocksCount
        Report = $reportPath
    }

    $summary | Format-List | Out-String | Write-Host
}
finally {
    if ($null -ne $synth) {
        try { $synth.Dispose() } catch {}
    }

    if ($null -ne $word) {
        try { $word.Quit() | Out-Null } catch {}
        try { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null } catch {}
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
