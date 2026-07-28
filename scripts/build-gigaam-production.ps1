[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Push-Location -LiteralPath $repositoryRoot
try {
    $sourceCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}$') {
        throw 'Cannot resolve a full lowercase Git commit for the production image.'
    }

    $workspaceChanges = @(& git status --porcelain=v1 --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) {
        throw 'Cannot inspect the Git worktree.'
    }
    if ($workspaceChanges.Count -gt 0) {
        $preview = ($workspaceChanges | Select-Object -First 20) -join [Environment]::NewLine
        throw "Production GigaAM images require a clean worktree. Commit or remove these changes first:`n$preview"
    }

    $env:VOICEMED_SOURCE_COMMIT = $sourceCommit
    $env:VOICEMED_SOURCE_DIRTY = 'false'

    & docker compose build --pull gigaam
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
