param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$JobId,
    [string]$ComposeFile = "docker-compose.yml"
)

$ErrorActionPreference = "Stop"
$project = Join-Path (Join-Path $PSScriptRoot "..\output") $JobId
$config = Join-Path $project ".job-config.json"
$state = Join-Path $project ".job-state.json"

if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "Missing immutable job configuration: $config"
}
if (-not (Test-Path -LiteralPath $state -PathType Leaf)) {
    throw "Missing durable job state: $state"
}

$before = Get-Content -LiteralPath $state -Raw | ConvertFrom-Json
Write-Host "Before restart: status=$($before.status) generation=$($before.execution_generation)"

docker compose -f $ComposeFile kill backend
if ($LASTEXITCODE -ne 0) { throw "docker compose kill failed" }
docker compose -f $ComposeFile up -d --force-recreate backend
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

$deadline = (Get-Date).AddMinutes(3)
do {
    Start-Sleep -Seconds 2
    $container = docker compose -f $ComposeFile ps --status running --quiet backend
} until ($container -or (Get-Date) -ge $deadline)
if (-not $container) { throw "Backend did not become running within three minutes" }

if (-not (Test-Path -LiteralPath $config -PathType Leaf) -or
    -not (Test-Path -LiteralPath $state -PathType Leaf)) {
    throw "The persistent output bind mount did not preserve the job project"
}

$after = Get-Content -LiteralPath $state -Raw | ConvertFrom-Json
if ($after.job_id -ne $JobId) { throw "Recovered state has the wrong job identity" }
if ([int]$after.execution_generation -lt [int]$before.execution_generation) {
    throw "Execution generation moved backwards"
}

Write-Host "After restart: status=$($after.status) generation=$($after.execution_generation)"
Write-Host "Checkpoint project survived backend kill/recreation. Inspect logs with:"
Write-Host "docker compose -f $ComposeFile logs --since 5m backend"
