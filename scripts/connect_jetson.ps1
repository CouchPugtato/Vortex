param(
    [Parameter(Mandatory = $true)]
    [string]$Alias,

    [string]$Host = "192.168.55.1",

    [string]$User = "vortex",

    [switch]$ReplaceKey
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Require-Command -Name "ssh"
Require-Command -Name "ssh-keygen"

$sshDir = Join-Path $env:USERPROFILE ".ssh"
$knownHosts = Join-Path $sshDir "known_hosts"

if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
}

if (-not (Test-Path $knownHosts)) {
    New-Item -ItemType File -Path $knownHosts -Force | Out-Null
}

if ($ReplaceKey) {
    Write-Host "Removing stored host keys for alias '$Alias' and host '$Host'..."
    & ssh-keygen -R $Alias -f $knownHosts | Out-Host
    & ssh-keygen -R $Host -f $knownHosts | Out-Host
}

$sshArgs = @(
    "-o", "HostKeyAlias=$Alias",
    "-o", "UserKnownHostsFile=$knownHosts",
    "-o", "StrictHostKeyChecking=accept-new",
    "$User@$Host"
)

Write-Host "Connecting to $User@$Host using host key alias '$Alias'..."
& ssh @sshArgs
