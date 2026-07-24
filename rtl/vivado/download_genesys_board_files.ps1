param([switch]$Force)

$ErrorActionPreference = "Stop"
$destination = Join-Path $PSScriptRoot "board_files\genesys-zu-5ev\C.0"
$baseUrl = "https://raw.githubusercontent.com/Digilent/vivado-boards/master/new/board_files/genesys-zu-5ev/C.0"
$required = @("board.xml", "preset.xml", "part0_pins.xml", "changelog.txt")

New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($name in $required) {
    $target = Join-Path $destination $name
    if ($Force -or -not (Test-Path -LiteralPath $target)) {
        Write-Host "Downloading Digilent Genesys ZU-5EV board file: $name"
        Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/$name" -OutFile $target
    }
    if ((Get-Item -LiteralPath $target).Length -eq 0) {
        throw "Downloaded board file is empty: $target"
    }
}

$boardText = Get-Content -LiteralPath (Join-Path $destination "board.xml") -Raw
if ($boardText -notmatch 'name="gzu_5ev"' -or
    $boardText -notmatch 'part_name="xczu5ev-sfvc784-1-e"') {
    throw "The downloaded board.xml is not the Digilent Genesys ZU-5EV definition"
}

$manifest = foreach ($name in $required) {
    $file = Get-Item -LiteralPath (Join-Path $destination $name)
    $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
    [pscustomobject]@{
        file = $name
        bytes = $file.Length
        sha256 = $hash.Hash.ToLowerInvariant()
        source = "$baseUrl/$name"
    }
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destination "download_manifest.json") -Encoding utf8

Write-Host "Genesys ZU-5EV board files ready: $destination" -ForegroundColor Green

