$ErrorActionPreference = "Stop"

$softwareDir = $PSScriptRoot
$releaseRoot = (Resolve-Path -LiteralPath (Join-Path $softwareDir "..\..")).Path
$xsa = Join-Path $releaseRoot "rtl\vivado\sigma_mobilenet_ps.xsa"
if (-not (Test-Path -LiteralPath $xsa)) {
    throw "Missing XSA. Export the implemented Vivado design to: $xsa"
}

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "Python was not found in PATH" }
& $python (Join-Path $softwareDir "generate_mobilenet_c_artifacts.py") --embed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$buildDir = Join-Path $softwareDir "baremetal_build"
if (Test-Path -LiteralPath $buildDir) {
    $resolvedWorkspace = (Resolve-Path -LiteralPath $buildDir).Path
    if (-not $resolvedWorkspace.StartsWith($softwareDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove workspace outside software directory: $resolvedWorkspace"
    }
    Remove-Item -LiteralPath $resolvedWorkspace -Recurse -Force
}

$xsctCommand = Get-Command xsct.bat -ErrorAction SilentlyContinue
$xsct = if ($xsctCommand) { $xsctCommand.Source } else { "D:\Xilinx\2025.1\Vitis\bin\xsct.bat" }
if (-not (Test-Path -LiteralPath $xsct)) { throw "XSCT not found. Add Vitis bin to PATH: $xsct" }
$argsList = @((Join-Path $softwareDir "build_mobilenet_baremetal.tcl"), $xsa, $buildDir)
& $xsct @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($source in @(
    (Join-Path $softwareDir "baremetal\sigma_mobilenet_baremetal.c"),
    (Join-Path $softwareDir "baremetal\mobilenet_mnist10k_data.S"),
    (Join-Path $softwareDir "baremetal\mobilenet_timing_generated.h"),
    (Join-Path $softwareDir "sigma_mobilenet_reference.c"),
    (Join-Path $softwareDir "sigma_mobilenet_reference.h"),
    (Join-Path $softwareDir "generated\mobilenet_layers_generated.h"),
    (Join-Path $softwareDir "generated\mobilenet_payload_generated.h")
)) {
    Copy-Item -LiteralPath $source -Destination $buildDir -Force
}

$toolBin = "D:\Xilinx\2025.1\Vitis\gnu\aarch64\nt\aarch64-none\bin"
$makeBin = "D:\Xilinx\2025.1\Vitis\gnuwin\bin"
$make = Join-Path $makeBin "make.exe"
if (-not (Test-Path -LiteralPath $make)) { throw "GNU make not found: $make" }
$env:Path = "$toolBin;$makeBin;$env:Path"
$env:RDI_PLATFORM = "win64"
Push-Location $buildDir
try {
    & $make 'BSP_FLAGS=-O3 -mcpu=cortex-a53 -c' 'CFLAGS=-O3 -mcpu=cortex-a53 -Wall -Wextra'
    $makeExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($makeExitCode -ne 0) { exit $makeExitCode }

$elf = Join-Path $buildDir "executable.elf"
if (-not (Test-Path -LiteralPath $elf)) { throw "Bare-metal ELF was not generated: $elf" }
Copy-Item -LiteralPath $elf -Destination (Join-Path $softwareDir "sigma_mobilenet_baremetal.elf") -Force
Remove-Item -LiteralPath $buildDir -Recurse -Force
Write-Host "Bare-metal A53 benchmark ready:" -ForegroundColor Green
Write-Host (Join-Path $softwareDir "sigma_mobilenet_baremetal.elf")
