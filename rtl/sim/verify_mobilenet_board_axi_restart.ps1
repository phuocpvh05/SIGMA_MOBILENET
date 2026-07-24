param([switch]$Rebuild)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$vmod = Join-Path $root "project_1\project_1.srcs\sources_1\imports\vmod"
$tb = Join-Path $root "tb\tb_mobilenet_board_axi_restart.sv"
$build = Join-Path $PSScriptRoot "xsim_mobilenet_board_axi_build"
$snapshot = "sigma_mobilenet_board_axi_restart"
$stamp = Join-Path $build "snapshot.ready"
$sources = Get-ChildItem -LiteralPath $vmod -Filter "*.v" | ForEach-Object FullName
$inputs = @($sources) + @($tb, $PSCommandPath)
$fresh = Test-Path $stamp
if ($fresh) {
    $stampTime = (Get-Item $stamp).LastWriteTimeUtc
    $fresh = -not ($inputs | Where-Object { (Get-Item $_).LastWriteTimeUtc -gt $stampTime })
}
New-Item -ItemType Directory -Path $build -Force | Out-Null
$name = "mobilenet_onchip_bf16.mem"
Copy-Item (Join-Path $PSScriptRoot $name) (Join-Path $build $name) -Force
if ($Rebuild -or -not $fresh) {
    $project = Join-Path $build "files.prj"
    $lines = $sources | ForEach-Object { 'verilog xil_defaultlib "' + ($_ -replace '\\','/') + '"' }
    $lines += 'sv xil_defaultlib "' + ($tb -replace '\\','/') + '"'
    Set-Content $project $lines -Encoding ascii
    Push-Location $build
    try {
        & (Get-Command xvlog.bat).Source -d SYNTHESIS --relax -i $vmod -prj $project -log compile.log
        if ($LASTEXITCODE) { throw "xvlog failed: $LASTEXITCODE" }
        & (Get-Command xelab.bat).Source --debug off --relax --mt 2 -L xil_defaultlib `
            -L unisims_ver -L unimacro_ver -L secureip -L xpm --snapshot $snapshot `
            xil_defaultlib.tb_mobilenet_board_axi_restart -log elaborate.log
        if ($LASTEXITCODE) { throw "xelab failed: $LASTEXITCODE" }
        Set-Content $stamp ready -Encoding ascii
    } finally { Pop-Location }
}
Set-Content (Join-Path $build "run.tcl") @("run all", "quit") -Encoding ascii
$image0 = Join-Path $PSScriptRoot "onchip_jobs\mnist_0\image_bf16.mem"
$image1 = Join-Path $PSScriptRoot "onchip_jobs\mnist_1\image_bf16.mem"
$argsFile = Join-Path $build "xsim.args"
Set-Content $argsFile @(
    $snapshot,
    "-tclbatch run.tcl",
    "-testplusarg IMAGE0=$($image0 -replace '\\','/')",
    "-testplusarg IMAGE1=$($image1 -replace '\\','/')",
    "-log $((Join-Path $build 'run.log') -replace '\\','/')"
) -Encoding ascii
Push-Location $build
try {
    & (Get-Command xsim.bat).Source -f $argsFile
    if ($LASTEXITCODE) { throw "xsim failed: $LASTEXITCODE" }
} finally { Pop-Location }
$log = Get-Content (Join-Path $build "run.log") -Raw
if ($log -notmatch "MOBILENET_AXI_RESTART PASSED") { throw "AXI restart regression did not pass" }
Write-Host "MobileNet AXI restart regression PASS: label 7 -> label 2, two real inferences"
