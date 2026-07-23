# Generate a standalone Cortex-A53 BSP/application tree directly with HSI.
# This avoids the deprecated XSCT-to-Vitis IDE bridge, which can time out on
# Windows 2025.1. Args: XSA OUTPUT_DIRECTORY

if {$argc != 2} {
    error "Usage: xsct build_mobilenet_baremetal.tcl XSA OUTPUT_DIRECTORY"
}
set xsa [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
if {![file exists $xsa]} { error "Missing XSA: $xsa" }
file mkdir $output_dir

hsi open_hw_design $xsa
hsi create_sw_design sigma_mobilenet_sw -proc psu_cortexa53_0 -os standalone
hsi generate_app -app empty_application -dir $output_dir
hsi close_sw_design [hsi current_sw_design]
hsi close_hw_design [hsi current_hw_design]

puts "SIGMA_MOBILENET_HSI_APP PASS output=$output_dir"
