# Refresh the existing PS-AXI MobileNet project for the MOB7 board-proven
# datapath.  This script updates generated BD
# products and resets runs only; it never launches synthesis/implementation.

if {[llength [get_projects -quiet]] == 0} {
    error "Open sigma_mobilenet_ps.xpr before sourcing this script"
}
if {[get_property NAME [current_project]] ne "sigma_mobilenet_ps"} {
    error "Wrong project. Open project_1/ps_mobilenet/build/sigma_mobilenet_ps.xpr first"
}
catch {close_design}

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir .. ..]]
set cnn_dir [file join $root_dir cnn]
set board_repo [file normalize [file join $script_dir board_files]]

# board.repoPaths is a session parameter, not a persistent project property.
# Restore it every time before regenerating the BD; otherwise Vivado can unset
# the Genesys ZU board_part while opening an otherwise valid project.
set board_xml [file join $board_repo genesys-zu-5ev C.0 board.xml]
if {![file exists $board_xml]} {
    error "Missing Genesys ZU board definition: $board_xml"
}
set_param board.repoPaths [list $board_repo]
set board_parts [get_board_parts -quiet *gzu_5ev*]
if {[llength $board_parts] != 1} {
    error "Expected one Genesys ZU-5EV board part, found: $board_parts"
}
set_property board_part [lindex $board_parts 0] [current_project]

set guardband_xdc [file normalize \
    [file join $script_dir sigma_mobilenet_board_guardband.xdc]]
if {![file exists $guardband_xdc]} {
    error "Missing MobileNet board guardband constraint: $guardband_xdc"
}
if {[llength [get_files -quiet $guardband_xdc]] == 0} {
    add_files -fileset constrs_1 -norecurse $guardband_xdc
}
set guardband_file [get_files $guardband_xdc]
set_property IS_ENABLED true $guardband_file
set_property USED_IN_SYNTHESIS true $guardband_file
set_property USED_IN_IMPLEMENTATION true $guardband_file
set_property PROCESSING_ORDER LATE $guardband_file

set weight_path [file normalize \
    [file join $cnn_dir mobilenet_onchip_bf16.mem]]
if {![file exists $weight_path]} {
    error "Missing board-proven MobileNet weight image: $weight_path"
}
if {[llength [get_files -quiet $weight_path]] == 0} {
    add_files -fileset sources_1 -norecurse $weight_path
    puts "Added [file tail $weight_path]"
}
foreach obsolete_name {
    mobilenet_onchip_bf16_wide.mem
    mobilenet_onchip_bf16_bank0.mem
    mobilenet_onchip_bf16_bank1.mem
    mobilenet_onchip_bf16_bank2.mem
    mobilenet_onchip_bf16_bank3.mem
} {
    set obsolete_path [file normalize [file join $cnn_dir $obsolete_name]]
    set obsolete_file [get_files -quiet $obsolete_path]
    if {[llength $obsolete_file] != 0} {
        remove_files $obsolete_file
        puts "Removed obsolete project memory: $obsolete_name"
    }
}

# Verilog-only avoids the unnecessary >100 MB VHDL functional netlist that
# previously left the OOC synthesis run busy after its DCP was already ready.
set_property simulator_language Verilog [current_project]

# Reparse the edited RTL before refreshing the module-reference IP.  Vivado's
# update_module_reference command intentionally does not refresh source files
# by itself.
update_compile_order -fileset sources_1

set bd_files [get_files -quiet */sigma_mobilenet_ps_bd.bd]
if {[llength $bd_files] != 1} {
    error "Expected one sigma_mobilenet_ps_bd.bd, found: $bd_files"
}
set bd_file [lindex $bd_files 0]
open_bd_design $bd_file

# The ZU5EV PS quantizes the earlier 275 MHz request to 250 MHz.  Request the
# realizable value explicitly so the PS clock, AXI metadata and accelerator
# clock register agree.  The bit-exact datapath remains unchanged.
set ps_cell [get_bd_cells -quiet ps]
if {[llength $ps_cell] != 1} {
    error "Expected one Zynq UltraScale+ PS cell named ps"
}
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {250} $ps_cell

# Force the BD module-reference IP to reread the edited RTL instead of
# retaining the source snapshot used by the previous OOC checkpoint.
# update_module_reference expects an IP instance (get_ips), not a BD-cell
# object (get_bd_cells).
set accel_ip [get_ips -quiet sigma_mobilenet_ps_bd_sigma_mobilenet_0]
if {[llength $accel_ip] != 1} {
    error "Expected the sigma_mobilenet module-reference IP instance"
}
update_module_reference $accel_ip
validate_bd_design
save_bd_design
reset_target all $bd_file
generate_target all $bd_file
update_compile_order -fileset sources_1

# This release must be synthesized from the current MOB7 RTL.  An old project
# may still have automatic incremental synthesis enabled and point at a
# checkpoint such as adder32.dcp; reusing that file can silently restore a
# pre-MOB7 netlist.  Vivado 2025.1 exposes this property as numeric 1/0.
set synth_run [get_runs synth_1]
set_property AUTO_INCREMENTAL_CHECKPOINT 0 $synth_run
set_property INCREMENTAL_CHECKPOINT {} $synth_run
set stale_incremental_dcps [get_files -quiet \
    */utils_1/imports/synth_1/*.dcp]
if {[llength $stale_incremental_dcps] != 0} {
    remove_files $stale_incremental_dcps
    puts "Removed stale incremental checkpoints: $stale_incremental_dcps"
}

# Clear dependent results from downstream to upstream after regenerating the
# BD products.  No run is launched; all following stages remain under GUI
# control.
foreach run_name {impl_1 synth_1 sigma_mobilenet_ps_bd_sigma_mobilenet_0_synth_1} {
    set run [get_runs -quiet $run_name]
    if {[llength $run] != 0} {
        reset_run $run
    }
}

puts "============================================================"
puts "SIGMA MobileNet MOB7 rollback/update complete"
puts "ROM        : board-proven 245450 x 16-bit on-chip weight image"
puts "Datapath   : board-proven 64-PE SIGMA/depthwise/dot4 implementation"
puts "Clock      : 250 MHz (realizable PS clock; timing-safe target)"
puts "Profile    : MOB7 (software rejects MOB4/MOB5/MOB6 bitstreams)"
puts "Guardband  : 0.200 ns setup + 0.050 ns hold"
puts "Simulator  : Verilog-only"
puts "Incremental: disabled; synth_1 must rebuild MOB7 from RTL"
puts "Runs       : reset only; nothing was launched"
puts "Next       : GUI -> Design Runs -> Run Synthesis"
puts "============================================================"
