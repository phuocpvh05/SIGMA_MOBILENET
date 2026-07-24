# Read-only MOB7 synthesis check. Run this from the Vivado Tcl Console after
# top-level synth_1 completes. It never launches or resets a run.
set synth_run [get_runs -quiet synth_1]
if {[llength $synth_run] != 1} {
    error "Cannot find top-level synth_1 in the open project"
}
set synth_progress [get_property PROGRESS $synth_run]
set synth_status [get_property STATUS $synth_run]
if {$synth_progress ne "100%" ||
    [string match -nocase *out-of-date* $synth_status] ||
    [string match -nocase *not\ started* $synth_status]} {
    error "Top-level synth_1 is not complete: STATUS='$synth_status', PROGRESS='$synth_progress'"
}
if {[catch {current_design} design_name] || $design_name eq "" ||
    [get_property DESIGN_MODE [current_design]] ne "GateLvl"} {
    open_run synth_1
}

set accel_cells [get_cells -quiet -hier *u_accelerator*]
set store_cells [get_cells -quiet -hier *u_store*]
set fast_super_cells [get_cells -quiet -hier *weight_super*]
if {[llength $accel_cells] == 0 || [llength $store_cells] == 0} {
    error "MOB7 accelerator/store hierarchy is missing from synth_1"
}
if {[llength $fast_super_cells] != 0} {
    error "Obsolete MOB4/MOB5/MOB6 256-bit fast-engine logic is still present"
}

puts "============================================================"
puts "SIGMA MobileNet MOB7 SYNTHESIS: PASS"
puts "Datapath : board-proven 64-PE implementation"
puts "Clock    : 250 MHz target"
puts "Old x5   : no weight_super hierarchy found"
puts "Next     : GUI -> Run Implementation"
puts "============================================================"
