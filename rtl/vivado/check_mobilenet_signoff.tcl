# Run from the Vivado Tcl Console after MobileNet implementation completes.
# It only reads the implemented design and writes reports; it never launches a run.

set script_dir [file dirname [file normalize [info script]]]
set report_dir [file join $script_dir reports]
file mkdir $report_dir

if {[catch {current_design} design_name] || $design_name eq ""} {
    open_run impl_1
}

report_utilization -hierarchical -file [file join $report_dir utilization_impl.rpt]
report_timing_summary -delay_type min_max -max_paths 50 \
    -file [file join $report_dir timing_impl.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_drc -file [file join $report_dir drc.rpt]

set setup_path [get_timing_paths -setup -max_paths 1]
set hold_path [get_timing_paths -hold -max_paths 1]
if {[llength $setup_path] == 0} { error "No setup timing path found" }
if {[llength $hold_path] == 0} { error "No hold timing path found" }

set wns [get_property SLACK $setup_path]
set whs [get_property SLACK $hold_path]
set clock_name [get_property STARTPOINT_CLOCK $setup_path]
set clock_obj [get_clocks -quiet $clock_name]
if {[llength $clock_obj] == 0} {
    set clock_obj [lindex [get_clocks] 0]
}
set period [get_property PERIOD $clock_obj]
set fmax_mhz [expr {1000.0 / ($period - $wns)}]
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
set guardband_files [get_files -quiet */sigma_mobilenet_board_guardband.xdc]
set guardband_ok [expr {
    [llength $guardband_files] == 1 &&
    [get_property IS_ENABLED [lindex $guardband_files 0]]
}]
set guardband_status [expr {$guardband_ok ? "enabled" : "MISSING"}]
set rom_replication_ok 1
set rom_replication_counts [list]
foreach bank {0 1 2 3} {
    set bank_regs [get_cells -quiet -hier \
        *weight_phys_rd_addr_bank${bank}_q_reg*]
    lappend rom_replication_counts [llength $bank_regs]
    if {[llength $bank_regs] < 14} {
        set rom_replication_ok 0
    }
}
set rom_replication_status [expr {$rom_replication_ok ? "PASS" : "FAIL"}]

puts "============================================================"
puts "SIGMA MobileNet PS-AXI implementation sign-off"
puts [format "Clock period : %.3f ns" $period]
puts [format "Setup WNS    : %+.3f ns" $wns]
puts [format "Hold WHS     : %+.3f ns" $whs]
puts [format "Fmax estimate: %.3f MHz" $fmax_mhz]
puts "Setup guardband: 0.200 ns ($guardband_status)"
puts "ROM address replicas: $rom_replication_counts ($rom_replication_status)"
puts "DRC errors   : [llength $drc_errors]"
puts "Reports      : $report_dir"
if {$wns >= 0.0 && $whs >= 0.0 && [llength $drc_errors] == 0 &&
    $guardband_ok && $rom_replication_ok} {
    puts "SIGN-OFF      : PASS"
} else {
    puts "SIGN-OFF      : FAIL"
}
puts "============================================================"

# Keep the UART benchmark tied to this exact routed implementation.  This
# writes only a small C header/report; it never starts synthesis or routing.
source [file join $script_dir export_mobilenet_timing.tcl]
