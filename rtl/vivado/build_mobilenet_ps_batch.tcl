# Build the generated PS-AXI project and export an XSA.
# Usage:
#   vivado -mode batch -source build_mobilenet_ps_batch.tcl -tclargs create
#   vivado -mode batch -source build_mobilenet_ps_batch.tcl -tclargs synth
#   vivado -mode batch -source build_mobilenet_ps_batch.tcl -tclargs bit

set script_dir [file dirname [file normalize [info script]]]
set target [expr {$argc > 0 ? [string tolower [lindex $argv 0]] : "create"}]
if {$target ni {create synth bit}} { error "Target must be create, synth or bit" }

source [file join $script_dir create_mobilenet_ps_project.tcl]
if {$target eq "create"} {
    close_project
    exit 0
}

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%" ||
    ![string match "*Complete*" [get_property STATUS [get_runs synth_1]]]} {
    error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
open_run synth_1
file mkdir [file join $script_dir reports]
report_utilization -hierarchical -file [file join $script_dir reports utilization_synth.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $script_dir reports timing_synth.rpt]
close_design
if {$target eq "synth"} {
    close_project
    exit 0
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%" ||
    ![string match "*Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Implementation/bitstream failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
report_utilization -hierarchical -file [file join $script_dir reports utilization_impl.rpt]
report_timing_summary -delay_type min_max -max_paths 50 \
    -file [file join $script_dir reports timing_impl.rpt]
report_route_status -file [file join $script_dir reports route_status.rpt]
report_drc -file [file join $script_dir reports drc.rpt]

# A generated .bit file is not sufficient sign-off: Vivado can still write one
# when setup timing is negative.  Refuse to export the board XSA unless the
# actual routed design meets the 300 MHz constraint.
set worst_setup_path [get_timing_paths -setup -max_paths 1]
if {[llength $worst_setup_path] == 0} {
    error "No setup timing path found in the implemented design"
}
set setup_wns [get_property SLACK $worst_setup_path]
if {$setup_wns < 0.0} {
    error "300 MHz timing sign-off failed: WNS=${setup_wns} ns; see reports/timing_impl.rpt"
}
puts "SIGMA_MOBILENET_TIMING PASS WNS=${setup_wns}ns"
close_design

set xsa [file join $script_dir sigma_mobilenet_ps.xsa]
write_hw_platform -fixed -include_bit -force -file $xsa
puts "SIGMA_MOBILENET_PS_BUILD PASS xsa=$xsa"
close_project
