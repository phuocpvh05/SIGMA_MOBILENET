# Export the routed MobileNet clock result into the bare-metal benchmark.
# Run manually from Vivado's Tcl Console after Implementation completes:
#   source <release>/rtl/vivado/export_mobilenet_timing.tcl
# Optional override when the design contains several clocks:
#   set SIGMA_TIMING_CLOCK clk_pl_0

set script_dir [file dirname [file normalize [info script]]]
set header_path [file normalize [file join $script_dir .. .. software a53 baremetal mobilenet_timing_generated.h]]
set report_dir [file normalize [file join $script_dir reports]]
file mkdir $report_dir

if {[catch {current_design} current_design_name] || $current_design_name eq ""} {
    if {[llength [get_runs -quiet impl_1]] == 0 ||
        [get_property PROGRESS [get_runs impl_1]] ne "100%"} {
        error "Implementation is not complete. Run Implementation in the GUI first."
    }
    open_run impl_1
}

if {[info exists SIGMA_TIMING_CLOCK]} {
    set sigma_clock [get_clocks -quiet $SIGMA_TIMING_CLOCK]
    if {[llength $sigma_clock] != 1} {
        error "SIGMA_TIMING_CLOCK '$SIGMA_TIMING_CLOCK' did not select exactly one clock."
    }
} else {
    set sigma_clock [get_clocks -quiet *clk_pl_0*]
    if {[llength $sigma_clock] != 1} {
        set sigma_clock [get_clocks -quiet *pl_clk0*]
    }
    if {[llength $sigma_clock] != 1} {
        set all_clocks [get_clocks -quiet]
        if {[llength $all_clocks] == 1} {
            set sigma_clock $all_clocks
        } else {
            error "Cannot identify the accelerator clock. Available clocks: $all_clocks. Set SIGMA_TIMING_CLOCK and source this script again."
        }
    }
}

set sigma_clock [lindex $sigma_clock 0]
set period_ns [get_property PERIOD $sigma_clock]
set worst_path [get_timing_paths -quiet -delay_type max -max_paths 1 \
    -from $sigma_clock -to $sigma_clock]
if {[llength $worst_path] == 0} {
    error "No register-to-register setup path was found for clock $sigma_clock."
}
set wns_ns [get_property SLACK [lindex $worst_path 0]]
set critical_delay_ns [expr {$period_ns - $wns_ns}]
if {$critical_delay_ns <= 0.0} {
    error "Invalid critical path delay: $critical_delay_ns ns"
}
set fmax_mhz [expr {1000.0 / $critical_delay_ns}]
set fmax_khz [expr {round($fmax_mhz * 1000.0)}]
set wns_ps [expr {round($wns_ns * 1000.0)}]
set period_ps [expr {round($period_ns * 1000.0)}]

set out [open $header_path w]
puts $out "#ifndef MOBILENET_TIMING_GENERATED_H"
puts $out "#define MOBILENET_TIMING_GENERATED_H"
puts $out ""
puts $out "/* Generated from the routed Vivado design. */"
puts $out "#define SIGMA_POST_ROUTE_VALID 1"
puts $out "#define SIGMA_POST_ROUTE_FMAX_KHZ ${fmax_khz}u"
puts $out "#define SIGMA_POST_ROUTE_WNS_PS $wns_ps"
puts $out "#define SIGMA_POST_ROUTE_PERIOD_PS ${period_ps}u"
puts $out ""
puts $out "#endif"
close $out

set text_report [file join $report_dir mobilenet_post_route_timing.txt]
set report [open $text_report w]
puts $report "clock=$sigma_clock"
puts $report [format "target_period_ns=%.6f" $period_ns]
puts $report [format "wns_ns=%.6f" $wns_ns]
puts $report [format "critical_delay_ns=%.6f" $critical_delay_ns]
puts $report [format "estimated_fmax_mhz=%.3f" $fmax_mhz]
close $report

puts "============================================================"
puts "SIGMA MobileNet post-route timing exported"
puts "Clock       : $sigma_clock"
puts [format "Target      : %.3f MHz (%.6f ns)" [expr {1000.0/$period_ns}] $period_ns]
puts [format "WNS         : %.6f ns" $wns_ns]
puts [format "Fmax limit  : %.3f MHz" $fmax_mhz]
puts "C header    : $header_path"
puts "Text report : $text_report"
puts "Rebuild the Vitis application so its UART log includes this result."
puts "============================================================"
