# Read-only timing diagnostic for the latest PS MobileNet routed checkpoint.
# This script never modifies a run and never launches synthesis/implementation.
set script_dir [file dirname [file normalize [info script]]]
set dcp [file normalize [file join $script_dir build sigma_mobilenet_ps.runs impl_1 \
    sigma_mobilenet_ps_bd_wrapper_postroute_physopt.dcp]]
if {![file exists $dcp]} {
    error "Missing routed checkpoint: $dcp"
}
open_checkpoint $dcp
set out [file normalize [file join $script_dir critical_paths_200.rpt]]
report_timing -delay_type max -max_paths 200 -nworst 10 \
    -path_type full_clock_expanded -file $out
puts "Wrote $out"
set summary_out [file normalize \
    [file join $script_dir critical_paths_all_summary.rpt]]
report_timing -delay_type max -slack_lesser_than 0.0 -max_paths 10000 \
    -nworst 1 -path_type summary -file $summary_out
puts "Wrote $summary_out"
close_design
