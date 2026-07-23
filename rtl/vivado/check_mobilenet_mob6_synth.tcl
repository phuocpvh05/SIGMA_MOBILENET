# Read-only MOB6 structural check.  Run from the Vivado Tcl Console only after
# synth_1 completes; this script never launches or resets a run.
set synth_run [get_runs -quiet synth_1]
if {[llength $synth_run] != 1} {
    error "Cannot find top-level synth_1 in the open project"
}
set synth_progress [get_property PROGRESS $synth_run]
set synth_status [get_property STATUS $synth_run]
if {$synth_progress ne "100%" ||
    [string match -nocase *out-of-date* $synth_status] ||
    [string match -nocase *not started* $synth_status]} {
    error "Top-level synth_1 is not complete: STATUS='$synth_status', PROGRESS='$synth_progress'. Run Synthesis in the GUI and wait for synth_1 itself, not only its OOC child."
}
if {[catch {current_design} design_name] || $design_name eq "" ||
    [get_property DESIGN_MODE [current_design]] ne "GateLvl"} {
    open_run synth_1
}

set all_ok 1
puts "============================================================"
puts "SIGMA MobileNet MOB6 synthesized bank-local ROM addresses"
foreach bank {0 1 2 3} {
    set bank_regs [get_cells -quiet -hier \
        *weight_phys_rd_addr_bank${bank}_q_reg*]
    set count [llength $bank_regs]
    # WEIGHT_PHYSICAL_AW is 14 bits.  Physical synthesis may legally merge or
    # replicate additional copies, so require the complete named bank-local
    # address vector here and leave placement quality to the guarded timing
    # sign-off after implementation.
    puts "bank${bank}: $count address registers (expected at least 14)"
    if {$count < 14} {
        set all_ok 0
    }
}
if {!$all_ok} {
    error "MOB6 bank-local ROM address registers are missing; do not run implementation"
}
puts "MOB6 STRUCTURE: PASS"
puts "Next: GUI -> Run Implementation"
puts "============================================================"
