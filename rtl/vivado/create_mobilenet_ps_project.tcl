# Create a self-contained Genesys ZU-5EV PS-AXI MobileNet project.
# This deliberately uses a separate Vivado project so the proven standalone
# JTAG checkpoint in ../project_1.xpr is never reset or overwritten.

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir .. ..]]
set build_dir [file join $script_dir build]
set board_repo [file join $script_dir board_files]
set vmod_dir [file join $root_dir rtl src]
set cnn_dir [file join $root_dir rtl model]
set project_name sigma_mobilenet_ps
set design_name sigma_mobilenet_ps_bd

foreach required [list \
    [file join $board_repo genesys-zu-5ev C.0 board.xml] \
    [file join $cnn_dir mobilenet_onchip_bf16_wide.mem] \
    [file join $cnn_dir mobilenet_onchip_bf16_bank0.mem] \
    [file join $cnn_dir mobilenet_onchip_bf16_bank1.mem] \
    [file join $cnn_dir mobilenet_onchip_bf16_bank2.mem] \
    [file join $cnn_dir mobilenet_onchip_bf16_bank3.mem] \
    [file join $script_dir sigma_mobilenet_board_guardband.xdc] \
    [file join $vmod_dir sigma_mobilenet_ps_axi.v]] {
    if {![file exists $required]} { error "Missing required file: $required" }
}

set_param board.repoPaths [list $board_repo]
set board_parts [get_board_parts -quiet *gzu_5ev*]
if {[llength $board_parts] != 1} {
    error "Expected one Genesys ZU-5EV board part in $board_repo, found: $board_parts"
}
set board_part [lindex $board_parts 0]

file mkdir $build_dir
create_project -force $project_name $build_dir -part xczu5ev-sfvc784-1-e
set_property board_part $board_part [current_project]
set_property target_language Verilog [current_project]
# The SIGMA hierarchy is Verilog-only.  Mixed mode makes every OOC synthesis
# export an unnecessary functional VHDL netlist; the 256-bit initialized model
# ROM expands that file beyond 100 MB and can keep Vivado busy long after the
# synthesis checkpoint is already complete.
set_property simulator_language Verilog [current_project]

set rtl_sources [list \
    benes.v \
    bfp16_mult.v \
    bfp32_adder.v \
    fan_network.v \
    flexdpe.v \
    flexdpu.v \
    mult_gen.v \
    mult_switch.v \
    sparsity_controller.v \
    sigma_mesh_switch.v \
    sigma_mesh_noc.v \
    sigma_sync_ram.v \
    sigma_top.v \
    sigma_fold_core.v \
    sigma_mobilenet_onchip_store.v \
    sigma_mobilenet_dot4.v \
    sigma_mobilenet_depthwise4.v \
    sigma_mobilenet_onchip_top.v \
    sigma_mobilenet_board_axi.v \
    sigma_mobilenet_ps_axi.v]
foreach name $rtl_sources {
    set path [file join $vmod_dir $name]
    if {![file exists $path]} { error "Missing MobileNet RTL: $path" }
    add_files -fileset sources_1 -norecurse $path
}
add_files -fileset sources_1 -norecurse [file join $vmod_dir sigma_mobilenet_layers.vh]
set_property FILE_TYPE {Verilog Header} \
    [get_files [file join $vmod_dir sigma_mobilenet_layers.vh]]
add_files -fileset sources_1 -norecurse [file join $cnn_dir mobilenet_onchip_bf16_wide.mem]
foreach bank {0 1 2 3} {
    add_files -fileset sources_1 -norecurse \
        [file join $cnn_dir mobilenet_onchip_bf16_bank${bank}.mem]
}
set_property include_dirs [list $vmod_dir] [get_filesets sources_1]
set guardband_xdc [file normalize \
    [file join $script_dir sigma_mobilenet_board_guardband.xdc]]
add_files -fileset constrs_1 -norecurse $guardband_xdc
set_property PROCESSING_ORDER LATE [get_files $guardband_xdc]
update_compile_order -fileset sources_1

create_bd_design $design_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps

# Keep the official Digilent DDR/MIO preset, then expose one 32-bit PS master
# and generate the accelerator clock from the PS.  The entire AXI/register/core
# path is synchronous at 300 MHz, avoiding a clock-converter latency penalty.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} \
    CONFIG.PSU__USE__S_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP3 {0} \
    CONFIG.PSU__USE__S_AXI_GP4 {0} \
    CONFIG.PSU__USE__S_AXI_GP5 {0} \
    CONFIG.PSU__USE__S_AXI_GP6 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {300}] $ps

set accel [create_bd_cell -type module -reference sigma_mobilenet_ps_axi sigma_mobilenet]
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc

set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_pl300]
set reset_inverter [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* rst_n_to_high]
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $reset_inverter
set locked [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* pl_clock_locked]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] $locked

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins sigma_mobilenet/S_AXI]

connect_bd_net [get_bd_pins ps/pl_clk0] \
    [get_bd_pins ps/maxihpm0_fpd_aclk] \
    [get_bd_pins axi_smc/aclk] \
    [get_bd_pins sigma_mobilenet/s_axi_aclk] \
    [get_bd_pins rst_pl300/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_n_to_high/Op1]
connect_bd_net [get_bd_pins rst_n_to_high/Res] [get_bd_pins rst_pl300/ext_reset_in]
connect_bd_net [get_bd_pins pl_clock_locked/dout] [get_bd_pins rst_pl300/dcm_locked]
connect_bd_net [get_bd_pins rst_pl300/peripheral_aresetn] \
    [get_bd_pins axi_smc/aresetn] \
    [get_bd_pins sigma_mobilenet/s_axi_aresetn]

assign_bd_address
set sigma_segments [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces ps/Data] \
    -filter {NAME =~ *sigma_mobilenet*}]
if {[llength $sigma_segments] != 1} {
    error "Expected one mapped SIGMA address segment, found: $sigma_segments; all segments: [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]]"
}
set_property offset 0xA0000000 $sigma_segments
set_property range 64K $sigma_segments

validate_bd_design
save_bd_design

# The teaching/GUI entry point intentionally stops here.  Everything below is
# output-product generation or build preparation and remains available only to
# the optional batch flow.
if {[info exists SIGMA_GUI_ONLY] && $SIGMA_GUI_ONLY} {
    puts "SIGMA_MOBILENET_BD_ONLY PASS design=$design_name"
} else {
    generate_target all [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]]
    set wrapper [make_wrapper -files [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]] -top]
    add_files -norecurse $wrapper
    set_property top ${design_name}_wrapper [get_filesets sources_1]
    update_compile_order -fileset sources_1

    # Export a hardware handoff immediately so the A53 software can be compiled
    # before the long implementation run.  The bit target overwrites this XSA
    # with an include-bitstream version after timing/DRC sign-off.
    write_hw_platform -fixed -force -file [file join $script_dir sigma_mobilenet_ps.xsa]

    set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

    puts "============================================================"
    puts "SIGMA MobileNet PS-AXI project created"
    puts "Project    : [file join $build_dir ${project_name}.xpr]"
    puts "Board      : $board_part"
    puts "PS master  : M_AXI_HPM0_FPD, 32-bit AXI4-Lite"
    puts "Address    : 0xA0000000..0xA000FFFF"
    puts "PL clock   : 300 MHz"
    puts "Guardband  : 0.200 ns setup + 0.050 ns hold required at 300 MHz"
    puts "On-chip    : all weights and intermediate activations"
    puts "XSA         : [file join $script_dir sigma_mobilenet_ps.xsa]"
    puts "============================================================"
}
