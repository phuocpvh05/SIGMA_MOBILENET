`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////

// Design: flexdpe.v
// Author: Eric Qin

// Description: SIGMA Macro PE (FLEX-DPE) top level design

/////////////////////////////////////////////////////////////////////////

module flexdpe(
	clk,
	rst,
	i_data_valid, // input data bus valid
	i_data_bus, // input data bus
	i_stationary, // control bit signaling input data is stored in stationary buffer
	i_dest_bus, // dest bus for xbar network (legacy)
	i_benes_bus, // dest bus for benes network
	i_vn_seperator, // alternate virtual neuron seperator

	o_data_valid, // valid data signals
	o_data_bus, // output data bus
	o_vn_bus // VN tag associated with every physical FAN output
);

	parameter IN_DATA_TYPE = 16; // input data type width (BFP16)
	parameter OUT_DATA_TYPE = 32; // output data type width (FP32)
	parameter NUM_PES = 2; // Per-DPE SIMD width; each network selects its profile.
	parameter LOG2_PES = 1;
	parameter FAN_ADD_PIPE_STAGES = 6;

	localparam LEVELS = 2 * LOG2_PES + 1;
	localparam BENES_CTRL_WIDTH = 2 * (LEVELS - 2) * NUM_PES + NUM_PES;

	input clk;
	input rst;
	input i_data_valid;
	input [NUM_PES * IN_DATA_TYPE -1 : 0] i_data_bus;
	input i_stationary;
	input [NUM_PES * LOG2_PES -1:0] i_dest_bus;
	input [BENES_CTRL_WIDTH -1:0] i_benes_bus;
	input [NUM_PES * LOG2_PES -1:0] i_vn_seperator;

	output [NUM_PES-1:0] o_data_valid;
	output [NUM_PES * OUT_DATA_TYPE -1:0] o_data_bus;
	output [NUM_PES * LOG2_PES -1:0] o_vn_bus;

	wire [NUM_PES * OUT_DATA_TYPE -1: 0] r_mult;

	wire [NUM_PES * IN_DATA_TYPE -1 : 0]  w_dist_bus; // output of xbar network
	wire w_mult_valid;

	reg [NUM_PES * IN_DATA_TYPE -1 : 0] r_data_bus_ff, r_data_bus_ff2;
	reg r_data_valid_ff, r_data_valid_ff2;
	reg r_stationary_ff, r_stationary_ff2;
	reg [BENES_CTRL_WIDTH -1:0] r_benes_bus_ff, r_benes_bus_ff2;

	// adjust some input signal delays from xbar and controller
	always @ (posedge clk) begin
		r_data_bus_ff <= i_data_bus;
		r_data_bus_ff2 <= r_data_bus_ff;
		r_data_valid_ff <= i_data_valid; 
		r_data_valid_ff2 <= r_data_valid_ff;
		r_stationary_ff <= i_stationary;
		r_stationary_ff2 <= r_stationary_ff;
		r_benes_bus_ff <= i_benes_bus;
		r_benes_bus_ff2 <= r_benes_bus_ff;
	end

	// instantize distribution network (benes)
	benes #(
		.DATA_TYPE(IN_DATA_TYPE),
		.NUM_PES(NUM_PES),
		.LEVELS(LEVELS))
		my_benes (
		.clk(clk),
		.rst(rst),
		.i_data_bus(r_data_bus_ff2),
		.i_mux_bus(r_benes_bus_ff2),
		.o_dist_bus(w_dist_bus)
	);

	// generate multiplier chain (output of xbar to input of multiplier chain)
	mult_gen #(
		.IN_DATA_TYPE(IN_DATA_TYPE),
		.OUT_DATA_TYPE(OUT_DATA_TYPE),
		.NUM_PES(NUM_PES))
		my_mult_gen (
		.clk(clk),
		.rst(rst),
		.i_valid(r_data_valid_ff2),
		.i_data_bus(w_dist_bus),
		.i_stationary(r_stationary_ff2),
		.o_valid(w_mult_valid),
		.o_data_bus(r_mult)
	);

	// instantiate fan reduction topology
	fan_network #(
		.DATA_TYPE(OUT_DATA_TYPE),
		.NUM_PES(NUM_PES),
		.LOG2_PES(LOG2_PES),
		.ADD_PIPE_STAGES(FAN_ADD_PIPE_STAGES))
		my_fan_network(
		.clk(clk),
		.rst(rst),
		.i_valid(w_mult_valid),
		.i_data_bus(r_mult),
		.i_vn_bus(i_vn_seperator),
		.o_valid(o_data_valid),
		.o_data_bus(o_data_bus),
		.o_vn_bus(o_vn_bus)
	);

endmodule
