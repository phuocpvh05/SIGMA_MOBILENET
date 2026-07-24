`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////

// Design: mult_switch.v
// Author: Eric Qin

// Description: Multiplier switch with local stationary buffer

/////////////////////////////////////////////////////////////////////////

module mult_switch(
	clk, 
	rst,
	i_valid, // input valid signal
	i_data, // input data
	i_stationary, // input control bit whether 
	o_valid, // output valid signal
	o_data // output data
);

	input clk;
	input rst;
	input i_valid;
	input [15:0] i_data;
	input i_stationary;

	output wire o_valid;
	output [31:0] o_data;

	reg [15:0] r_buffer; // buffer to hold stationary value
	reg r_buffer_valid; // valid buffer entry
	reg [7:0] r_valid_pipe;
	
	wire [15:0] w_A;
	wire [15:0] w_B;
	
	// The payload does not require reset: r_buffer_valid is the architectural
	// state that protects it. Keeping reset off 16 payload bits per PE reduces
	// control-set pressure while preserving stationary-buffer semantics.
	always @ (posedge clk) begin
		if (i_stationary == 1'b1 && i_valid == 1'b1)
			r_buffer <= i_data;
	end
	always @ (posedge clk) begin
		if (rst == 1'b1)
			r_buffer_valid <= 1'b0;
		else if (i_stationary == 1'b1 && i_valid == 1'b1)
			r_buffer_valid <= 1'b1;
	end
		
	assign w_A = (r_buffer_valid == 1'b1 && i_valid == 1'b1) ? i_data : 'd0; // streaming
	assign w_B = (r_buffer_valid == 1'b1 && i_valid == 1'b1) ? r_buffer : 'd0; // stationary
	
    // Match the eight registered stages in bf16_mult_pipeline. The multiplier
    // continues to accept one streaming operand every clock.
	always @ (posedge clk) begin
		if (rst)
			r_valid_pipe <= 0;
		else
			r_valid_pipe <= {r_valid_pipe[6:0],
			                 r_buffer_valid && i_valid};
	end
	assign o_valid = r_valid_pipe[7];

	// instantiate multiplier 
	multiplier my_multiplier (
		.clk(clk),
		.A(w_A), // stationary value
		.B(w_B), // streaming value
		.O(o_data[31:16])
	);
	
	// convert BF16 to FP32
	assign o_data[15:0] = 16'h0000; 

endmodule
