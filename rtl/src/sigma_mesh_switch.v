`timescale 1ns / 1ps
//==========================================================================
// Statically configured SIGMA mesh switch.
// Port order: 0=local, 1=north, 2=east, 3=south, 4=west.
// Each output independently selects one input (0..4) or 3'b111 to drop.
// Selecting one input on multiple outputs implements the paper's multicast.
// Registered outputs break combinational loops in the bidirectional mesh.
//==========================================================================

module sigma_mesh_switch #(
    parameter DATA_WIDTH = 32,
    parameter NUM_PORTS  = 5,
    parameter SEL_WIDTH  = 3
) (
    input  wire clk,
    input  wire rst,
    input  wire [NUM_PORTS*DATA_WIDTH-1:0] i_data,
    input  wire [NUM_PORTS-1:0] i_valid,
    input  wire [NUM_PORTS*SEL_WIDTH-1:0] i_route,
    output wire [NUM_PORTS*DATA_WIDTH-1:0] o_data,
    output wire [NUM_PORTS-1:0] o_valid
);
    genvar out_port;
    generate
        for (out_port = 0; out_port < NUM_PORTS; out_port = out_port + 1) begin : g_output
            reg [DATA_WIDTH-1:0] out_data_reg;
            reg out_valid_reg;
            reg [DATA_WIDTH-1:0] selected_data;
            reg selected_valid;
            wire [SEL_WIDTH-1:0] selected_input =
                i_route[out_port*SEL_WIDTH +: SEL_WIDTH];

            // Fixed source slices avoid a variable read from the complete
            // 5*DATA_WIDTH bus.  The functionality is still a programmable
            // five-input switch and selecting one source on several outputs
            // still performs SIGMA multicast.
            always @(*) begin
                selected_data = 0;
                selected_valid = 1'b0;
                case (selected_input)
                    3'd0: if (i_valid[0]) begin
                        selected_data = i_data[0*DATA_WIDTH +: DATA_WIDTH];
                        selected_valid = 1'b1;
                    end
                    3'd1: if (i_valid[1]) begin
                        selected_data = i_data[1*DATA_WIDTH +: DATA_WIDTH];
                        selected_valid = 1'b1;
                    end
                    3'd2: if (i_valid[2]) begin
                        selected_data = i_data[2*DATA_WIDTH +: DATA_WIDTH];
                        selected_valid = 1'b1;
                    end
                    3'd3: if (i_valid[3]) begin
                        selected_data = i_data[3*DATA_WIDTH +: DATA_WIDTH];
                        selected_valid = 1'b1;
                    end
                    3'd4: if (i_valid[4]) begin
                        selected_data = i_data[4*DATA_WIDTH +: DATA_WIDTH];
                        selected_valid = 1'b1;
                    end
                    default: begin end
                endcase
            end
            // Data has no architectural meaning while valid is low.  Holding
            // it removes selected_valid from the synchronous reset/data cone
            // of every 64-bit mesh register and leaves only the small valid
            // pipeline resettable.
            always @(posedge clk)
                if (selected_valid)
                    out_data_reg <= selected_data;
            always @(posedge clk) begin
                if (rst)
                    out_valid_reg <= 1'b0;
                else
                    out_valid_reg <= selected_valid;
            end
            assign o_data[out_port*DATA_WIDTH +: DATA_WIDTH] = out_data_reg;
            assign o_valid[out_port] = out_valid_reg;
        end
    endgenerate
endmodule
