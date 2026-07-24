`timescale 1ns/1ps
// Strict one-write/one-read synchronous RAM template for Vivado inference.
module sigma_sync_ram #(
    parameter WIDTH = 16,
    parameter DEPTH = 256,
    parameter ADDR_WIDTH = $clog2(DEPTH),
    parameter RAM_STYLE = "block"
) (
    input  wire clk,
    input  wire wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [WIDTH-1:0] wr_data,
    input  wire rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output reg  [WIDTH-1:0] rd_data
);
    (* ram_style = RAM_STYLE *) reg [WIDTH-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        if (rd_en)
            rd_data <= mem[rd_addr];
    end
endmodule
