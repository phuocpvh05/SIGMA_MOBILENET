`timescale 1ns / 1ps

// Fixed two-lane SIGMA activation packer.
module sparsity_packer #(
    parameter DATA_TYPE = 16
) (
    input  wire [2*DATA_TYPE-1:0] i_data,
    input  wire [1:0]             i_A_bitmap,
    input  wire [1:0]             i_REGOR,
    output reg  [2*DATA_TYPE-1:0] o_comp_data
);
    wire [1:0] eff = i_A_bitmap & i_REGOR;
    wire [DATA_TYPE-1:0] a0 = i_data[0*DATA_TYPE +: DATA_TYPE];
    wire [DATA_TYPE-1:0] a1 = i_data[1*DATA_TYPE +: DATA_TYPE];

    always @(*) begin
        o_comp_data = 0;
        case (eff)
            2'b01: o_comp_data[0*DATA_TYPE +: DATA_TYPE] = a0;
            2'b10: o_comp_data[0*DATA_TYPE +: DATA_TYPE] = a1;
            2'b11: begin
                o_comp_data[0*DATA_TYPE +: DATA_TYPE] = a0;
                o_comp_data[1*DATA_TYPE +: DATA_TYPE] = a1;
            end
            default: begin end
        endcase
    end
endmodule

// Fixed two-lane stationary compactor. Legal physical tiles are K=1,N<=2
// and K=2,N=1. An empty VN keeps a zero placeholder so the FAN emits zero.
module stationary_compactor #(
    parameter DATA_TYPE = 16
) (
    input  wire [2*DATA_TYPE-1:0] i_B_mat,
    input  wire [1:0]             i_B_bitmap,
    input  wire [7:0]             i_K,
    input  wire [7:0]             i_N,
    output reg  [2*DATA_TYPE-1:0] o_B_packed,
    output reg  [1:0]             o_vn_sep,
    output reg  [7:0]             o_packed_count
);
    wire [DATA_TYPE-1:0] b0 = i_B_mat[0*DATA_TYPE +: DATA_TYPE];
    wire [DATA_TYPE-1:0] b1 = i_B_mat[1*DATA_TYPE +: DATA_TYPE];

    always @(*) begin
        o_B_packed = 0;
        o_vn_sep = 2'b10;
        o_packed_count = 0;

        if (i_N != 0) begin
            if (i_K == 1) begin
                if (i_B_bitmap[0])
                    o_B_packed[0*DATA_TYPE +: DATA_TYPE] = b0;
                o_packed_count = 1;
                if (i_N > 1) begin
                    if (i_B_bitmap[1])
                        o_B_packed[1*DATA_TYPE +: DATA_TYPE] = b1;
                    o_packed_count = 2;
                end
            end else begin
                case (i_B_bitmap)
                    2'b00: o_packed_count = 1;
                    2'b01: begin
                        o_B_packed[0*DATA_TYPE +: DATA_TYPE] = b0;
                        o_packed_count = 1;
                    end
                    2'b10: begin
                        o_B_packed[0*DATA_TYPE +: DATA_TYPE] = b1;
                        o_packed_count = 1;
                    end
                    default: begin
                        o_B_packed[0*DATA_TYPE +: DATA_TYPE] = b0;
                        o_B_packed[1*DATA_TYPE +: DATA_TYPE] = b1;
                        o_vn_sep = 2'b00;
                        o_packed_count = 2;
                    end
                endcase
            end
        end
    end
endmodule
