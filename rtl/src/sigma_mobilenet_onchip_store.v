`timescale 1ns / 1ps
// Local model/activation store for MobileNetV2-0.25.  The 490,900-byte BF16
// model image is initialized into Block RAM with the bitstream.  Three small
// activation banks use UltraRAM: they are runtime-written (no initialization
// requirement), remain fully on-chip, and avoid both BRAM exhaustion and a
// very large distributed-RAM address fanout on the 5EV.
module sigma_mobilenet_onchip_store #(
    parameter WEIGHT_WORDS = 245450,
    parameter WEIGHT_AW = 18,
    parameter BANK_WORDS = 12288,
    parameter BANK_AW = 14,
    parameter WEIGHT_INIT_FILE = "mobilenet_onchip_bf16.mem"
) (
    input  wire clk,
    input  wire i_weight_rd_en,
    input  wire [WEIGHT_AW-1:0] i_weight_rd_addr,
    output reg  [15:0] o_weight_rd_data,

    input  wire i_activation_wr_en,
    input  wire [1:0] i_activation_wr_bank,
    input  wire [BANK_AW-1:0] i_activation_wr_addr,
    input  wire [15:0] i_activation_wr_data,

    input  wire i_activation_rd_en,
    input  wire [1:0] i_activation_rd_bank,
    input  wire [BANK_AW-1:0] i_activation_rd_addr,
    output wire [15:0] o_activation_rd_data,

    input  wire i_skip_rd_en,
    input  wire [1:0] i_skip_rd_bank,
    input  wire [BANK_AW-1:0] i_skip_rd_addr,
    output wire [15:0] o_skip_rd_data
);
    (* ram_style = "block" *) reg [15:0] weight_mem [0:WEIGHT_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation0 [0:BANK_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation1 [0:BANK_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation2 [0:BANK_WORDS-1];

    reg [15:0] bank0_q, bank1_q, bank2_q;
    wire bank0_primary = i_activation_rd_en && (i_activation_rd_bank == 0);
    wire bank1_primary = i_activation_rd_en && (i_activation_rd_bank == 1);
    wire bank2_primary = i_activation_rd_en && (i_activation_rd_bank == 2);
    wire bank0_skip = i_skip_rd_en && (i_skip_rd_bank == 0);
    wire bank1_skip = i_skip_rd_en && (i_skip_rd_bank == 1);
    wire bank2_skip = i_skip_rd_en && (i_skip_rd_bank == 2);
    wire [BANK_AW-1:0] bank0_rd_addr = bank0_primary ? i_activation_rd_addr : i_skip_rd_addr;
    wire [BANK_AW-1:0] bank1_rd_addr = bank1_primary ? i_activation_rd_addr : i_skip_rd_addr;
    wire [BANK_AW-1:0] bank2_rd_addr = bank2_primary ? i_activation_rd_addr : i_skip_rd_addr;
    assign o_activation_rd_data = (i_activation_rd_bank == 0) ? bank0_q :
                                  (i_activation_rd_bank == 1) ? bank1_q : bank2_q;
    assign o_skip_rd_data = (i_skip_rd_bank == 0) ? bank0_q :
                            (i_skip_rd_bank == 1) ? bank1_q : bank2_q;

    initial begin
        if (WEIGHT_INIT_FILE != "")
            $readmemh(WEIGHT_INIT_FILE, weight_mem);
    end

    always @(posedge clk) begin
        if (i_weight_rd_en)
            o_weight_rd_data <= weight_mem[i_weight_rd_addr];

        if (i_activation_wr_en) begin
            case (i_activation_wr_bank)
                0: activation0[i_activation_wr_addr] <= i_activation_wr_data;
                1: activation1[i_activation_wr_addr] <= i_activation_wr_data;
                2: activation2[i_activation_wr_addr] <= i_activation_wr_data;
                default: ;
            endcase
        end

        if (bank0_primary || bank0_skip)
            bank0_q <= activation0[bank0_rd_addr];
        if (bank1_primary || bank1_skip)
            bank1_q <= activation1[bank1_rd_addr];
        if (bank2_primary || bank2_skip)
            bank2_q <= activation2[bank2_rd_addr];

    end
endmodule
