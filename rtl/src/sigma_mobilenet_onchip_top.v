`timescale 1ns / 1ps
// Autonomous MobileNetV2-0.25 controller for the SIGMA 4x4 performance core.
//
// Runtime traffic is only a 28x28 BF16 image plus start/status.  All 53 learned
// Conv/Depthwise/Pointwise/Linear layers, BatchNorm-folded weights, residual
// adds, ReLU6 and the final argmax execute in programmable logic.  Activations
// use HWC layout in three local banks; no DDR/AXI payload is used by this top.
module sigma_mobilenet_onchip_top #(
    parameter WEIGHT_INIT_FILE = "mobilenet_onchip_bf16_wide.mem",
    parameter WEIGHT_BANK0_INIT_FILE = "mobilenet_onchip_bf16_bank0.mem",
    parameter WEIGHT_BANK1_INIT_FILE = "mobilenet_onchip_bf16_bank1.mem",
    parameter WEIGHT_BANK2_INIT_FILE = "mobilenet_onchip_bf16_bank2.mem",
    parameter WEIGHT_BANK3_INIT_FILE = "mobilenet_onchip_bf16_bank3.mem",
    parameter WEIGHT_WORDS = 245450,
    parameter WEIGHT_WIDE_WORDS = 64632,
    parameter BANK_WORDS = 12288,
    parameter MAX_TOTAL_K = 256,
    parameter CORE_NUM_PES = 4,
    parameter CORE_LOG2_PES = 2
) (
    input  wire clk,
    input  wire rst,
    input  wire i_image_wr_en,
    input  wire [9:0] i_image_wr_addr,
    input  wire [15:0] i_image_wr_data,
    input  wire i_start,
    output reg  o_busy,
    output reg  o_done,
    output reg  [3:0] o_prediction,
    output reg  o_error,
    output reg  [31:0] o_cycles,
    output reg  [31:0] o_load_cycles,
    output reg  [31:0] o_core_cycles,
    output reg  [31:0] o_post_cycles,
    output reg  [31:0] o_depthwise_cycles,
    output reg  [31:0] o_pointwise_cycles,
    output wire [7:0] o_layer
);
    localparam MAX_M = 16;
    localparam MAX_N = 16;
    localparam NUM_DPE = 16;
    localparam BENES_CTRL_WIDTH =
        2 * (2 * CORE_LOG2_PES - 1) * CORE_NUM_PES + CORE_NUM_PES;
    localparam MESH_ROUTE_WIDTH = 15;
    localparam LAYER_COUNT = 53;
    localparam LAST_LAYER = LAYER_COUNT - 1;

    localparam KIND_CONV = 2'd0;
    localparam KIND_DEPTHWISE = 2'd1;
    localparam KIND_LINEAR = 2'd2;

    localparam S_IDLE = 6'd0;
    localparam S_SET_LAYER = 6'd1;
    localparam S_LAYER_READY = 6'd2;
    localparam S_PREP_TILE = 6'd3;
    localparam S_A_REQ = 6'd4;
    localparam S_A_WAIT = 6'd5;
    localparam S_A_WRITE = 6'd6;
    localparam S_B_REQ = 6'd7;
    localparam S_B_WAIT = 6'd8;
    localparam S_B_WRITE = 6'd9;
    localparam S_CORE_START = 6'd10;
    localparam S_CORE_WAIT = 6'd11;
    localparam S_C_REQ = 6'd12;
    localparam S_C_WAIT = 6'd13;
    localparam S_C_CAPTURE = 6'd14;
    localparam S_PARTIAL_WAIT = 6'd15;
    localparam S_RESULT = 6'd16;
    localparam S_SKIP_WAIT = 6'd17;
    localparam S_SKIP_LAUNCH = 6'd18;
    localparam S_RESIDUAL_WAIT = 6'd19;
    localparam S_STORE_RESULT = 6'd20;
    localparam S_C_ADVANCE = 6'd21;
    localparam S_ADVANCE_M = 6'd22;
    localparam S_DONE = 6'd23;
    localparam S_A_COORD = 6'd24;
    localparam S_A_LINEAR = 6'd25;
    localparam S_B_INDEX = 6'd26;
    localparam S_B_ADDR = 6'd27;
    localparam S_REUSE_A_NEXT_N = 6'd28;
    localparam S_B_STREAM = 6'd29;
    localparam S_A_STREAM = 6'd30;
    localparam S_C_STREAM_INIT = 6'd31;
    localparam S_C_STREAM_ADDR = 6'd32;
    localparam S_C_STREAM = 6'd33;
    localparam S_PREP_COUNT = 6'd34;
    localparam S_DW_START = 6'd35;
    localparam S_DW_WAIT = 6'd36;
    localparam S_RES_STREAM_INIT = 6'd37;
    localparam S_RES_STREAM_ADDR = 6'd38;
    localparam S_RES_STREAM = 6'd39;
    localparam S_FAST_ARGMAX_REQ = 6'd40;
    localparam S_FAST_ARGMAX_WAIT = 6'd41;
    localparam S_FAST_ARGMAX_CAPTURE = 6'd42;
    localparam S_FAST_ARGMAX_COMPARE = 6'd43;
    localparam RES_ADD_LATENCY = 7;

    reg [5:0] state;
    reg [7:0] layer_idx;
    assign o_layer = layer_idx;

    // Generated layer descriptor registers.
    reg [1:0] layer_kind;
    reg [11:0] layer_in_c, layer_out_c;
    reg [5:0] layer_in_h, layer_in_w, layer_out_h, layer_out_w;
    reg [15:0] layer_M;
    reg [11:0] layer_N, layer_K;
    reg [1:0] layer_kernel, layer_stride, layer_pad;
    reg [18:0] weight_offset;
    reg [18:0] packed_weight_offset;
    reg [1:0] src_bank, dst_bank, skip_bank;
    reg layer_relu6, layer_residual;

    reg [15:0] m_base;
    reg [11:0] n_base, k_base;
    reg [4:0] tile_M, tile_N;
    reg [8:0] chunk_K;
    reg [5:0] tile_start_y, tile_start_x;
    reg [5:0] a_out_y, a_out_x;
    reg [4:0] a_row, b_col, c_row, c_col;
    reg [8:0] a_k, b_k;
    reg [11:0] a_channel;
    reg [1:0] a_kernel_y, a_kernel_x;
    reg [4:0] advance_count;
    reg reuse_B_tile;
    // Registered convolution address pipeline.  Chaining output-coordinate,
    // pixel-index and channel-stride multipliers in S_A_REQ produced the
    // former 5.9 ns board critical path.
    reg signed [7:0] a_calc_y, a_calc_x;
    reg [11:0] a_pixel_index;
    reg [15:0] a_issue_count, a_total_count;
    reg a_s1_valid, a_s1_bias, a_s1_linear;
    reg [15:0] a_s1_core_addr;
    reg [11:0] a_s1_global_k, a_s1_channel;
    reg signed [7:0] a_s1_calc_y, a_s1_calc_x;
    reg a_s2_valid, a_s2_direct, a_s2_linear;
    reg [15:0] a_s2_core_addr;
    reg [15:0] a_s2_direct_data;
    reg [11:0] a_s2_global_k, a_s2_channel, a_s2_pixel;
    reg a_resp_valid_d1, a_resp_valid_d2;
    reg a_resp_mem_d1, a_resp_mem_d2;
    reg [15:0] a_resp_addr_d1, a_resp_addr_d2;
    reg [15:0] a_resp_data_d1, a_resp_data_d2;
    // Registered weight-address pipeline.  Keep the index additions,
    // variable multiply and ROM-offset addition in separate cycles so Vivado
    // cannot build the former two-DSP/12-level combinational chain.
    reg [11:0] b_global_k, b_global_n;
    reg [23:0] b_weight_product;
    reg [17:0] b_stream_weight_addr;
    reg [9:0] b_issue_count;
    reg b_read_valid_d1, b_read_valid_d2, b_read_valid_d3, b_read_valid_d4,
        b_read_valid_d5;
    reg [15:0] b_core_addr_d1, b_core_addr_d2, b_core_addr_d3,
               b_core_addr_d4, b_core_addr_d5;

    // Reuse the existing three-cycle C-read sequence to pipeline the output
    // activation address.  This avoids a second row-index multiply/add chain
    // becoming critical after the weight path is removed.
    reg [15:0] output_row_index;
    reg [27:0] output_row_base;
    reg [13:0] result_linear_addr;
    reg [8:0] c_issue_count, c_resp_count, c_total_count;
    reg [4:0] c_resp_row, c_resp_col;
    reg [13:0] c_resp_linear_addr;
    // Residual outputs are independent.  Issue one C/skip pair per clock and
    // carry its activation address beside the seven-edge FP32 add pipeline.
    // The former state machine waited roughly sixteen clocks per element.
    reg [8:0] res_issue_count, res_total_count;
    reg [4:0] res_issue_row, res_issue_col;
    reg [13:0] res_issue_linear_addr, res_capture_linear_addr;
    reg [15:0] skip_data_d1;
    reg [RES_ADD_LATENCY-1:0] res_add_valid_pipe;
    reg [RES_ADD_LATENCY-1:0] res_add_last_pipe;
    reg [13:0] res_addr_pipe [0:RES_ADD_LATENCY-1];
    integer res_pipe_index;

    reg weight_rd_en;
    reg [17:0] weight_rd_addr;
    wire [15:0] weight_rd_data;
    reg activation_rd_en;
    reg [1:0] activation_rd_bank;
    reg [13:0] activation_rd_addr;
    wire [15:0] activation_rd_data;
    reg skip_rd_en;
    reg [1:0] skip_rd_bank;
    reg [13:0] skip_rd_addr;
    wire [15:0] skip_rd_data;
    reg ctrl_activation_wr_en;
    reg [1:0] ctrl_activation_wr_bank;
    reg [13:0] ctrl_activation_wr_addr;
    reg [15:0] ctrl_activation_wr_data;

    reg dw_start;
    reg fast_mode_m1;
    wire dw_busy, dw_done, dw_error;
    wire [31:0] dw_layer_cycles;
    wire dw_weight_rd_en;
    wire [17:0] dw_weight_rd_addr;
    wire dw_weight_wide_rd_en;
    wire [17:0] dw_weight_wide_rd_addr;
    wire dw_weight_super_rd_en;
    wire [17:0] dw_weight_super_rd_addr;
    wire dw_activation_rd_en;
    wire [1:0] dw_activation_rd_bank;
    wire [13:0] dw_activation_rd_addr;
    wire dw_activation_wide_rd_en;
    wire [1:0] dw_activation_wide_rd_bank;
    wire [13:0] dw_activation_wide_rd_addr;
    wire [63:0] activation_wide_rd_data;
    wire dw_activation_wr_en;
    wire [1:0] dw_activation_wr_bank;
    wire [13:0] dw_activation_wr_addr;
    wire [15:0] dw_activation_wr_data;
    wire dw_activation_wide_wr_en;
    wire [1:0] dw_activation_wide_wr_bank;
    wire [13:0] dw_activation_wide_wr_addr;
    wire [63:0] dw_activation_wide_wr_data;

    wire store_weight_rd_en = dw_busy ? dw_weight_rd_en : weight_rd_en;
    wire [17:0] store_weight_rd_addr = dw_busy ? dw_weight_rd_addr :
                                                       weight_rd_addr;
    reg weight_wide_rd_en;
    reg [17:0] weight_wide_rd_addr;
    wire store_weight_wide_rd_en = dw_busy ? dw_weight_wide_rd_en :
                                              weight_wide_rd_en;
    wire [17:0] store_weight_wide_rd_addr = dw_busy ?
        dw_weight_wide_rd_addr : weight_wide_rd_addr;
    wire [63:0] weight_wide_rd_data;
    wire [255:0] weight_super_rd_data;
    wire store_activation_rd_en = dw_busy ? dw_activation_rd_en :
                                                    activation_rd_en;
    wire [1:0] store_activation_rd_bank = dw_busy ? dw_activation_rd_bank :
                                                           activation_rd_bank;
    wire [13:0] store_activation_rd_addr = dw_busy ? dw_activation_rd_addr :
                                                            activation_rd_addr;

    wire store_wr_en = o_busy ? (dw_busy ? dw_activation_wr_en :
                                           ctrl_activation_wr_en) :
                                    i_image_wr_en;
    wire [1:0] store_wr_bank = o_busy ? (dw_busy ? dw_activation_wr_bank :
                                                   ctrl_activation_wr_bank) :
                                      2'd0;
    wire [13:0] store_wr_addr = o_busy ? (dw_busy ? dw_activation_wr_addr :
                                                    ctrl_activation_wr_addr) :
                                      {4'd0, i_image_wr_addr};
    wire [15:0] store_wr_data = o_busy ? (dw_busy ? dw_activation_wr_data :
                                                    ctrl_activation_wr_data) :
                                      i_image_wr_data;
    wire store_wide_wr_en = o_busy && dw_busy && dw_activation_wide_wr_en;

    sigma_mobilenet_onchip_store #(
        .WEIGHT_WORDS(WEIGHT_WORDS),
        .WEIGHT_WIDE_WORDS(WEIGHT_WIDE_WORDS),
        .BANK_WORDS(BANK_WORDS),
        .WEIGHT_INIT_FILE(WEIGHT_INIT_FILE),
        .WEIGHT_BANK0_INIT_FILE(WEIGHT_BANK0_INIT_FILE),
        .WEIGHT_BANK1_INIT_FILE(WEIGHT_BANK1_INIT_FILE),
        .WEIGHT_BANK2_INIT_FILE(WEIGHT_BANK2_INIT_FILE),
        .WEIGHT_BANK3_INIT_FILE(WEIGHT_BANK3_INIT_FILE)
    ) u_store (
        .clk(clk),
        .i_weight_rd_en(store_weight_rd_en),
        .i_weight_rd_addr(store_weight_rd_addr),
        .o_weight_rd_data(weight_rd_data),
        .i_weight_wide_rd_en(store_weight_wide_rd_en),
        .i_weight_wide_rd_addr(store_weight_wide_rd_addr),
        .o_weight_wide_rd_data(weight_wide_rd_data),
        .i_weight_super_rd_en(dw_busy && dw_weight_super_rd_en),
        .i_weight_super_rd_addr(dw_weight_super_rd_addr),
        .o_weight_super_rd_data(weight_super_rd_data),
        .i_activation_wr_en(store_wr_en),
        .i_activation_wr_bank(store_wr_bank),
        .i_activation_wr_addr(store_wr_addr),
        .i_activation_wr_data(store_wr_data),
        .i_activation_wide_wr_en(store_wide_wr_en),
        .i_activation_wide_wr_bank(dw_activation_wide_wr_bank),
        .i_activation_wide_wr_addr(dw_activation_wide_wr_addr),
        .i_activation_wide_wr_data(dw_activation_wide_wr_data),
        .i_activation_rd_en(store_activation_rd_en),
        .i_activation_rd_bank(store_activation_rd_bank),
        .i_activation_rd_addr(store_activation_rd_addr),
        .o_activation_rd_data(activation_rd_data),
        .i_activation_wide_rd_en(dw_busy && dw_activation_wide_rd_en),
        .i_activation_wide_rd_bank(dw_activation_wide_rd_bank),
        .i_activation_wide_rd_addr(dw_activation_wide_rd_addr),
        .o_activation_wide_rd_data(activation_wide_rd_data),
        .i_skip_rd_en(skip_rd_en),
        .i_skip_rd_bank(skip_rd_bank),
        .i_skip_rd_addr(skip_rd_addr),
        .o_skip_rd_data(skip_rd_data)
    );

    sigma_mobilenet_depthwise4 u_depthwise4 (
        .clk(clk), .rst(rst), .i_start(dw_start),
        .i_mode_m1(fast_mode_m1),
        .i_M(layer_M), .i_N(layer_N), .i_K(layer_K),
        .i_channels(layer_N),
        .i_in_h(layer_in_h), .i_in_w(layer_in_w),
        .i_out_h(layer_out_h), .i_out_w(layer_out_w),
        .i_stride(layer_stride), .i_pad(layer_pad),
        .i_weight_offset(packed_weight_offset),
        .i_src_bank(src_bank), .i_dst_bank(dst_bank),
        .i_relu6(layer_relu6),
        .o_weight_rd_en(dw_weight_rd_en),
        .o_weight_rd_addr(dw_weight_rd_addr),
        .i_weight_rd_data(weight_rd_data),
        .o_weight_wide_rd_en(dw_weight_wide_rd_en),
        .o_weight_wide_rd_addr(dw_weight_wide_rd_addr),
        .i_weight_wide_rd_data(weight_wide_rd_data),
        .o_weight_super_rd_en(dw_weight_super_rd_en),
        .o_weight_super_rd_addr(dw_weight_super_rd_addr),
        .i_weight_super_rd_data(weight_super_rd_data),
        .o_activation_rd_en(dw_activation_rd_en),
        .o_activation_rd_bank(dw_activation_rd_bank),
        .o_activation_rd_addr(dw_activation_rd_addr),
        .i_activation_rd_data(activation_rd_data),
        .o_activation_wide_rd_en(dw_activation_wide_rd_en),
        .o_activation_wide_rd_bank(dw_activation_wide_rd_bank),
        .o_activation_wide_rd_addr(dw_activation_wide_rd_addr),
        .i_activation_wide_rd_data(activation_wide_rd_data),
        .o_activation_wr_en(dw_activation_wr_en),
        .o_activation_wr_bank(dw_activation_wr_bank),
        .o_activation_wr_addr(dw_activation_wr_addr),
        .o_activation_wr_data(dw_activation_wr_data),
        .o_activation_wide_wr_en(dw_activation_wide_wr_en),
        .o_activation_wide_wr_bank(dw_activation_wide_wr_bank),
        .o_activation_wide_wr_addr(dw_activation_wide_wr_addr),
        .o_activation_wide_wr_data(dw_activation_wide_wr_data),
        .o_busy(dw_busy), .o_done(dw_done), .o_error(dw_error),
        .o_cycles(dw_layer_cycles)
    );

    reg core_start, core_A_wr_en, core_B_wr_en, core_B_wide_wr_en;
    reg [15:0] core_A_wr_addr, core_B_wr_addr;
    reg [15:0] core_A_wr_data, core_B_wr_data;
    reg [15:0] core_B_wide_wr_addr;
    reg [63:0] core_B_wide_wr_data;
    reg core_C_rd_en;
    reg [15:0] core_C_rd_addr;
    wire [31:0] core_C_rd_data;
    wire core_C_rd_valid;
    wire core_busy, core_done, core_error;

    sigma_fold_core #(
        .NUM_PES(CORE_NUM_PES), .LOG2_PES(CORE_LOG2_PES),
        .MESH_ROWS(4), .MESH_COLS(4),
        .MAX_M(MAX_M), .MAX_N(MAX_N), .FOLD_K(CORE_NUM_PES),
        .MAX_TOTAL_K(MAX_TOTAL_K), .AUTO_BENES_CONFIG(1),
        .PIPELINED_FOLD_ACCUM(1), .FOLD_ACCUM_LANES(4),
        // The eight A/B fold banks are 1024x16 each.  Mapping these regular
        // memories into BRAM costs only eight RAMB18 primitives and removes
        // thousands of LUTRAM address/data endpoints from the 300 MHz route.
        // The registered model ROM still leaves enough BRAM on the ZU-5EV.
        .SCRATCH_RAM_STYLE("block"),
        // This covers both folded and inner SIGMA result banks.  The inner
        // banks are only 16x32 bits; LUTRAM avoids wasting 16 RAMB18 blocks
        // and leaves room for the three-BRAM JTAG-to-AXI transport.
        .DIRECT_BANK_WRITE(1), .RESULT_RAM_STYLE("distributed")
    ) u_core (
        .clk(clk), .rst(rst), .i_start(core_start),
        // Explicit extension is required here.  Leaving the formal MSBs
        // implicit makes XSim drive them as Z for these variable-width ports,
        // which prevents sigma_fold_core's load-limit comparisons completing.
        .i_M({3'b000, tile_M}), .i_N({3'b000, tile_N}),
        .i_K_total({7'b0000000, chunk_K}),
        .i_A_wr_en(core_A_wr_en), .i_A_wr_addr(core_A_wr_addr),
        .i_A_wr_data(core_A_wr_data),
        .i_B_wr_en(core_B_wr_en), .i_B_wr_addr(core_B_wr_addr),
        .i_B_wr_data(core_B_wr_data),
        .i_B_wide_wr_en(core_B_wide_wr_en),
        .i_B_wide_wr_addr(core_B_wide_wr_addr),
        .i_B_wide_wr_data(core_B_wide_wr_data),
        .i_benes_config({NUM_DPE*BENES_CTRL_WIDTH{1'b0}}),
        .i_mesh_route_enable(1'b0),
        .i_mesh_route_config({NUM_DPE*MESH_ROUTE_WIDTH{1'b0}}),
        .i_mesh_hops(48'd0),
        .i_C_rd_en(core_C_rd_en), .i_C_rd_addr(core_C_rd_addr),
        .o_C_rd_data(core_C_rd_data), .o_C_rd_valid(core_C_rd_valid),
        .o_busy(core_busy), .o_done(core_done), .o_error(core_error),
        .o_C_row(), .o_C_valid(), .o_config_active(), .o_config_fold(),
        .o_config_tile(), .o_config_row()
    );

    (* ram_style = "distributed" *) reg [31:0] partial_mem [0:MAX_M*MAX_N-1];
    reg [31:0] add_a, add_b;
    wire [31:0] add_out;
    reg [3:0] add_wait;
    fp32_add_pipeline u_post_add(
        .clk(clk), .rst(rst), .a(add_a), .b(add_b), .out(add_out));

    reg [31:0] result_value;
    reg argmax_valid;
    reg [31:0] argmax_key;
    reg [3:0] argmax_index;
    reg [11:0] fast_argmax_index;
    reg [31:0] argmax_candidate_key;
    reg [3:0] argmax_candidate_index;

    integer calc_addr;
    integer global_k;
    reg [31:0] processed_value;

    function [31:0] float_key;
        input [31:0] value;
        begin float_key = value[31] ? ~value : (value ^ 32'h80000000); end
    endfunction

    function [31:0] relu6_fp32;
        input [31:0] value;
        begin
            if (value[31])
                relu6_fp32 = 0;
            else if (float_key(value) > float_key(32'h40c00000))
                relu6_fp32 = 32'h40c00000;
            else
                relu6_fp32 = value;
        end
    endfunction

    function [15:0] fp32_to_bf16;
        input [31:0] value;
        reg [32:0] rounded;
        begin
            rounded = {1'b0, value} + 33'h000007fff + value[16];
            fp32_to_bf16 = rounded[31:16];
        end
    endfunction

    wire final_chunk = (k_base + chunk_K >= layer_K);
    wire [7:0] partial_addr = c_row * MAX_N + c_col;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            layer_idx <= 0;
            o_busy <= 0;
            o_done <= 0;
            o_prediction <= 0;
            o_error <= 0;
            o_cycles <= 0;
            o_load_cycles <= 0;
            o_core_cycles <= 0;
            o_post_cycles <= 0;
            o_depthwise_cycles <= 0;
            o_pointwise_cycles <= 0;
            dw_start <= 0;
            fast_mode_m1 <= 0;
            core_start <= 0;
            core_A_wr_en <= 0;
            core_B_wr_en <= 0;
            core_B_wide_wr_en <= 0;
            core_C_rd_en <= 0;
            weight_rd_en <= 0;
            weight_wide_rd_en <= 0;
            activation_rd_en <= 0;
            skip_rd_en <= 0;
            ctrl_activation_wr_en <= 0;
            argmax_valid <= 0;
            argmax_key <= 0;
            argmax_candidate_key <= 0;
            argmax_candidate_index <= 0;
            reuse_B_tile <= 0;
            add_a <= 0;
            add_b <= 0;
            a_calc_y <= 0;
            a_calc_x <= 0;
            a_pixel_index <= 0;
            a_issue_count <= 0;
            a_total_count <= 0;
            a_s1_valid <= 0;
            a_s2_valid <= 0;
            a_resp_valid_d1 <= 0;
            a_resp_valid_d2 <= 0;
            a_resp_mem_d1 <= 0;
            a_resp_mem_d2 <= 0;
            b_global_k <= 0;
            b_global_n <= 0;
            b_weight_product <= 0;
            b_stream_weight_addr <= 0;
            b_issue_count <= 0;
            b_read_valid_d1 <= 0;
            b_read_valid_d2 <= 0;
            b_read_valid_d3 <= 0;
            b_read_valid_d4 <= 0;
            b_read_valid_d5 <= 0;
            b_core_addr_d1 <= 0;
            b_core_addr_d2 <= 0;
            b_core_addr_d3 <= 0;
            b_core_addr_d4 <= 0;
            b_core_addr_d5 <= 0;
            output_row_index <= 0;
            output_row_base <= 0;
            result_linear_addr <= 0;
            c_issue_count <= 0;
            c_resp_count <= 0;
            c_total_count <= 0;
            c_resp_row <= 0;
            c_resp_col <= 0;
            c_resp_linear_addr <= 0;
            res_issue_count <= 0;
            res_total_count <= 0;
            res_issue_row <= 0;
            res_issue_col <= 0;
            res_issue_linear_addr <= 0;
            res_capture_linear_addr <= 0;
            skip_data_d1 <= 0;
            res_add_valid_pipe <= 0;
            res_add_last_pipe <= 0;
            for (res_pipe_index = 0; res_pipe_index < RES_ADD_LATENCY;
                 res_pipe_index = res_pipe_index + 1)
                res_addr_pipe[res_pipe_index] <= 0;
        end else begin
            dw_start <= 0;
            core_start <= 0;
            core_A_wr_en <= 0;
            core_B_wr_en <= 0;
            core_B_wide_wr_en <= 0;
            core_C_rd_en <= 0;
            weight_rd_en <= 0;
            weight_wide_rd_en <= 0;
            activation_rd_en <= 0;
            skip_rd_en <= 0;
            ctrl_activation_wr_en <= 0;
            o_done <= 0;
            // Align the one-cycle activation-store response with the
            // two-cycle external C-bank response.  The FP32 adder accepts a
            // new independent residual pair every clock.
            skip_data_d1 <= skip_rd_data;
            for (res_pipe_index = RES_ADD_LATENCY-1; res_pipe_index > 0;
                 res_pipe_index = res_pipe_index - 1) begin
                res_add_valid_pipe[res_pipe_index] <=
                    res_add_valid_pipe[res_pipe_index-1];
                res_add_last_pipe[res_pipe_index] <=
                    res_add_last_pipe[res_pipe_index-1];
                res_addr_pipe[res_pipe_index] <=
                    res_addr_pipe[res_pipe_index-1];
            end
            res_add_valid_pipe[0] <= 1'b0;
            res_add_last_pipe[0] <= 1'b0;
            if (o_busy)
                o_cycles <= o_cycles + 1'b1;

            // Performance counters deliberately classify only productive
            // work.  The difference between o_cycles and these four counters
            // is controller/tile-transition overhead and is reported by the
            // board software instead of being hidden in the compute number.
            if (o_busy) begin
                case (state)
                    S_A_REQ, S_A_WAIT, S_A_WRITE, S_B_REQ, S_B_WAIT,
                    S_B_WRITE, S_A_COORD, S_A_LINEAR, S_B_INDEX, S_B_ADDR,
                    S_B_STREAM, S_A_STREAM, S_PREP_COUNT,
                    S_REUSE_A_NEXT_N:
                        o_load_cycles <= o_load_cycles + 1'b1;
                    S_CORE_START, S_CORE_WAIT:
                        o_core_cycles <= o_core_cycles + 1'b1;
                    S_C_REQ, S_C_WAIT, S_C_CAPTURE, S_PARTIAL_WAIT,
                    S_RESULT, S_SKIP_WAIT, S_SKIP_LAUNCH,
                    S_RESIDUAL_WAIT, S_STORE_RESULT, S_C_ADVANCE,
                     S_C_STREAM_INIT, S_C_STREAM_ADDR, S_C_STREAM,
                     S_RES_STREAM_INIT, S_RES_STREAM_ADDR, S_RES_STREAM,
                     S_FAST_ARGMAX_REQ, S_FAST_ARGMAX_WAIT,
                     S_FAST_ARGMAX_CAPTURE, S_FAST_ARGMAX_COMPARE:
                        o_post_cycles <= o_post_cycles + 1'b1;
                    S_DW_START, S_DW_WAIT:
                        if (fast_mode_m1)
                            o_pointwise_cycles <= o_pointwise_cycles + 1'b1;
                        else
                            o_depthwise_cycles <= o_depthwise_cycles + 1'b1;
                    default: ;
                endcase
            end

            case (state)
                S_IDLE: begin
                    o_busy <= 0;
                    if (i_start) begin
                        o_busy <= 1;
                        o_error <= 0;
                        o_cycles <= 0;
                        o_load_cycles <= 0;
                        o_core_cycles <= 0;
                        o_post_cycles <= 0;
                        o_depthwise_cycles <= 0;
                        o_pointwise_cycles <= 0;
                        o_prediction <= 0;
                        layer_idx <= 0;
                        state <= S_SET_LAYER;
                    end
                end

                S_SET_LAYER: begin
                    case (layer_idx)
`define SIGMA_MOBILENET_LAYER_CASE
`include "sigma_mobilenet_layers.vh"
`undef SIGMA_MOBILENET_LAYER_CASE
                    endcase
                    state <= S_LAYER_READY;
                end

                S_LAYER_READY: begin
                    if ((layer_M == 0) || (layer_N == 0) || (layer_K == 0) ||
                        (layer_M > 16'h3fff) ||
                        ((layer_kind != KIND_LINEAR) && (layer_K > MAX_TOTAL_K))) begin
                        o_error <= 1;
                        state <= S_DONE;
                    end else begin
                        m_base <= 0;
                        n_base <= 0;
                        k_base <= 0;
                        tile_start_y <= 0;
                        tile_start_x <= 0;
                        reuse_B_tile <= 0;
                        if (layer_idx == LAST_LAYER)
                            argmax_valid <= 0;
                        if ((layer_kind == KIND_DEPTHWISE) &&
                            (layer_K == 10) &&
                            (layer_N == layer_in_c) && !layer_residual) begin
                            fast_mode_m1 <= 0;
                            state <= S_DW_START;
                        end else if ((layer_M == 1) && !layer_residual) begin
                            fast_mode_m1 <= 1;
                            state <= S_DW_START;
                        end else
                            state <= S_PREP_TILE;
                    end
                end

                S_DW_START: begin
                    // Depthwise K=10 is a regular 3x3+bias dot product.  Four
                    // spatial outputs are launched together through the same
                    // BF16 multiplier/FAN arithmetic used by a SIGMA DPE.
                    dw_start <= 1;
                    state <= S_DW_WAIT;
                end

                S_DW_WAIT: begin
                    if (dw_done) begin
                        if (dw_error) begin
                            o_error <= 1;
                            state <= S_DONE;
                        end else if (layer_idx == LAST_LAYER) begin
                            // The streamed M=1 classifier writes ten BF16
                            // logits to its destination bank.  Read them back
                            // locally and choose the largest; no CPU/off-chip
                            // post-processing is involved.
                            fast_argmax_index <= 0;
                            argmax_valid <= 0;
                            state <= S_FAST_ARGMAX_REQ;
                        end else begin
                            layer_idx <= layer_idx + 1'b1;
                            state <= S_SET_LAYER;
                        end
                    end
                end

                S_FAST_ARGMAX_REQ: begin
                    activation_rd_en <= 1'b1;
                    activation_rd_bank <= dst_bank;
                    activation_rd_addr <= fast_argmax_index;
                    state <= S_FAST_ARGMAX_WAIT;
                end

                S_FAST_ARGMAX_WAIT: begin
                    // One controller cycle plus the store's registered read
                    // output keeps the address/data association explicit.
                    state <= S_FAST_ARGMAX_CAPTURE;
                end

                S_FAST_ARGMAX_CAPTURE: begin
                    // Register the sortable key next to the argmax registers.
                    // This keeps the 32-bit comparison off the distant SIGMA
                    // result-bus path at 300 MHz.
                    argmax_candidate_key <=
                        float_key({activation_rd_data, 16'h0000});
                    argmax_candidate_index <= fast_argmax_index[3:0];
                    state <= S_FAST_ARGMAX_COMPARE;
                end

                S_FAST_ARGMAX_COMPARE: begin
                    if (!argmax_valid ||
                        (argmax_candidate_key > argmax_key)) begin
                        argmax_valid <= 1'b1;
                        argmax_key <= argmax_candidate_key;
                        argmax_index <= argmax_candidate_index;
                    end
                    if (argmax_candidate_index + 1 >= layer_N) begin
                        if (!argmax_valid ||
                            (argmax_candidate_key > argmax_key))
                            o_prediction <= argmax_candidate_index;
                        else
                            o_prediction <= argmax_index;
                        state <= S_DONE;
                    end else begin
                        fast_argmax_index <= argmax_candidate_index + 1'b1;
                        state <= S_FAST_ARGMAX_REQ;
                    end
                end

                S_PREP_TILE: begin
                    // Register the clipped tile dimensions first.  Computing
                    // a_total_count directly from layer_M/layer_K in this
                    // state chained subtract, compare, mux and DSP multiply
                    // into one cycle and was the PS build's critical path.
                    tile_M <= ((layer_M - m_base) > MAX_M) ? MAX_M :
                              (layer_M - m_base);
                    if (layer_kind == KIND_DEPTHWISE)
                        tile_N <= 1;
                    else
                        tile_N <= ((layer_N - n_base) > MAX_N) ? MAX_N :
                                  (layer_N - n_base);
                    chunk_K <= ((layer_K - k_base) > MAX_TOTAL_K) ?
                               MAX_TOTAL_K : (layer_K - k_base);
                    a_row <= 0;
                    a_k <= 0;
                    a_channel <= 0;
                    a_kernel_y <= 0;
                    a_kernel_x <= 0;
                    a_out_y <= tile_start_y;
                    a_out_x <= tile_start_x;
                    a_issue_count <= 0;
                    a_s1_valid <= 0;
                    a_s2_valid <= 0;
                    a_resp_valid_d1 <= 0;
                    a_resp_valid_d2 <= 0;
                    state <= S_PREP_COUNT;
                end

                S_PREP_COUNT: begin
                    // Both operands now come directly from registers.  The
                    // explicit extensions preserve the full 16*256 = 4096
                    // range and keep the multiplier as a single DSP stage.
                    a_total_count <= {11'd0, tile_M} * {7'd0, chunk_K};
                    state <= S_A_STREAM;
                end

                S_A_STREAM: begin
                    // Five-stage address/read/write pipeline.  Coordinate and
                    // HWC address multiplies stay on separate registered
                    // stages, while the engine accepts one A word per clock.
                    a_resp_valid_d2 <= a_resp_valid_d1;
                    a_resp_mem_d2 <= a_resp_mem_d1;
                    a_resp_addr_d2 <= a_resp_addr_d1;
                    a_resp_data_d2 <= a_resp_data_d1;
                    a_resp_valid_d1 <= 1'b0;
                    a_s2_valid <= 1'b0;
                    a_s1_valid <= 1'b0;

                    if (a_resp_valid_d2) begin
                        core_A_wr_en <= 1'b1;
                        core_A_wr_addr <= a_resp_addr_d2;
                        core_A_wr_data <= a_resp_mem_d2 ?
                                          activation_rd_data : a_resp_data_d2;
                    end

                    if (a_s2_valid) begin
                        a_resp_valid_d1 <= 1'b1;
                        a_resp_addr_d1 <= a_s2_core_addr;
                        if (a_s2_direct) begin
                            a_resp_mem_d1 <= 1'b0;
                            a_resp_data_d1 <= a_s2_direct_data;
                        end else begin
                            a_resp_mem_d1 <= 1'b1;
                            activation_rd_en <= 1'b1;
                            activation_rd_bank <= src_bank;
                            if (a_s2_linear)
                                activation_rd_addr <= {2'b00, a_s2_global_k};
                            else
                                activation_rd_addr <=
                                    a_s2_pixel * layer_in_c + a_s2_channel;
                        end
                    end

                    if (a_s1_valid) begin
                        a_s2_valid <= 1'b1;
                        a_s2_core_addr <= a_s1_core_addr;
                        a_s2_global_k <= a_s1_global_k;
                        a_s2_channel <= a_s1_channel;
                        a_s2_linear <= a_s1_linear;
                        if (a_s1_bias) begin
                            a_s2_direct <= 1'b1;
                            a_s2_direct_data <= 16'h3f80;
                        end else if (a_s1_linear) begin
                            a_s2_direct <= 1'b0;
                        end else if ((a_s1_calc_y < 0) ||
                                     (a_s1_calc_x < 0) ||
                                     (a_s1_calc_y >= $signed({1'b0, layer_in_h})) ||
                                     (a_s1_calc_x >= $signed({1'b0, layer_in_w}))) begin
                            a_s2_direct <= 1'b1;
                            a_s2_direct_data <= 16'h0000;
                        end else begin
                            a_s2_direct <= 1'b0;
                            a_s2_pixel <=
                                a_s1_calc_y * $signed({1'b0, layer_in_w}) +
                                a_s1_calc_x;
                        end
                    end

                    if (a_issue_count < a_total_count) begin
                        global_k = k_base + a_k;
                        a_s1_valid <= 1'b1;
                        a_s1_core_addr <= a_row * MAX_TOTAL_K + a_k;
                        a_s1_global_k <= global_k;
                        a_s1_channel <= (layer_kind == KIND_DEPTHWISE) ?
                                        n_base : a_channel;
                        a_s1_bias <= (global_k + 1 >= layer_K);
                        a_s1_linear <= (layer_kind == KIND_LINEAR);
                        a_s1_calc_y <= $signed({1'b0, a_out_y}) *
                                       $signed({1'b0, layer_stride}) +
                                       $signed({1'b0, a_kernel_y}) -
                                       $signed({1'b0, layer_pad});
                        a_s1_calc_x <= $signed({1'b0, a_out_x}) *
                                       $signed({1'b0, layer_stride}) +
                                       $signed({1'b0, a_kernel_x}) -
                                       $signed({1'b0, layer_pad});
                        a_issue_count <= a_issue_count + 1'b1;

                        if (a_k + 1 >= chunk_K) begin
                            a_k <= 0;
                            a_channel <= 0;
                            a_kernel_y <= 0;
                            a_kernel_x <= 0;
                            if (a_row + 1 < tile_M) begin
                                a_row <= a_row + 1'b1;
                                if (a_out_x + 1 >= layer_out_w) begin
                                    a_out_x <= 0;
                                    a_out_y <= a_out_y + 1'b1;
                                end else
                                    a_out_x <= a_out_x + 1'b1;
                            end
                        end else begin
                            a_k <= a_k + 1'b1;
                            if ((layer_kind != KIND_LINEAR) &&
                                (global_k + 1 < layer_K - 1)) begin
                                if (a_kernel_x + 1 >= layer_kernel) begin
                                    a_kernel_x <= 0;
                                    if (a_kernel_y + 1 >= layer_kernel) begin
                                        a_kernel_y <= 0;
                                        if (layer_kind == KIND_CONV)
                                            a_channel <= a_channel + 1'b1;
                                    end else
                                        a_kernel_y <= a_kernel_y + 1'b1;
                                end else
                                    a_kernel_x <= a_kernel_x + 1'b1;
                            end
                        end
                    end

                    if ((a_issue_count >= a_total_count) &&
                        !a_s1_valid && !a_s2_valid &&
                        !a_resp_valid_d1 && !a_resp_valid_d2 &&
                        !core_A_wr_en) begin
                        b_col <= 0;
                        b_k <= 0;
                        state <= reuse_B_tile ? S_CORE_START : S_B_REQ;
                    end
                end

                S_A_REQ: begin
                    global_k = k_base + a_k;
                    core_A_wr_addr <= a_row * MAX_TOTAL_K + a_k;
                    if (global_k + 1 >= layer_K) begin
                        core_A_wr_data <= 16'h3f80;
                        core_A_wr_en <= 1;
                        state <= S_A_WRITE;
                    end else if (layer_kind == KIND_LINEAR) begin
                        activation_rd_en <= 1;
                        activation_rd_bank <= src_bank;
                        activation_rd_addr <= global_k[13:0];
                        state <= S_A_WAIT;
                    end else begin
                        a_calc_y <= $signed({1'b0, a_out_y}) *
                                    $signed({1'b0, layer_stride}) +
                                    $signed({1'b0, a_kernel_y}) -
                                    $signed({1'b0, layer_pad});
                        a_calc_x <= $signed({1'b0, a_out_x}) *
                                    $signed({1'b0, layer_stride}) +
                                    $signed({1'b0, a_kernel_x}) -
                                    $signed({1'b0, layer_pad});
                        state <= S_A_COORD;
                    end
                end

                S_A_COORD: begin
                    if ((a_calc_y < 0) || (a_calc_x < 0) ||
                        (a_calc_y >= $signed({1'b0, layer_in_h})) ||
                        (a_calc_x >= $signed({1'b0, layer_in_w}))) begin
                        core_A_wr_data <= 0;
                        core_A_wr_en <= 1;
                        state <= S_A_WRITE;
                    end else begin
                        a_pixel_index <=
                            (a_calc_y * $signed({1'b0, layer_in_w})) + a_calc_x;
                        state <= S_A_LINEAR;
                    end
                end

                S_A_LINEAR: begin
                    calc_addr = a_pixel_index * layer_in_c;
                    if (layer_kind == KIND_DEPTHWISE)
                        calc_addr = calc_addr + n_base;
                    else
                        calc_addr = calc_addr + a_channel;
                    activation_rd_en <= 1;
                    activation_rd_bank <= src_bank;
                    activation_rd_addr <= calc_addr[13:0];
                    state <= S_A_WAIT;
                end

                S_A_WAIT: state <= S_A_WRITE;

                S_A_WRITE: begin
                    if (!core_A_wr_en) begin
                        core_A_wr_en <= 1;
                        core_A_wr_addr <= a_row * MAX_TOTAL_K + a_k;
                        core_A_wr_data <= activation_rd_data;
                    end
                    if (a_k + 1 >= chunk_K) begin
                        a_k <= 0;
                        a_channel <= 0;
                        a_kernel_y <= 0;
                        a_kernel_x <= 0;
                        if (a_row + 1 >= tile_M) begin
                            b_col <= 0;
                            b_k <= 0;
                            state <= S_B_REQ;
                        end else begin
                            a_row <= a_row + 1'b1;
                            if (a_out_x + 1 >= layer_out_w) begin
                                a_out_x <= 0;
                                a_out_y <= a_out_y + 1'b1;
                            end else
                                a_out_x <= a_out_x + 1'b1;
                            state <= S_A_REQ;
                        end
                    end else begin
                        a_k <= a_k + 1'b1;
                        if ((layer_kind != KIND_LINEAR) &&
                            (k_base + a_k + 1 < layer_K - 1)) begin
                            if (a_kernel_x + 1 >= layer_kernel) begin
                                a_kernel_x <= 0;
                                if (a_kernel_y + 1 >= layer_kernel) begin
                                    a_kernel_y <= 0;
                                    if (layer_kind == KIND_CONV)
                                        a_channel <= a_channel + 1'b1;
                                end else
                                    a_kernel_y <= a_kernel_y + 1'b1;
                            end else
                                a_kernel_x <= a_kernel_x + 1'b1;
                        end
                        state <= S_A_REQ;
                    end
                end

                S_B_REQ: begin
                    b_global_k <= k_base + b_k;
                    b_global_n <= n_base + b_col;
                    state <= S_B_INDEX;
                end

                S_B_INDEX: begin
                    if (layer_kind == KIND_DEPTHWISE)
                        b_weight_product <= b_global_n * layer_K;
                    else
                        b_weight_product <= b_global_n *
                            ((layer_K + 3) >> 2);
                    state <= S_B_ADDR;
                end

                S_B_ADDR: begin
                    if (layer_kind == KIND_DEPTHWISE)
                        b_stream_weight_addr <= weight_offset +
                                                b_weight_product + b_global_k;
                    else
                        b_stream_weight_addr <= packed_weight_offset +
                                                b_weight_product +
                                                (b_global_k >> 2);
                    b_issue_count <= 0;
                    b_read_valid_d1 <= 0;
                    b_read_valid_d2 <= 0;
                    b_read_valid_d3 <= 0;
                    b_read_valid_d4 <= 0;
                    b_read_valid_d5 <= 0;
                    state <= S_B_STREAM;
                end

                S_B_STREAM: begin
                    // Each packed ROM row is one complete four-PE K fold.  The
                    // four independent B banks therefore receive four BF16
                    // weights per clock after the registered BRAM pipeline.
                    b_read_valid_d5 <= b_read_valid_d4;
                    b_core_addr_d5 <= b_core_addr_d4;
                    b_read_valid_d4 <= b_read_valid_d3;
                    b_core_addr_d4 <= b_core_addr_d3;
                    b_read_valid_d3 <= b_read_valid_d2;
                    b_core_addr_d3 <= b_core_addr_d2;
                    b_read_valid_d2 <= b_read_valid_d1;
                    b_core_addr_d2 <= b_core_addr_d1;
                    b_read_valid_d1 <= 1'b0;
                    if (b_issue_count < ((chunk_K + 3) >> 2)) begin
                        weight_wide_rd_en <= 1'b1;
                        weight_wide_rd_addr <= b_stream_weight_addr;
                        b_read_valid_d1 <= 1'b1;
                        b_core_addr_d1 <=
                            b_col * MAX_TOTAL_K + (b_issue_count << 2);
                        b_issue_count <= b_issue_count + 1'b1;
                        b_stream_weight_addr <=
                            b_stream_weight_addr + 1'b1;
                    end
                    if (b_read_valid_d5) begin
                        core_B_wide_wr_en <= 1'b1;
                        core_B_wide_wr_addr <= b_core_addr_d5;
                        core_B_wide_wr_data <= weight_wide_rd_data;
                    end
                    if ((b_issue_count >= ((chunk_K + 3) >> 2)) &&
                        !b_read_valid_d1 && !b_read_valid_d2 &&
                        !b_read_valid_d3 && !b_read_valid_d4 &&
                        !b_read_valid_d5 &&
                        !core_B_wide_wr_en) begin
                        if (b_col + 1 >= tile_N)
                            state <= S_CORE_START;
                        else begin
                            b_col <= b_col + 1'b1;
                            b_k <= 0;
                            state <= S_B_REQ;
                        end
                    end
                end

                S_B_WAIT: state <= S_B_WRITE;

                S_B_WRITE: begin
                    core_B_wr_en <= 1;
                    core_B_wr_addr <= b_col * MAX_TOTAL_K + b_k;
                    core_B_wr_data <= weight_rd_data;
                    if (b_k + 1 >= chunk_K) begin
                        b_k <= 0;
                        if (b_col + 1 >= tile_N)
                            state <= S_CORE_START;
                        else begin
                            b_col <= b_col + 1'b1;
                            state <= S_B_REQ;
                        end
                    end else begin
                        b_k <= b_k + 1'b1;
                        state <= S_B_REQ;
                    end
                end

                S_CORE_START: begin
                    core_start <= 1;
                    state <= S_CORE_WAIT;
                end

                S_CORE_WAIT: begin
                    if (core_error) begin
                        o_error <= 1;
                        state <= S_DONE;
                    end else if (core_done) begin
                        // The common unchunked/non-residual case has no
                        // read-modify-write dependency. Drain its banked C
                        // scratchpad at one result per clock. Residual and
                        // multi-chunk layers retain the proven serial path.
                        if ((k_base == 0) && final_chunk) begin
                            if (layer_residual)
                                state <= S_RES_STREAM_INIT;
                            else
                                state <= S_C_STREAM_INIT;
                        end else begin
                            c_row <= 0;
                            c_col <= 0;
                            state <= S_C_REQ;
                        end
                    end
                end

                S_C_STREAM_INIT: begin
                    c_row <= 0;
                    c_col <= 0;
                    c_resp_row <= 0;
                    c_resp_col <= 0;
                    c_issue_count <= 0;
                    c_resp_count <= 0;
                    c_total_count <= tile_M * tile_N;
                    // Pipeline row-stride multiplication away from the
                    // streaming write address and the 300 MHz data path.
                    output_row_index <= m_base;
                    state <= S_C_STREAM_ADDR;
                end

                S_C_STREAM_ADDR: begin
                    c_resp_linear_addr <= output_row_index * layer_N + n_base;
                    state <= S_C_STREAM;
                end

                S_C_STREAM: begin
                    // Result banks accept one independent address every clock;
                    // responses are ordered, so only row/column counters are
                    // required instead of a wide address FIFO.
                    if (c_issue_count < c_total_count) begin
                        core_C_rd_en <= 1'b1;
                        core_C_rd_addr <= c_row * MAX_N + c_col;
                        c_issue_count <= c_issue_count + 1'b1;
                        if (c_col + 1 < tile_N)
                            c_col <= c_col + 1'b1;
                        else begin
                            c_col <= 0;
                            c_row <= c_row + 1'b1;
                        end
                    end

                    if (core_C_rd_valid) begin
                        processed_value = layer_relu6 ?
                            relu6_fp32(core_C_rd_data) : core_C_rd_data;
                        ctrl_activation_wr_en <= 1'b1;
                        ctrl_activation_wr_bank <= dst_bank;
                        ctrl_activation_wr_addr <= c_resp_linear_addr;
                        ctrl_activation_wr_data <= fp32_to_bf16(processed_value);

                        c_resp_count <= c_resp_count + 1'b1;
                        if (c_resp_col + 1 < tile_N) begin
                            c_resp_col <= c_resp_col + 1'b1;
                            c_resp_linear_addr <= c_resp_linear_addr + 1'b1;
                        end else begin
                            c_resp_col <= 0;
                            c_resp_row <= c_resp_row + 1'b1;
                            c_resp_linear_addr <= c_resp_linear_addr +
                                (layer_N - tile_N + 1'b1);
                        end

                        if (c_resp_count + 1'b1 >= c_total_count) begin
                            if ((layer_M > MAX_M) &&
                                (m_base + tile_M < layer_M)) begin
                                m_base <= m_base + tile_M;
                                advance_count <= tile_M;
                                reuse_B_tile <= 1'b1;
                                state <= S_ADVANCE_M;
                            end else if ((layer_M > MAX_M) &&
                                         (n_base + tile_N < layer_N)) begin
                                n_base <= n_base + tile_N;
                                m_base <= 0;
                                k_base <= 0;
                                tile_start_y <= 0;
                                tile_start_x <= 0;
                                reuse_B_tile <= 1'b0;
                                state <= S_PREP_TILE;
                            end else if (n_base + tile_N < layer_N) begin
                                n_base <= n_base + tile_N;
                                k_base <= 0;
                                reuse_B_tile <= 1'b0;
                                if (layer_kind != KIND_DEPTHWISE)
                                    state <= S_REUSE_A_NEXT_N;
                                else
                                    state <= S_PREP_TILE;
                            end else if (m_base + tile_M < layer_M) begin
                                m_base <= m_base + tile_M;
                                n_base <= 0;
                                k_base <= 0;
                                advance_count <= tile_M;
                                reuse_B_tile <= 1'b0;
                                state <= S_ADVANCE_M;
                            end else if (layer_idx == LAST_LAYER) begin
                                // Read the ten stored BF16 logits through the
                                // local argmax pipeline.  This removes the
                                // long core-result -> comparator -> CE path.
                                fast_argmax_index <= 0;
                                argmax_valid <= 0;
                                state <= S_FAST_ARGMAX_REQ;
                            end else begin
                                layer_idx <= layer_idx + 1'b1;
                                state <= S_SET_LAYER;
                            end
                        end
                    end
                end

                S_RES_STREAM_INIT: begin
                    res_issue_count <= 0;
                    res_total_count <= tile_M * tile_N;
                    res_issue_row <= 0;
                    res_issue_col <= 0;
                    c_resp_count <= 0;
                    c_resp_row <= 0;
                    c_resp_col <= 0;
                    res_add_valid_pipe <= 0;
                    res_add_last_pipe <= 0;
                    // Keep the row multiply out of the streaming state.
                    output_row_index <= m_base;
                    state <= S_RES_STREAM_ADDR;
                end

                S_RES_STREAM_ADDR: begin
                    res_issue_linear_addr <=
                        output_row_index * layer_N + n_base;
                    res_capture_linear_addr <=
                        output_row_index * layer_N + n_base;
                    state <= S_RES_STREAM;
                end

                S_RES_STREAM: begin
                    // C and skip responses remain ordered.  Issue one pair per
                    // clock, add through the six-stage FP32 pipeline, then
                    // round/store in the same order.
                    if (res_issue_count < res_total_count) begin
                        core_C_rd_en <= 1'b1;
                        core_C_rd_addr <=
                            res_issue_row * MAX_N + res_issue_col;
                        skip_rd_en <= 1'b1;
                        skip_rd_bank <= skip_bank;
                        skip_rd_addr <= res_issue_linear_addr;
                        res_issue_count <= res_issue_count + 1'b1;
                        if (res_issue_col + 1 < tile_N) begin
                            res_issue_col <= res_issue_col + 1'b1;
                            res_issue_linear_addr <=
                                res_issue_linear_addr + 1'b1;
                        end else begin
                            res_issue_col <= 0;
                            res_issue_row <= res_issue_row + 1'b1;
                            res_issue_linear_addr <= res_issue_linear_addr +
                                (layer_N - tile_N + 1'b1);
                        end
                    end

                    if (core_C_rd_valid) begin
                        add_a <= core_C_rd_data;
                        add_b <= {skip_data_d1, 16'd0};
                        res_add_valid_pipe[0] <= 1'b1;
                        res_add_last_pipe[0] <=
                            (c_resp_count + 1'b1 >= res_total_count);
                        res_addr_pipe[0] <= res_capture_linear_addr;
                        c_resp_count <= c_resp_count + 1'b1;
                        if (c_resp_col + 1 < tile_N) begin
                            c_resp_col <= c_resp_col + 1'b1;
                            res_capture_linear_addr <=
                                res_capture_linear_addr + 1'b1;
                        end else begin
                            c_resp_col <= 0;
                            c_resp_row <= c_resp_row + 1'b1;
                            res_capture_linear_addr <=
                                res_capture_linear_addr +
                                (layer_N - tile_N + 1'b1);
                        end
                    end

                    if (res_add_valid_pipe[RES_ADD_LATENCY-1]) begin
                        processed_value = layer_relu6 ?
                            relu6_fp32(add_out) : add_out;
                        ctrl_activation_wr_en <= 1'b1;
                        ctrl_activation_wr_bank <= dst_bank;
                        ctrl_activation_wr_addr <=
                            res_addr_pipe[RES_ADD_LATENCY-1];
                        ctrl_activation_wr_data <=
                            fp32_to_bf16(processed_value);
                        if (res_add_last_pipe[RES_ADD_LATENCY-1]) begin
                            // Reuse the proven tile/layer transition logic.
                            c_row <= tile_M - 1'b1;
                            c_col <= tile_N - 1'b1;
                            state <= S_C_ADVANCE;
                        end
                    end
                end

                S_C_REQ: begin
                    core_C_rd_en <= 1;
                    core_C_rd_addr <= c_row * MAX_N + c_col;
                    output_row_index <= m_base + c_row;
                    state <= S_C_WAIT;
                end

                S_C_WAIT: begin
                    output_row_base <= output_row_index * layer_N;
                    state <= S_C_CAPTURE;
                end

                S_C_CAPTURE: begin
                    result_linear_addr <= output_row_base + n_base + c_col;
                    if (core_C_rd_valid) begin
                        if (k_base == 0) begin
                            if (final_chunk) begin
                                result_value <= core_C_rd_data;
                                state <= S_RESULT;
                            end else begin
                                partial_mem[partial_addr] <= core_C_rd_data;
                                state <= S_C_ADVANCE;
                            end
                        end else begin
                            add_a <= partial_mem[partial_addr];
                            add_b <= core_C_rd_data;
                            add_wait <= 0;
                            state <= S_PARTIAL_WAIT;
                        end
                    end
                end

                S_PARTIAL_WAIT: begin
                    if (add_wait == 6) begin
                        if (final_chunk) begin
                            result_value <= add_out;
                            state <= S_RESULT;
                        end else begin
                            partial_mem[partial_addr] <= add_out;
                            state <= S_C_ADVANCE;
                        end
                    end else
                        add_wait <= add_wait + 1'b1;
                end

                S_RESULT: begin
                    if (layer_residual) begin
                        skip_rd_en <= 1;
                        skip_rd_bank <= skip_bank;
                        skip_rd_addr <= result_linear_addr;
                        state <= S_SKIP_WAIT;
                    end else
                        state <= S_STORE_RESULT;
                end

                S_SKIP_WAIT: state <= S_SKIP_LAUNCH;

                S_SKIP_LAUNCH: begin
                    add_a <= result_value;
                    add_b <= {skip_rd_data, 16'd0};
                    add_wait <= 0;
                    state <= S_RESIDUAL_WAIT;
                end

                S_RESIDUAL_WAIT: begin
                    if (add_wait == 6) begin
                        result_value <= add_out;
                        state <= S_STORE_RESULT;
                    end else
                        add_wait <= add_wait + 1'b1;
                end

                S_STORE_RESULT: begin
                    processed_value = layer_relu6 ? relu6_fp32(result_value) :
                                                    result_value;
                    ctrl_activation_wr_en <= 1;
                    ctrl_activation_wr_bank <= dst_bank;
                    ctrl_activation_wr_addr <= result_linear_addr;
                    ctrl_activation_wr_data <= fp32_to_bf16(processed_value);
                    state <= S_C_ADVANCE;
                end

                S_C_ADVANCE: begin
                    if (c_col + 1 < tile_N) begin
                        c_col <= c_col + 1'b1;
                        state <= S_C_REQ;
                    end else if (c_row + 1 < tile_M) begin
                        c_col <= 0;
                        c_row <= c_row + 1'b1;
                        state <= S_C_REQ;
                    end else if (!final_chunk) begin
                        k_base <= k_base + chunk_K;
                        reuse_B_tile <= 1'b0;
                        state <= S_PREP_TILE;
                    end else if ((layer_K <= MAX_TOTAL_K) &&
                                 (layer_M > MAX_M) &&
                                 (m_base + tile_M < layer_M)) begin
                        // For unchunked Conv/depthwise tiles, B is unchanged
                        // across the complete spatial sweep.  Hold it in the
                        // core and reload only A for the next M tile.
                        m_base <= m_base + tile_M;
                        advance_count <= tile_M;
                        reuse_B_tile <= 1'b1;
                        state <= S_ADVANCE_M;
                    end else if ((layer_K <= MAX_TOTAL_K) &&
                                 (layer_M > MAX_M) &&
                                 (n_base + tile_N < layer_N)) begin
                        // New output-channel group: restart the spatial sweep
                        // and replace the resident B tile once.
                        n_base <= n_base + tile_N;
                        m_base <= 0;
                        k_base <= 0;
                        tile_start_y <= 0;
                        tile_start_x <= 0;
                        reuse_B_tile <= 1'b0;
                        state <= S_PREP_TILE;
                    end else if (n_base + tile_N < layer_N) begin
                        n_base <= n_base + tile_N;
                        k_base <= 0;
                        reuse_B_tile <= 1'b0;
                        // A is independent of the output-channel tile for
                        // Conv/Linear layers.  Keep it resident in the core
                        // and stream only the next B tile.  Chunked and
                        // depthwise layers still take the conservative path.
                        if ((layer_kind != KIND_DEPTHWISE) && (k_base == 0))
                            state <= S_REUSE_A_NEXT_N;
                        else
                            state <= S_PREP_TILE;
                    end else if (m_base + tile_M < layer_M) begin
                        m_base <= m_base + tile_M;
                        n_base <= 0;
                        k_base <= 0;
                        advance_count <= tile_M;
                        reuse_B_tile <= 1'b0;
                        state <= S_ADVANCE_M;
                    end else if (layer_idx == LAST_LAYER) begin
                        fast_argmax_index <= 0;
                        argmax_valid <= 0;
                        state <= S_FAST_ARGMAX_REQ;
                    end else begin
                        layer_idx <= layer_idx + 1'b1;
                        state <= S_SET_LAYER;
                    end
                end

                S_ADVANCE_M: begin
                    if (tile_start_x + 1 >= layer_out_w) begin
                        tile_start_x <= 0;
                        tile_start_y <= tile_start_y + 1'b1;
                    end else
                        tile_start_x <= tile_start_x + 1'b1;
                    if (advance_count <= 1)
                        state <= S_PREP_TILE;
                    else
                        advance_count <= advance_count - 1'b1;
                end

                S_REUSE_A_NEXT_N: begin
                    tile_N <= ((layer_N - n_base) > MAX_N) ? MAX_N :
                              (layer_N - n_base);
                    b_col <= 0;
                    b_k <= 0;
                    state <= S_B_REQ;
                end

                S_DONE: begin
                    o_busy <= 0;
                    o_done <= 1;
                    if (!i_start)
                        state <= S_IDLE;
                end

                default: begin
                    o_error <= 1;
                    state <= S_DONE;
                end
            endcase
        end
    end
endmodule
