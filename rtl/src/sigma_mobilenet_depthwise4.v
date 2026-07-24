`timescale 1ns / 1ps
// MobileNet depthwise and M=1 pointwise fast path.
//
// The generic SIGMA GEMM path is excellent for shared-A Conv/Linear tiles but
// a depthwise layer has a different A vector for every output channel.  Mapping
// it as N=1 repeatedly leaves fifteen DPEs idle and pays the complete mesh
// launch/drain cost for every channel.  This controller keeps the SIGMA BF16
// multiplier/FAN arithmetic, caches one channel's ten folded weights, and
// evaluates four spatial outputs together.  For M=1 pointwise layers it
// caches A once and evaluates four output channels together.  Learned weights
// and activations remain in the same on-chip ROM/three-bank scratchpad.
module sigma_mobilenet_depthwise4 (
    input  wire         clk,
    input  wire         rst,
    input  wire         i_start,
    input  wire         i_mode_m1,
    input  wire [15:0]  i_M,
    input  wire [11:0]  i_N,
    input  wire [11:0]  i_K,
    input  wire [11:0]  i_channels,
    input  wire [5:0]   i_in_h,
    input  wire [5:0]   i_in_w,
    input  wire [5:0]   i_out_h,
    input  wire [5:0]   i_out_w,
    input  wire [1:0]   i_stride,
    input  wire [1:0]   i_pad,
    input  wire [18:0]  i_weight_offset,
    input  wire [1:0]   i_src_bank,
    input  wire [1:0]   i_dst_bank,
    input  wire         i_relu6,

    output reg          o_weight_rd_en,
    output reg  [17:0]  o_weight_rd_addr,
    input  wire [15:0]  i_weight_rd_data,
    output reg          o_activation_rd_en,
    output reg  [1:0]   o_activation_rd_bank,
    output reg  [13:0]  o_activation_rd_addr,
    input  wire [15:0]  i_activation_rd_data,
    output reg          o_activation_wr_en,
    output reg  [1:0]   o_activation_wr_bank,
    output reg  [13:0]  o_activation_wr_addr,
    output reg  [15:0]  o_activation_wr_data,

    output reg          o_busy,
    output reg          o_done,
    output reg          o_error,
    output reg  [31:0]  o_cycles
);
    localparam LANES = 4;
    localparam K_WORDS = 10;

    localparam ST_IDLE = 5'd0;
    localparam ST_CHANNEL_ADDR = 5'd1;
    localparam ST_WEIGHT_STREAM = 5'd2;
    localparam ST_GROUP_PREP = 5'd3;
    localparam ST_GROUP_COORD = 5'd4;
    localparam ST_ACT_STREAM = 5'd5;
    localparam ST_DOT_START = 5'd6;
    localparam ST_DOT_FEED = 5'd7;
    localparam ST_DOT_WAIT = 5'd8;
    localparam ST_STORE = 5'd9;
    localparam ST_ADVANCE = 5'd10;
    localparam ST_DONE = 5'd11;
    localparam ST_M1_A_PREP = 5'd12;
    localparam ST_M1_A_STREAM = 5'd13;
    localparam ST_M1_GROUP = 5'd14;
    localparam ST_M1_WEIGHT_STREAM = 5'd15;
    localparam ST_M1_DOT_START = 5'd16;
    localparam ST_M1_DOT_FEED = 5'd17;
    localparam ST_M1_DOT_WAIT = 5'd18;
    localparam ST_M1_STORE = 5'd19;
    localparam ST_M1_ADVANCE = 5'd20;

    reg [4:0] state;
    reg r_mode_m1;
    reg [15:0] r_M;
    reg [11:0] r_N, r_K;
    reg [11:0] r_channels;
    reg [5:0] r_in_h, r_in_w, r_out_h, r_out_w;
    reg [1:0] r_stride, r_pad, r_src_bank, r_dst_bank;
    reg [18:0] r_weight_offset;
    reg r_relu6;

    reg [11:0] channel;
    reg [15:0] pixel_base;
    reg [5:0] pixel_cursor_y, pixel_cursor_x;
    reg [2:0] lane_count;
    reg [2:0] coord_lane;
    reg [5:0] lane_y [0:LANES-1];
    reg [5:0] lane_x [0:LANES-1];
    reg [27:0] group_output_base;

    reg [18:0] channel_weight_base;
    reg [3:0] weight_issue_count;
    reg weight_valid_d1, weight_valid_d2;
    reg [3:0] weight_index_d1, weight_index_d2;
    // A depthwise dot needs four consecutive weights per cycle.  Describing
    // this as one array gave the inferred LUTRAM four asynchronous read ports;
    // Vivado replicated RAMD32 cells and its write clock/data became the
    // post-route critical path.  Ten scalar registers plus a 4-word fold mux
    // are the exact hardware required by this fixed 3x3+bias kernel.
    reg [15:0] dw_weight_0, dw_weight_1, dw_weight_2, dw_weight_3;
    reg [15:0] dw_weight_4, dw_weight_5, dw_weight_6, dw_weight_7;
    reg [15:0] dw_weight_8, dw_weight_9;
    reg [63:0] dw_weight_fold_bus;

    // M=1 pointwise/linear cache.  The old flat arrays required four parallel
    // A reads and sixteen parallel weight reads from the same inferred memory.
    // Vivado implemented that as a heavily replicated multi-port LUTRAM whose
    // BRAM-to-cache write data drove thousands of endpoints.  Bank by PE (and
    // by output lane for B) instead: every bank has one asynchronous read port
    // at dot_fold and one synchronous write port.  Capacity and arithmetic are
    // unchanged, but the physical memory now matches the 4x4 datapath exactly.
    (* ram_style = "distributed" *) reg [15:0] m1_a_p0 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_a_p1 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_a_p2 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_a_p3 [0:63];

    (* ram_style = "distributed" *) reg [15:0] m1_w_l0_p0 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l0_p1 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l0_p2 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l0_p3 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l1_p0 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l1_p1 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l1_p2 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l1_p3 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l2_p0 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l2_p1 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l2_p2 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l2_p3 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l3_p0 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l3_p1 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l3_p2 [0:63];
    (* ram_style = "distributed" *) reg [15:0] m1_w_l3_p3 [0:63];
    reg [11:0] m1_output_base;
    reg [11:0] m1_a_issue_count;
    reg m1_a_valid_d1, m1_a_valid_d2;
    reg [7:0] m1_a_index_d1, m1_a_index_d2;
    reg [10:0] m1_weight_issue_count;
    reg [11:0] m1_weight_total;
    reg [1:0] m1_weight_lane;
    reg [11:0] m1_weight_k;
    reg [17:0] m1_weight_addr_current;
    reg m1_weight_valid_d1, m1_weight_valid_d2;
    reg [9:0] m1_weight_index_d1, m1_weight_index_d2;

    wire [63:0] m1_a_fold_bus;
    wire [255:0] m1_weight_fold_bus;

    reg [6:0] act_issue_count;
    reg [2:0] act_issue_lane;
    reg [3:0] act_issue_k;
    reg [15:0] act_cache [0:LANES*K_WORDS-1];

    reg s1_valid;
    reg [2:0] s1_lane;
    reg [3:0] s1_k;
    reg signed [7:0] s1_y, s1_x;
    reg s1_bias;

    reg s2_valid, s2_direct;
    reg [2:0] s2_lane;
    reg [3:0] s2_k;
    reg [15:0] s2_direct_data;
    reg [11:0] s2_pixel;

    reg s3_valid, s3_direct;
    reg [2:0] s3_lane;
    reg [3:0] s3_k;
    reg [15:0] s3_direct_data;
    reg [25:0] s3_address;

    reg s4_valid, s4_direct;
    reg [2:0] s4_lane;
    reg [3:0] s4_k;
    reg [15:0] s4_direct_data;
    reg s5_valid, s5_direct;
    reg [2:0] s5_lane;
    reg [3:0] s5_k;
    reg [15:0] s5_direct_data;
    reg [15:0] activation_data_d1;

    reg dot_start, dot_input_valid;
    reg [6:0] dot_fold;
    reg [255:0] dot_a_bus, dot_b_bus;
    wire dot_busy, dot_result_valid;
    wire [127:0] dot_result_bus;
    wire [6:0] dot_expected_folds = r_mode_m1 ?
        ((r_K + 3) >> 2) : 7'd3;
    reg [127:0] result_latched;
    reg [2:0] store_lane;

    assign m1_a_fold_bus[ 0 +: 16] = m1_a_p0[dot_fold[5:0]];
    assign m1_a_fold_bus[16 +: 16] = m1_a_p1[dot_fold[5:0]];
    assign m1_a_fold_bus[32 +: 16] = m1_a_p2[dot_fold[5:0]];
    assign m1_a_fold_bus[48 +: 16] = m1_a_p3[dot_fold[5:0]];

    assign m1_weight_fold_bus[  0 +: 16] = m1_w_l0_p0[dot_fold[5:0]];
    assign m1_weight_fold_bus[ 16 +: 16] = m1_w_l0_p1[dot_fold[5:0]];
    assign m1_weight_fold_bus[ 32 +: 16] = m1_w_l0_p2[dot_fold[5:0]];
    assign m1_weight_fold_bus[ 48 +: 16] = m1_w_l0_p3[dot_fold[5:0]];
    assign m1_weight_fold_bus[ 64 +: 16] = m1_w_l1_p0[dot_fold[5:0]];
    assign m1_weight_fold_bus[ 80 +: 16] = m1_w_l1_p1[dot_fold[5:0]];
    assign m1_weight_fold_bus[ 96 +: 16] = m1_w_l1_p2[dot_fold[5:0]];
    assign m1_weight_fold_bus[112 +: 16] = m1_w_l1_p3[dot_fold[5:0]];
    assign m1_weight_fold_bus[128 +: 16] = m1_w_l2_p0[dot_fold[5:0]];
    assign m1_weight_fold_bus[144 +: 16] = m1_w_l2_p1[dot_fold[5:0]];
    assign m1_weight_fold_bus[160 +: 16] = m1_w_l2_p2[dot_fold[5:0]];
    assign m1_weight_fold_bus[176 +: 16] = m1_w_l2_p3[dot_fold[5:0]];
    assign m1_weight_fold_bus[192 +: 16] = m1_w_l3_p0[dot_fold[5:0]];
    assign m1_weight_fold_bus[208 +: 16] = m1_w_l3_p1[dot_fold[5:0]];
    assign m1_weight_fold_bus[224 +: 16] = m1_w_l3_p2[dot_fold[5:0]];
    assign m1_weight_fold_bus[240 +: 16] = m1_w_l3_p3[dot_fold[5:0]];

    always @* begin
        case (dot_fold[1:0])
            2'd0: dw_weight_fold_bus = {
                dw_weight_3, dw_weight_2, dw_weight_1, dw_weight_0};
            2'd1: dw_weight_fold_bus = {
                dw_weight_7, dw_weight_6, dw_weight_5, dw_weight_4};
            default: dw_weight_fold_bus = {
                16'h0000, 16'h0000, dw_weight_9, dw_weight_8};
        endcase
    end

    integer build_lane, build_pe;
    integer operand_index;

    sigma_mobilenet_dot4 u_dot4 (
        .clk(clk), .rst(rst), .i_start(dot_start),
        .i_fold_count(dot_expected_folds),
        .i_valid(dot_input_valid),
        .i_a_bus(dot_a_bus), .i_b_bus(dot_b_bus),
        .o_busy(dot_busy), .o_valid(dot_result_valid),
        .o_result_bus(dot_result_bus)
    );

    function [1:0] kernel_y;
        input [3:0] index;
        begin
            if (index < 3) kernel_y = 0;
            else if (index < 6) kernel_y = 1;
            else kernel_y = 2;
        end
    endfunction

    function [1:0] kernel_x;
        input [3:0] index;
        begin
            case (index)
                0, 3, 6: kernel_x = 0;
                1, 4, 7: kernel_x = 1;
                default: kernel_x = 2;
            endcase
        end
    endfunction

    function [31:0] relu6_fp32;
        input [31:0] value;
        begin
            if (value[31])
                relu6_fp32 = 0;
            else if (value > 32'h40c00000)
                relu6_fp32 = 32'h40c00000;
            else
                relu6_fp32 = value;
        end
    endfunction

    function [15:0] fp32_to_bf16;
        input [31:0] value;
        reg [31:0] rounded;
        begin
            rounded = value + 32'h00007fff + value[16];
            fp32_to_bf16 = rounded[31:16];
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            o_busy <= 0;
            o_done <= 0;
            o_error <= 0;
            o_cycles <= 0;
            o_weight_rd_en <= 0;
            o_activation_rd_en <= 0;
            o_activation_wr_en <= 0;
            dot_start <= 0;
            dot_input_valid <= 0;
            s1_valid <= 0;
            s2_valid <= 0;
            s3_valid <= 0;
            s4_valid <= 0;
            s5_valid <= 0;
            weight_valid_d1 <= 0;
            weight_valid_d2 <= 0;
            m1_a_valid_d1 <= 0;
            m1_a_valid_d2 <= 0;
            m1_weight_valid_d1 <= 0;
            m1_weight_valid_d2 <= 0;
        end else begin
            // The inferred UltraRAM updates its registered output on the same
            // edge that advances our address-tag pipeline.  Hold one copy so
            // a tag is paired with its own request instead of the following
            // request when reads are issued on consecutive clocks.
            activation_data_d1 <= i_activation_rd_data;
            o_done <= 0;
            o_weight_rd_en <= 0;
            o_activation_rd_en <= 0;
            o_activation_wr_en <= 0;
            dot_start <= 0;
            dot_input_valid <= 0;
            if (o_busy)
                o_cycles <= o_cycles + 1'b1;

            case (state)
                ST_IDLE: begin
                    o_busy <= 0;
                    if (i_start) begin
                        r_mode_m1 <= i_mode_m1;
                        r_M <= i_M;
                        r_N <= i_N;
                        r_K <= i_K;
                        r_channels <= i_channels;
                        r_in_h <= i_in_h;
                        r_in_w <= i_in_w;
                        r_out_h <= i_out_h;
                        r_out_w <= i_out_w;
                        r_stride <= i_stride;
                        r_pad <= i_pad;
                        r_weight_offset <= i_weight_offset;
                        r_src_bank <= i_src_bank;
                        r_dst_bank <= i_dst_bank;
                        r_relu6 <= i_relu6;
                        channel <= 0;
                        pixel_base <= 0;
                        pixel_cursor_y <= 0;
                        pixel_cursor_x <= 0;
                        o_busy <= 1;
                        o_error <= i_mode_m1 ?
                                   ((i_M != 1) || (i_N == 0) ||
                                    (i_K == 0) || (i_K > 256)) :
                                   ((i_M == 0) || (i_channels == 0) ||
                                    (i_out_h == 0) || (i_out_w == 0));
                        o_cycles <= 0;
                        if (i_mode_m1)
                            state <= ((i_M != 1) || (i_N == 0) ||
                                      (i_K == 0) || (i_K > 256)) ?
                                     ST_DONE : ST_M1_A_PREP;
                        else
                            state <= ((i_M == 0) || (i_channels == 0) ||
                                      (i_out_h == 0) || (i_out_w == 0)) ?
                                     ST_DONE : ST_CHANNEL_ADDR;
                    end
                end

                ST_CHANNEL_ADDR: begin
                    channel_weight_base <= r_weight_offset + channel * K_WORDS;
                    weight_issue_count <= 0;
                    weight_valid_d1 <= 0;
                    weight_valid_d2 <= 0;
                    state <= ST_WEIGHT_STREAM;
                end

                ST_WEIGHT_STREAM: begin
                    weight_valid_d2 <= weight_valid_d1;
                    weight_index_d2 <= weight_index_d1;
                    weight_valid_d1 <= 0;
                    if (weight_issue_count < K_WORDS) begin
                        o_weight_rd_en <= 1;
                        o_weight_rd_addr <= channel_weight_base + weight_issue_count;
                        weight_valid_d1 <= 1;
                        weight_index_d1 <= weight_issue_count;
                        weight_issue_count <= weight_issue_count + 1'b1;
                    end
                    if (weight_valid_d2) begin
                        case (weight_index_d2)
                            4'd0: dw_weight_0 <= i_weight_rd_data;
                            4'd1: dw_weight_1 <= i_weight_rd_data;
                            4'd2: dw_weight_2 <= i_weight_rd_data;
                            4'd3: dw_weight_3 <= i_weight_rd_data;
                            4'd4: dw_weight_4 <= i_weight_rd_data;
                            4'd5: dw_weight_5 <= i_weight_rd_data;
                            4'd6: dw_weight_6 <= i_weight_rd_data;
                            4'd7: dw_weight_7 <= i_weight_rd_data;
                            4'd8: dw_weight_8 <= i_weight_rd_data;
                            default: dw_weight_9 <= i_weight_rd_data;
                        endcase
                    end
                    if ((weight_issue_count >= K_WORDS) &&
                        !weight_valid_d1 && !weight_valid_d2) begin
                        pixel_base <= 0;
                        pixel_cursor_y <= 0;
                        pixel_cursor_x <= 0;
                        state <= ST_GROUP_PREP;
                    end
                end

                ST_GROUP_PREP: begin
                    if ((r_M - pixel_base) >= LANES)
                        lane_count <= LANES;
                    else
                        lane_count <= r_M - pixel_base;
                    group_output_base <= pixel_base * r_channels + channel;
                    coord_lane <= 0;
                    state <= ST_GROUP_COORD;
                end

                ST_GROUP_COORD: begin
                    lane_y[coord_lane] <= pixel_cursor_y;
                    lane_x[coord_lane] <= pixel_cursor_x;
                    if (pixel_cursor_x + 1 >= r_out_w) begin
                        pixel_cursor_x <= 0;
                        pixel_cursor_y <= pixel_cursor_y + 1'b1;
                    end else
                        pixel_cursor_x <= pixel_cursor_x + 1'b1;

                    if (coord_lane + 1 >= lane_count) begin
                        act_issue_count <= 0;
                        act_issue_lane <= 0;
                        act_issue_k <= 0;
                        s1_valid <= 0;
                        s2_valid <= 0;
                        s3_valid <= 0;
                        s4_valid <= 0;
                        s5_valid <= 0;
                        state <= ST_ACT_STREAM;
                    end else
                        coord_lane <= coord_lane + 1'b1;
                end

                ST_ACT_STREAM: begin
                    // Capture the oldest response before advancing the fully
                    // registered coordinate/address/read pipeline.
                    if (s5_valid)
                        act_cache[s5_lane*K_WORDS + s5_k] <=
                            s5_direct ? s5_direct_data : activation_data_d1;

                    s5_valid <= s4_valid;
                    s5_direct <= s4_direct;
                    s5_lane <= s4_lane;
                    s5_k <= s4_k;
                    s5_direct_data <= s4_direct_data;

                    s4_valid <= s3_valid;
                    s4_direct <= s3_direct;
                    s4_lane <= s3_lane;
                    s4_k <= s3_k;
                    s4_direct_data <= s3_direct_data;

                    s3_valid <= s2_valid;
                    s3_direct <= s2_direct;
                    s3_lane <= s2_lane;
                    s3_k <= s2_k;
                    s3_direct_data <= s2_direct_data;
                    if (s2_valid && !s2_direct) begin
                        s3_address <= s2_pixel * r_channels + channel;
                        o_activation_rd_en <= 1;
                        o_activation_rd_bank <= r_src_bank;
                        o_activation_rd_addr <= s2_pixel * r_channels + channel;
                    end

                    s2_valid <= s1_valid;
                    s2_lane <= s1_lane;
                    s2_k <= s1_k;
                    if (s1_valid) begin
                        if (s1_bias) begin
                            s2_direct <= 1;
                            s2_direct_data <= 16'h3f80;
                        end else if ((s1_y < 0) || (s1_x < 0) ||
                                     (s1_y >= $signed({1'b0, r_in_h})) ||
                                     (s1_x >= $signed({1'b0, r_in_w}))) begin
                            s2_direct <= 1;
                            s2_direct_data <= 0;
                        end else begin
                            s2_direct <= 0;
                            s2_direct_data <= 0;
                            s2_pixel <= s1_y * $signed({1'b0, r_in_w}) + s1_x;
                        end
                    end

                    s1_valid <= 0;
                    if (act_issue_count < lane_count*K_WORDS) begin
                        s1_valid <= 1;
                        s1_lane <= act_issue_lane;
                        s1_k <= act_issue_k;
                        s1_bias <= (act_issue_k == K_WORDS-1);
                        s1_y <= $signed({1'b0, lane_y[act_issue_lane]}) *
                                $signed({1'b0, r_stride}) +
                                $signed({1'b0, kernel_y(act_issue_k)}) -
                                $signed({1'b0, r_pad});
                        s1_x <= $signed({1'b0, lane_x[act_issue_lane]}) *
                                $signed({1'b0, r_stride}) +
                                $signed({1'b0, kernel_x(act_issue_k)}) -
                                $signed({1'b0, r_pad});
                        act_issue_count <= act_issue_count + 1'b1;
                        if (act_issue_k + 1 >= K_WORDS) begin
                            act_issue_k <= 0;
                            act_issue_lane <= act_issue_lane + 1'b1;
                        end else
                            act_issue_k <= act_issue_k + 1'b1;
                    end

                    if ((act_issue_count >= lane_count*K_WORDS) &&
                        !s1_valid && !s2_valid && !s3_valid &&
                        !s4_valid && !s5_valid)
                        state <= ST_DOT_START;
                end

                ST_DOT_START: begin
                    dot_start <= 1;
                    dot_fold <= 0;
                    state <= ST_DOT_FEED;
                end

                ST_DOT_FEED: begin
                    dot_input_valid <= 1;
                    for (build_lane = 0; build_lane < LANES;
                         build_lane = build_lane + 1) begin
                        for (build_pe = 0; build_pe < 4;
                             build_pe = build_pe + 1) begin
                            operand_index = dot_fold*4 + build_pe;
                            if ((build_lane < lane_count) &&
                                (operand_index < K_WORDS)) begin
                                dot_a_bus[(build_lane*4+build_pe)*16 +: 16] <=
                                    act_cache[build_lane*K_WORDS + operand_index];
                                dot_b_bus[(build_lane*4+build_pe)*16 +: 16] <=
                                    dw_weight_fold_bus[build_pe*16 +: 16];
                            end else begin
                                dot_a_bus[(build_lane*4+build_pe)*16 +: 16] <= 0;
                                dot_b_bus[(build_lane*4+build_pe)*16 +: 16] <= 0;
                            end
                        end
                    end
                    if (dot_fold == 2)
                        state <= ST_DOT_WAIT;
                    else
                        dot_fold <= dot_fold + 1'b1;
                end

                ST_DOT_WAIT: begin
                    if (dot_result_valid) begin
                        result_latched <= dot_result_bus;
                        store_lane <= 0;
                        state <= ST_STORE;
                    end
                end

                ST_STORE: begin
                    o_activation_wr_en <= 1;
                    o_activation_wr_bank <= r_dst_bank;
                    o_activation_wr_addr <= group_output_base +
                                            store_lane*r_channels;
                    o_activation_wr_data <= fp32_to_bf16(
                        r_relu6 ? relu6_fp32(
                            result_latched[store_lane*32 +: 32]) :
                            result_latched[store_lane*32 +: 32]);
                    if (store_lane + 1 >= lane_count)
                        state <= ST_ADVANCE;
                    else
                        store_lane <= store_lane + 1'b1;
                end

                ST_ADVANCE: begin
                    if (pixel_base + lane_count >= r_M) begin
                        if (channel + 1 >= r_channels)
                            state <= ST_DONE;
                        else begin
                            channel <= channel + 1'b1;
                            pixel_base <= 0;
                            pixel_cursor_y <= 0;
                            pixel_cursor_x <= 0;
                            state <= ST_CHANNEL_ADDR;
                        end
                    end else begin
                        pixel_base <= pixel_base + lane_count;
                        state <= ST_GROUP_PREP;
                    end
                end

                // Pointwise fast path for M=1, K<=256.  The input vector is
                // loaded once and reused for every four-output group.  Model
                // ROM bandwidth is then the limiting resource instead of the
                // generic mesh's repeated launch/drain sequence.
                ST_M1_A_PREP: begin
                    case ((r_K - 1'b1) & 12'h003)
                        0: m1_a_p0[(r_K - 1'b1) >> 2] <= 16'h3f80;
                        1: m1_a_p1[(r_K - 1'b1) >> 2] <= 16'h3f80;
                        2: m1_a_p2[(r_K - 1'b1) >> 2] <= 16'h3f80;
                        default: m1_a_p3[(r_K - 1'b1) >> 2] <= 16'h3f80;
                    endcase
                    m1_a_issue_count <= 0;
                    m1_a_valid_d1 <= 0;
                    m1_a_valid_d2 <= 0;
                    m1_output_base <= 0;
                    state <= ST_M1_A_STREAM;
                end

                ST_M1_A_STREAM: begin
                    m1_a_valid_d2 <= m1_a_valid_d1;
                    m1_a_index_d2 <= m1_a_index_d1;
                    m1_a_valid_d1 <= 0;
                    if (m1_a_issue_count + 1 < r_K) begin
                        o_activation_rd_en <= 1;
                        o_activation_rd_bank <= r_src_bank;
                        o_activation_rd_addr <= {2'b00, m1_a_issue_count};
                        m1_a_valid_d1 <= 1;
                        m1_a_index_d1 <= m1_a_issue_count[7:0];
                        m1_a_issue_count <= m1_a_issue_count + 1'b1;
                    end
                    if (m1_a_valid_d2) begin
                        case (m1_a_index_d2[1:0])
                            0: m1_a_p0[m1_a_index_d2[7:2]] <=
                                   i_activation_rd_data;
                            1: m1_a_p1[m1_a_index_d2[7:2]] <=
                                   i_activation_rd_data;
                            2: m1_a_p2[m1_a_index_d2[7:2]] <=
                                   i_activation_rd_data;
                            default: m1_a_p3[m1_a_index_d2[7:2]] <=
                                         i_activation_rd_data;
                        endcase
                    end
                    if ((m1_a_issue_count + 1 >= r_K) &&
                        !m1_a_valid_d1 && !m1_a_valid_d2)
                        state <= ST_M1_GROUP;
                end

                ST_M1_GROUP: begin
                    if ((r_N - m1_output_base) >= LANES) begin
                        lane_count <= LANES;
                        m1_weight_total <= LANES * r_K;
                    end else begin
                        lane_count <= r_N - m1_output_base;
                        m1_weight_total <= (r_N - m1_output_base) * r_K;
                    end
                    m1_weight_issue_count <= 0;
                    m1_weight_lane <= 0;
                    m1_weight_k <= 0;
                    m1_weight_addr_current <= r_weight_offset + m1_output_base;
                    m1_weight_valid_d1 <= 0;
                    m1_weight_valid_d2 <= 0;
                    state <= ST_M1_WEIGHT_STREAM;
                end

                ST_M1_WEIGHT_STREAM: begin
                    m1_weight_valid_d2 <= m1_weight_valid_d1;
                    m1_weight_index_d2 <= m1_weight_index_d1;
                    m1_weight_valid_d1 <= 0;
                    if (m1_weight_issue_count < m1_weight_total) begin
                        o_weight_rd_en <= 1;
                        o_weight_rd_addr <= m1_weight_addr_current;
                        m1_weight_valid_d1 <= 1;
                        m1_weight_index_d1 <=
                            {m1_weight_lane, 8'd0} + m1_weight_k[7:0];
                        m1_weight_issue_count <=
                            m1_weight_issue_count + 1'b1;
                        if (m1_weight_k + 1 >= r_K) begin
                            m1_weight_k <= 0;
                            m1_weight_lane <= m1_weight_lane + 1'b1;
                            m1_weight_addr_current <= r_weight_offset +
                                m1_output_base + m1_weight_lane + 1'b1;
                        end else begin
                            m1_weight_k <= m1_weight_k + 1'b1;
                            m1_weight_addr_current <=
                                m1_weight_addr_current + r_N;
                        end
                    end
                    if (m1_weight_valid_d2) begin
                        case ({m1_weight_index_d2[9:8],
                               m1_weight_index_d2[1:0]})
                            4'h0: m1_w_l0_p0[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h1: m1_w_l0_p1[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h2: m1_w_l0_p2[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h3: m1_w_l0_p3[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h4: m1_w_l1_p0[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h5: m1_w_l1_p1[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h6: m1_w_l1_p2[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h7: m1_w_l1_p3[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h8: m1_w_l2_p0[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'h9: m1_w_l2_p1[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'ha: m1_w_l2_p2[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'hb: m1_w_l2_p3[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'hc: m1_w_l3_p0[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'hd: m1_w_l3_p1[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            4'he: m1_w_l3_p2[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                            default: m1_w_l3_p3[m1_weight_index_d2[7:2]] <= i_weight_rd_data;
                        endcase
                    end
                    if ((m1_weight_issue_count >= m1_weight_total) &&
                        !m1_weight_valid_d1 && !m1_weight_valid_d2)
                        state <= ST_M1_DOT_START;
                end

                ST_M1_DOT_START: begin
                    dot_start <= 1;
                    dot_fold <= 0;
                    state <= ST_M1_DOT_FEED;
                end

                ST_M1_DOT_FEED: begin
                    dot_input_valid <= 1;
                    for (build_lane = 0; build_lane < LANES;
                         build_lane = build_lane + 1) begin
                        for (build_pe = 0; build_pe < 4;
                             build_pe = build_pe + 1) begin
                            operand_index = dot_fold*4 + build_pe;
                            if ((build_lane < lane_count) &&
                                (operand_index < r_K)) begin
                                dot_a_bus[(build_lane*4+build_pe)*16 +: 16] <=
                                    m1_a_fold_bus[build_pe*16 +: 16];
                                dot_b_bus[(build_lane*4+build_pe)*16 +: 16] <=
                                    m1_weight_fold_bus[
                                        (build_lane*4+build_pe)*16 +: 16];
                            end else begin
                                dot_a_bus[(build_lane*4+build_pe)*16 +: 16] <= 0;
                                dot_b_bus[(build_lane*4+build_pe)*16 +: 16] <= 0;
                            end
                        end
                    end
                    if (dot_fold*4 + 4 >= r_K)
                        state <= ST_M1_DOT_WAIT;
                    else
                        dot_fold <= dot_fold + 1'b1;
                end

                ST_M1_DOT_WAIT: begin
                    if (dot_result_valid) begin
                        result_latched <= dot_result_bus;
                        store_lane <= 0;
                        state <= ST_M1_STORE;
                    end
                end

                ST_M1_STORE: begin
                    o_activation_wr_en <= 1;
                    o_activation_wr_bank <= r_dst_bank;
                    o_activation_wr_addr <= m1_output_base + store_lane;
                    o_activation_wr_data <= fp32_to_bf16(
                        r_relu6 ? relu6_fp32(
                            result_latched[store_lane*32 +: 32]) :
                            result_latched[store_lane*32 +: 32]);
                    if (store_lane + 1 >= lane_count)
                        state <= ST_M1_ADVANCE;
                    else
                        store_lane <= store_lane + 1'b1;
                end

                ST_M1_ADVANCE: begin
                    if (m1_output_base + lane_count >= r_N)
                        state <= ST_DONE;
                    else begin
                        m1_output_base <= m1_output_base + lane_count;
                        state <= ST_M1_GROUP;
                    end
                end

                ST_DONE: begin
                    o_busy <= 0;
                    o_done <= 1;
                    state <= ST_IDLE;
                end

                default: begin
                    o_error <= 1;
                    state <= ST_DONE;
                end
            endcase
        end
    end
endmodule
