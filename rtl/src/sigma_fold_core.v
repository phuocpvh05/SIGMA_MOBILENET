`timescale 1ns / 1ps
//==========================================================================
// NUM_PES-bank scratchpad-backed SIGMA core with hardware K-fold accumulation.
// A/B are loaded through SRAM-like write ports. The controller stages one A
// row and one B column concurrently per cycle, then slices logical
// K into <=NUM_PES-element stationary folds, invokes the 4x4 Flex-DPU, and combines
// fold results in an FP32 C scratchpad using the same IEEE adder as the FAN.
//==========================================================================

module sigma_fold_core #(
    parameter IN_DATA_TYPE = 16,
    parameter OUT_DATA_TYPE = 32,
    parameter NUM_PES = 2,
    parameter LOG2_PES = 1,
    parameter MESH_ROWS = 4,
    parameter MESH_COLS = 4,
    parameter NUM_DPE = MESH_ROWS * MESH_COLS,
    parameter MESH_ROUTE_WIDTH = 15,
    parameter MESH_HOP_WIDTH = 3,
    parameter MAX_M = 16,
    parameter MAX_N = 16,
    parameter FOLD_K = NUM_PES,
    parameter MAX_TOTAL_K = 64,
    parameter AUTO_BENES_CONFIG = 0,
    parameter DIRECT_BANK_WRITE = 0,
    // MobileNet/AlexNet enable a fully pipelined one-result-per-cycle fold
    // accumulator.  The default remains the proven legacy schedule so LeNet,
    // generic GEMM and the future SonicBoom profile are not changed.
    parameter PIPELINED_FOLD_ACCUM = 0,
    // Number of independent C columns accumulated per clock after a K fold.
    // Result memories are already physically banked by column, so widening
    // this path does not replicate storage or the SIGMA mesh.  Keep one lane
    // by default; the MobileNet profile uses four to remove the serial
    // 256-cycle fold-reduction bottleneck from every 16x16 tile.
    parameter FOLD_ACCUM_LANES = 1,
    parameter SCRATCH_RAM_STYLE = "block",
    // The per-column C/fold memories are only MAX_M x 32 bits.  Keep the
    // historical BRAM mapping by default, but allow memory-heavy on-chip
    // models to place these tiny banks in LUTRAM and reserve BRAM for weights.
    parameter RESULT_RAM_STYLE = "block",
    parameter A_DEPTH = MAX_M * MAX_TOTAL_K,
    parameter B_DEPTH = MAX_N * MAX_TOTAL_K,
    parameter C_DEPTH = MAX_M * MAX_N,
    parameter BENES_CTRL_WIDTH = 2 * (2 * LOG2_PES - 1) * NUM_PES + NUM_PES
) (
    input  wire clk,
    input  wire rst,
    input  wire i_start,
    input  wire [7:0] i_M,
    input  wire [7:0] i_N,
    input  wire [15:0] i_K_total,

    input  wire i_A_wr_en,
    input  wire [15:0] i_A_wr_addr,
    input  wire [IN_DATA_TYPE-1:0] i_A_wr_data,
    input  wire i_B_wr_en,
    input  wire [15:0] i_B_wr_addr,
    input  wire [IN_DATA_TYPE-1:0] i_B_wr_data,
    // Optional four-lane write used by the packed MobileNet controller.  With
    // FOLD_K=4 and DIRECT_BANK_WRITE=1, one 64-bit ROM row maps exactly onto
    // the four independent B scratchpad banks.
    input  wire i_B_wide_wr_en,
    input  wire [15:0] i_B_wide_wr_addr,
    input  wire [4*IN_DATA_TYPE-1:0] i_B_wide_wr_data,
    input  wire [NUM_DPE*BENES_CTRL_WIDTH-1:0] i_benes_config,
    input  wire i_mesh_route_enable,
    input  wire [NUM_DPE*MESH_ROUTE_WIDTH-1:0] i_mesh_route_config,
    input  wire [NUM_DPE*MESH_HOP_WIDTH-1:0] i_mesh_hops,

    input  wire i_C_rd_en,
    input  wire [15:0] i_C_rd_addr,
    output reg  [OUT_DATA_TYPE-1:0] o_C_rd_data,
    output reg  o_C_rd_valid,

    output reg  o_busy,
    output reg  o_done,
    output reg  o_error,
    output reg  [MAX_N*OUT_DATA_TYPE-1:0] o_C_row,
    output reg  o_C_valid,
    output wire o_config_active,
    output wire [7:0] o_config_fold,
    output wire [7:0] o_config_tile,
    output wire [7:0] o_config_row
);
    localparam CORE_IDLE = 4'd0;
    localparam CORE_PREP = 4'd1;
    localparam CORE_LOAD = 4'd2;
    localparam CORE_START = 4'd3;
    localparam CORE_WAIT = 4'd4;
    localparam CORE_OUTPUT = 4'd5;
    localparam CORE_DONE = 4'd6;
    localparam CORE_ACCUM = 4'd7;
    localparam CORE_LOAD_CAPTURE = 4'd8;
    localparam CORE_ACCUM_REQ = 4'd9;
    localparam CORE_OUTPUT_REQ = 4'd10;
    localparam CORE_ACCUM_LAUNCH = 4'd11;
    localparam CORE_ACCUM_WAIT = 4'd12;
    localparam CORE_LOAD_STAGE = 4'd13;
    localparam CORE_ACCUM_STREAM = 4'd14;
    localparam ACCUM_PIPE_LATENCY = 7;
    localparam MAX_FOLDS = (MAX_TOTAL_K + FOLD_K - 1) / FOLD_K;
    localparam A_BANK_DEPTH = MAX_M * MAX_FOLDS;
    localparam B_BANK_DEPTH = MAX_N * MAX_FOLDS;
    localparam A_FOLD_INDEX_W = $clog2(MAX_M * FOLD_K + 1);
    localparam B_FOLD_INDEX_W = $clog2(MAX_N * FOLD_K + 1);

    reg [3:0] state;
    reg [7:0] fold_idx;
    reg [7:0] fold_row_count;
    reg [7:0] output_row;
    reg [7:0] load_limit_q;
    reg [7:0] load_remaining;
    reg [7:0] load_A_remaining;
    reg [7:0] load_B_remaining;
    reg load_A_active;
    reg load_B_active;
    reg [15:0] bank_read_addr_ctr;
    reg [15:0] accum_addr;
    reg [2:0] accum_wait_count;
    reg [15:0] accum_issue_addr;
    reg [7:0] accum_issue_row_index;
    reg [7:0] accum_issue_col_index;
    reg [15:0] accum_pending_addr;
    reg [7:0] accum_pending_lane_count;
    reg [15:0] accum_write_count;
    reg [15:0] accum_total;
    reg accum_read_pending;
    reg accum_pending_last;
    reg [ACCUM_PIPE_LATENCY-1:0] accum_pipe_valid;
    reg [ACCUM_PIPE_LATENCY-1:0] accum_pipe_last;
    reg [15:0] accum_addr_pipe [0:ACCUM_PIPE_LATENCY-1];
    reg [7:0] accum_lane_count_pipe [0:ACCUM_PIPE_LATENCY-1];
    reg sigma_start;
    reg [7:0] r_M, r_N;
    reg [15:0] r_K_total;
    reg [7:0] r_fold_logical_k;
    reg [7:0] r_fold_runtime_k;
    reg r_first_fold;
    reg [A_FOLD_INDEX_W-1:0] r_fold_A_base;
    // Must address every word in the flattened N x FOLD_K weight tile.
    // Eight bits was sufficient for 16 lanes but wrapped halfway through a
    // 16-column tile when the AlexNet profile grew to 32 lanes.
    reg [B_FOLD_INDEX_W-1:0] r_fold_B_base;

    // FOLD_K independent single-port banks provide one complete fold vector.
    // Each generated bank is a conventional 1-D synchronous RAM so Vivado can
    // infer BRAM instead of expanding a multi-dimensional array into FFs.
    wire [FOLD_K*IN_DATA_TYPE-1:0] A_bank_q;
    wire [FOLD_K*IN_DATA_TYPE-1:0] B_bank_q;
    reg [FOLD_K*IN_DATA_TYPE-1:0] r_A_bank_stage;
    reg [FOLD_K*IN_DATA_TYPE-1:0] r_B_bank_stage;
    reg [FOLD_K-1:0] r_A_bank_nonzero;
    reg [FOLD_K-1:0] r_B_bank_nonzero;

    wire [15:0] A_wr_row = i_A_wr_addr / MAX_TOTAL_K;
    wire [15:0] A_wr_k = i_A_wr_addr % MAX_TOTAL_K;
    wire [15:0] A_wr_fold = A_wr_k / FOLD_K;
    wire [15:0] A_wr_lane = A_wr_k % FOLD_K;
    wire [15:0] B_wr_col = i_B_wr_addr / MAX_TOTAL_K;
    wire [15:0] B_wr_k = i_B_wr_addr % MAX_TOTAL_K;
    wire [15:0] B_wr_fold = B_wr_k / FOLD_K;
    wire [15:0] B_wr_lane = B_wr_k % FOLD_K;
    // Fast write mapping for clients that use the canonical row-major linear
    // address and choose MAX_TOTAL_K divisible by FOLD_K.  In that layout the
    // bank word is simply linear_addr/FOLD_K and the bank lane is the remainder;
    // no row decode or row*MAX_FOLDS reconstruction is required.
    wire [15:0] A_wr_direct_addr = i_A_wr_addr / FOLD_K;
    wire [15:0] B_wr_direct_addr = i_B_wr_addr / FOLD_K;
    wire [15:0] A_wr_bank_addr = DIRECT_BANK_WRITE ? A_wr_direct_addr :
                                 (A_wr_row*MAX_FOLDS + A_wr_fold);
    wire [15:0] B_wr_bank_addr = DIRECT_BANK_WRITE ? B_wr_direct_addr :
                                 (B_wr_col*MAX_FOLDS + B_wr_fold);
    wire [15:0] A_wr_bank_lane = DIRECT_BANK_WRITE ?
                                 (i_A_wr_addr % FOLD_K) : A_wr_lane;
    wire [15:0] B_wr_bank_lane = DIRECT_BANK_WRITE ?
                                 (i_B_wr_addr % FOLD_K) : B_wr_lane;
    wire [15:0] B_wide_wr_bank_addr = i_B_wide_wr_addr / FOLD_K;
    wire B_wide_wr_addr_valid = DIRECT_BANK_WRITE && (FOLD_K == 4) &&
        (i_B_wide_wr_addr[1:0] == 0) &&
        (i_B_wide_wr_addr + 3 < B_DEPTH) &&
        (B_wide_wr_bank_addr < B_BANK_DEPTH);
    wire A_wr_addr_valid = DIRECT_BANK_WRITE ?
        ((i_A_wr_addr < A_DEPTH) && (A_wr_direct_addr < A_BANK_DEPTH)) :
        ((i_A_wr_addr < A_DEPTH) && (A_wr_row < MAX_M) &&
         (A_wr_fold < MAX_FOLDS));
    wire B_wr_addr_valid = DIRECT_BANK_WRITE ?
        ((i_B_wr_addr < B_DEPTH) && (B_wr_direct_addr < B_BANK_DEPTH)) :
        ((i_B_wr_addr < B_DEPTH) && (B_wr_col < MAX_N) &&
         (B_wr_fold < MAX_FOLDS));

    wire [15:0] fold_base = fold_idx * FOLD_K;
    wire [15:0] fold_remaining = r_K_total - fold_base;
    wire [7:0] next_fold_logical_k =
        (fold_remaining > FOLD_K) ? FOLD_K : fold_remaining[7:0];

    function [7:0] padded_fold_k;
        input [7:0] value;
        begin
            // The sparse two-lane path keeps its exact K=1/K=2 behavior.
            // Wider performance profiles pad the final fold with zeros and
            // always execute a full dense stationary vector.
            if (FOLD_K == 2) begin
                if (value <= 1) padded_fold_k = 1;
                else padded_fold_k = 2;
            end else begin
                padded_fold_k = FOLD_K;
            end
        end
    endfunction
    wire [7:0] next_fold_runtime_k = padded_fold_k(next_fold_logical_k);
    wire [7:0] fold_count = (r_K_total + FOLD_K - 1) / FOLD_K;

    // CORE_LOAD visits one row/column per iteration at a fixed fold.  Keep the
    // physical bank address as an induction counter instead of rebuilding
    // load_index*MAX_FOLDS+fold_idx on the BRAM address path every cycle.
    wire [15:0] bank_read_addr = bank_read_addr_ctr;
    genvar scratch_bank;
    generate
        for (scratch_bank = 0; scratch_bank < FOLD_K;
             scratch_bank = scratch_bank + 1) begin : g_scratch_bank
            localparam integer A_AW = $clog2(A_BANK_DEPTH);
            localparam integer B_AW = $clog2(B_BANK_DEPTH);
            wire a_wr = i_A_wr_en && !o_busy && A_wr_addr_valid &&
                        (A_wr_bank_lane == scratch_bank);
            wire b_wide_wr = i_B_wide_wr_en && !o_busy &&
                             B_wide_wr_addr_valid;
            wire b_wr = (i_B_wr_en && !o_busy && B_wr_addr_valid &&
                         (B_wr_bank_lane == scratch_bank)) || b_wide_wr;
            wire [A_AW-1:0] a_wr_addr = A_wr_bank_addr[A_AW-1:0];
            wire [B_AW-1:0] b_wr_addr = b_wide_wr ?
                B_wide_wr_bank_addr[B_AW-1:0] :
                B_wr_bank_addr[B_AW-1:0];
            wire [IN_DATA_TYPE-1:0] b_wr_data = b_wide_wr ?
                i_B_wide_wr_data[scratch_bank*IN_DATA_TYPE +: IN_DATA_TYPE] :
                i_B_wr_data;
            wire [A_AW-1:0] a_rd_addr = bank_read_addr[A_AW-1:0];
            wire [B_AW-1:0] b_rd_addr = bank_read_addr[B_AW-1:0];
            sigma_sync_ram #(
                .WIDTH(IN_DATA_TYPE), .DEPTH(A_BANK_DEPTH),
                .ADDR_WIDTH(A_AW), .RAM_STYLE(SCRATCH_RAM_STYLE)
            ) u_A_bank (
                .clk(clk), .wr_en(a_wr),
                .wr_addr(a_wr_addr), .wr_data(i_A_wr_data),
                .rd_en(state == CORE_LOAD), .rd_addr(a_rd_addr),
                .rd_data(A_bank_q[scratch_bank*IN_DATA_TYPE +: IN_DATA_TYPE])
            );
            sigma_sync_ram #(
                .WIDTH(IN_DATA_TYPE), .DEPTH(B_BANK_DEPTH),
                .ADDR_WIDTH(B_AW), .RAM_STYLE(SCRATCH_RAM_STYLE)
            ) u_B_bank (
                .clk(clk), .wr_en(b_wr),
                .wr_addr(b_wr_addr), .wr_data(b_wr_data),
                .rd_en(state == CORE_LOAD), .rd_addr(b_rd_addr),
                .rd_data(B_bank_q[scratch_bank*IN_DATA_TYPE +: IN_DATA_TYPE])
            );
        end
    endgenerate

    localparam integer A_FOLD_WORDS = MAX_M * FOLD_K;
    localparam integer B_FOLD_WORDS = MAX_N * FOLD_K;
    reg [IN_DATA_TYPE-1:0] fold_A_words [0:A_FOLD_WORDS-1];
    reg [IN_DATA_TYPE-1:0] fold_B_words [0:B_FOLD_WORDS-1];
    reg fold_A_bits [0:A_FOLD_WORDS-1];
    reg fold_B_bits [0:B_FOLD_WORDS-1];
    wire [A_FOLD_WORDS*IN_DATA_TYPE-1:0] fold_A_mat;
    wire [B_FOLD_WORDS*IN_DATA_TYPE-1:0] fold_B_mat;
    wire [A_FOLD_WORDS-1:0] fold_A_bitmap;
    wire [B_FOLD_WORDS-1:0] fold_B_bitmap;
    genvar fold_aw, fold_bw;
    generate
        for (fold_aw = 0; fold_aw < A_FOLD_WORDS; fold_aw = fold_aw + 1) begin : g_fold_A_flat
            assign fold_A_mat[fold_aw*IN_DATA_TYPE +: IN_DATA_TYPE] =
                fold_A_words[fold_aw];
            assign fold_A_bitmap[fold_aw] = fold_A_bits[fold_aw];
        end
        for (fold_bw = 0; fold_bw < B_FOLD_WORDS; fold_bw = fold_bw + 1) begin : g_fold_B_flat
            assign fold_B_mat[fold_bw*IN_DATA_TYPE +: IN_DATA_TYPE] =
                fold_B_words[fold_bw];
            assign fold_B_bitmap[fold_bw] = fold_B_bits[fold_bw];
        end
    endgenerate
    integer fk, fold_dest;

    wire sigma_done, sigma_error, sigma_C_valid;
    wire [MAX_N*OUT_DATA_TYPE-1:0] sigma_C_row;
    // Register the completed SIGMA row before it fans out to the distributed
    // result-bank write enables/data.  Besides isolating the compute fabric,
    // this removes sigma_top's output-state decode from the RAM WE path.
    (* max_fanout = 16 *) reg sigma_C_capture_valid;
    (* max_fanout = 16 *) reg sigma_C_capture_primary;
    reg [7:0] sigma_C_capture_row;
    reg [MAX_N*OUT_DATA_TYPE-1:0] sigma_C_row_q;

    wire [7:0] accum_row = accum_addr / MAX_N;
    wire [7:0] accum_col = accum_addr % MAX_N;
    wire [7:0] accum_pending_col = accum_pending_addr % MAX_N;
    wire [7:0] accum_issue_lane_count =
        ((r_N - accum_issue_col_index) > FOLD_ACCUM_LANES) ?
        FOLD_ACCUM_LANES : (r_N - accum_issue_col_index);
    wire [7:0] accum_write_row =
        accum_addr_pipe[ACCUM_PIPE_LATENCY-1] / MAX_N;
    wire [7:0] accum_write_col =
        accum_addr_pipe[ACCUM_PIPE_LATENCY-1] % MAX_N;
    wire accum_stream_read = (PIPELINED_FOLD_ACCUM != 0) &&
                             (state == CORE_ACCUM_STREAM) &&
                             (accum_issue_addr != accum_total);
    wire [MAX_N*OUT_DATA_TYPE-1:0] C_bank_q;
    wire [MAX_N*OUT_DATA_TYPE-1:0] fold_bank_q;
    wire [FOLD_ACCUM_LANES*OUT_DATA_TYPE-1:0] accumulated_value_bus;
    reg [FOLD_ACCUM_LANES*OUT_DATA_TYPE-1:0] fold_add_a_bus;
    reg [FOLD_ACCUM_LANES*OUT_DATA_TYPE-1:0] fold_add_b_bus;
    wire [OUT_DATA_TYPE-1:0] accumulated_value =
        accumulated_value_bus[0 +: OUT_DATA_TYPE];
    reg C_external_pending;
    reg [7:0] C_external_col;

    genvar result_bank;
    generate
        for (result_bank = 0; result_bank < MAX_N;
             result_bank = result_bank + 1) begin : g_result_bank
            localparam integer R_AW = $clog2(MAX_M);
            wire capture = sigma_C_capture_valid && (result_bank < r_N);
            wire c_capture = capture && sigma_C_capture_primary;
            wire f_capture = capture && !sigma_C_capture_primary;
            wire legacy_c_accum = (state == CORE_ACCUM) &&
                                  (accum_col == result_bank);
            reg stream_c_accum;
            reg [OUT_DATA_TYPE-1:0] stream_c_accum_data;
            integer stream_lane;
            always @* begin
                stream_c_accum = 1'b0;
                stream_c_accum_data = 0;
                for (stream_lane = 0; stream_lane < FOLD_ACCUM_LANES;
                     stream_lane = stream_lane + 1) begin
                    if ((PIPELINED_FOLD_ACCUM != 0) &&
                        (state == CORE_ACCUM_STREAM) &&
                        accum_pipe_valid[ACCUM_PIPE_LATENCY-1] &&
                        (stream_lane < accum_lane_count_pipe[
                            ACCUM_PIPE_LATENCY-1]) &&
                        (accum_write_col + stream_lane == result_bank)) begin
                        stream_c_accum = 1'b1;
                        stream_c_accum_data = accumulated_value_bus[
                            stream_lane*OUT_DATA_TYPE +: OUT_DATA_TYPE];
                    end
                end
            end
            wire c_accum = legacy_c_accum || stream_c_accum;
            wire [R_AW-1:0] c_wr_addr = stream_c_accum ?
                accum_write_row[R_AW-1:0] :
                (legacy_c_accum ? accum_row[R_AW-1:0] :
                                  sigma_C_capture_row[R_AW-1:0]);
            wire [OUT_DATA_TYPE-1:0] c_wr_data = stream_c_accum ?
                stream_c_accum_data :
                (legacy_c_accum ? accumulated_value :
                 sigma_C_row_q[result_bank*OUT_DATA_TYPE +: OUT_DATA_TYPE]);
            wire internal_read = (state == CORE_ACCUM_REQ) ||
                                 (state == CORE_OUTPUT_REQ) ||
                                 accum_stream_read;
            wire external_read = i_C_rd_en && !o_busy &&
                                 (i_C_rd_addr < C_DEPTH);
            wire [R_AW-1:0] result_rd_addr = external_read ?
                (i_C_rd_addr / MAX_N) :
                ((state == CORE_OUTPUT_REQ) ? output_row[R_AW-1:0] :
                 (accum_stream_read ? accum_issue_row_index[R_AW-1:0] :
                                      accum_row[R_AW-1:0]));
            sigma_sync_ram #(
                .WIDTH(OUT_DATA_TYPE), .DEPTH(MAX_M),
                .ADDR_WIDTH(R_AW), .RAM_STYLE(RESULT_RAM_STYLE)
            ) u_C_bank (
                .clk(clk), .wr_en(c_capture || c_accum),
                .wr_addr(c_wr_addr), .wr_data(c_wr_data),
                .rd_en(internal_read || external_read), .rd_addr(result_rd_addr),
                .rd_data(C_bank_q[result_bank*OUT_DATA_TYPE +: OUT_DATA_TYPE])
            );
            sigma_sync_ram #(
                .WIDTH(OUT_DATA_TYPE), .DEPTH(MAX_M),
                .ADDR_WIDTH(R_AW), .RAM_STYLE(RESULT_RAM_STYLE)
            ) u_fold_bank (
                .clk(clk), .wr_en(f_capture),
                .wr_addr(sigma_C_capture_row[R_AW-1:0]),
                .wr_data(sigma_C_row_q[result_bank*OUT_DATA_TYPE +: OUT_DATA_TYPE]),
                .rd_en((state == CORE_ACCUM_REQ) || accum_stream_read),
                .rd_addr(accum_stream_read ? accum_issue_row_index[R_AW-1:0] :
                                             accum_row[R_AW-1:0]),
                .rd_data(fold_bank_q[result_bank*OUT_DATA_TYPE +: OUT_DATA_TYPE])
            );
        end
    endgenerate

    // K-fold accumulation is not on the Flex-DPE throughput-critical path.
    // Reuse one FP32 adder over the result scratchpad instead of instantiating
    // MAX_N parallel result banks; the default 32-PE build uses MAX_N=16.
    // Fold accumulation is outside the row-throughput pipeline, but it still
    // must meet the accelerator clock.  The former combinational adder formed
    // a 45-level, 14.4 ns BRAM-to-BRAM path.  Reuse the verified six-stage
    // FP32 pipeline and wait for its registered result before writing C.
    genvar accum_lane;
    generate
        for (accum_lane = 0; accum_lane < FOLD_ACCUM_LANES;
             accum_lane = accum_lane + 1) begin : g_fold_accum_lane
            fp32_add_pipeline u_fold_adder (
                .clk(clk), .rst(rst),
                .a(fold_add_a_bus[accum_lane*OUT_DATA_TYPE +: OUT_DATA_TYPE]),
                .b(fold_add_b_bus[accum_lane*OUT_DATA_TYPE +: OUT_DATA_TYPE]),
                .out(accumulated_value_bus[
                    accum_lane*OUT_DATA_TYPE +: OUT_DATA_TYPE])
            );
        end
    endgenerate

    integer output_col;
    integer accum_pipe_index;
    integer accum_feed_lane;
    always @(posedge clk) begin
        if (rst) begin
            state <= CORE_IDLE;
            fold_idx <= 0;
            fold_row_count <= 0;
            output_row <= 0;
            load_limit_q <= 0;
            load_remaining <= 0;
            load_A_remaining <= 0;
            load_B_remaining <= 0;
            load_A_active <= 0;
            load_B_active <= 0;
            bank_read_addr_ctr <= 0;
            accum_addr <= 0;
            accum_wait_count <= 0;
            accum_issue_addr <= 0;
            accum_issue_row_index <= 0;
            accum_issue_col_index <= 0;
            accum_pending_addr <= 0;
            accum_pending_lane_count <= 0;
            accum_write_count <= 0;
            accum_total <= 0;
            accum_read_pending <= 0;
            accum_pending_last <= 0;
            accum_pipe_valid <= 0;
            accum_pipe_last <= 0;
            for (accum_pipe_index = 0;
                 accum_pipe_index < ACCUM_PIPE_LATENCY;
                 accum_pipe_index = accum_pipe_index + 1) begin
                accum_addr_pipe[accum_pipe_index] <= 0;
                accum_lane_count_pipe[accum_pipe_index] <= 0;
            end
            sigma_start <= 0;
            o_busy <= 0;
            o_done <= 0;
            o_error <= 0;
            o_C_row <= 0;
            o_C_valid <= 0;
            o_C_rd_data <= 0;
            o_C_rd_valid <= 0;
            C_external_pending <= 0;
            C_external_col <= 0;
            r_M <= 0;
            r_N <= 0;
            r_K_total <= 0;
            r_fold_logical_k <= 0;
            r_fold_runtime_k <= 0;
            r_first_fold <= 0;
            r_fold_A_base <= 0;
            r_fold_B_base <= 0;
            r_A_bank_stage <= 0;
            r_B_bank_stage <= 0;
            r_A_bank_nonzero <= 0;
            r_B_bank_nonzero <= 0;
            sigma_C_capture_valid <= 0;
            sigma_C_capture_primary <= 0;
            sigma_C_capture_row <= 0;
            sigma_C_row_q <= 0;
            fold_add_a_bus <= 0;
            fold_add_b_bus <= 0;
        end else begin
            sigma_start <= 0;
            o_C_valid <= 0;
            o_C_rd_valid <= C_external_pending;
            if (C_external_pending)
                o_C_rd_data <= C_bank_q[C_external_col*OUT_DATA_TYPE +: OUT_DATA_TYPE];
            C_external_pending <= i_C_rd_en && !o_busy && (i_C_rd_addr < C_DEPTH);
            if (i_C_rd_en && !o_busy && (i_C_rd_addr < C_DEPTH))
                C_external_col <= i_C_rd_addr % MAX_N;

            sigma_C_capture_valid <= (state == CORE_WAIT) && sigma_C_valid &&
                                     (fold_row_count < r_M);
            if ((state == CORE_WAIT) && sigma_C_valid && (fold_row_count < r_M)) begin
                sigma_C_capture_primary <= r_first_fold;
                sigma_C_capture_row <= fold_row_count;
                sigma_C_row_q <= sigma_C_row;
                fold_row_count <= fold_row_count + 1'b1;
            end

            case (state)
                CORE_IDLE: begin
                    o_busy <= 0;
                    o_done <= 0;
                    o_error <= 0;
                    if (i_start) begin
                        if ((i_M == 0) || (i_M > MAX_M) ||
                            (i_N == 0) || (i_N > MAX_N) ||
                            (i_K_total == 0) || (i_K_total > MAX_TOTAL_K)) begin
                            o_error <= 1'b1;
                            o_done <= 1'b1;
                            state <= CORE_DONE;
                        end else begin
                            r_M <= i_M;
                            r_N <= i_N;
                            r_K_total <= i_K_total;
                            load_limit_q <= (i_M > i_N) ? i_M : i_N;
                            o_busy <= 1'b1;
                            fold_idx <= 0;
                            fold_row_count <= 0;
                            state <= CORE_PREP;
                        end
                    end
                end
                CORE_PREP: begin
                    // Isolate fold_idx/K arithmetic from the wide A/B staging
                    // arrays. All load cycles consume this registered fold
                    // descriptor instead of re-decoding it per destination.
                    r_fold_logical_k <= next_fold_logical_k;
                    r_fold_runtime_k <= next_fold_runtime_k;
                    r_first_fold <= (fold_idx == 0);
                    r_fold_A_base <= 0;
                    r_fold_B_base <= 0;
                    fold_row_count <= 0;
                    load_remaining <= load_limit_q;
                    load_A_remaining <= r_M;
                    load_B_remaining <= r_N;
                    load_A_active <= 1'b1;
                    load_B_active <= 1'b1;
                    bank_read_addr_ctr <= fold_idx;
                    state <= CORE_LOAD;
                end
                CORE_LOAD: begin
                    // Address phase for synchronous bank reads.
                    state <= CORE_LOAD_STAGE;
                end
                CORE_LOAD_STAGE: begin
                    // Register BRAM outputs and their nonzero flags before
                    // decoding the variable fold destinations. This removes
                    // BRAM clock-to-out from the deepest metadata path.
                    r_A_bank_stage <= A_bank_q;
                    r_B_bank_stage <= B_bank_q;
                    for (fk = 0; fk < FOLD_K; fk = fk + 1) begin
                        r_A_bank_nonzero[fk] <=
                            |A_bank_q[fk*IN_DATA_TYPE +: IN_DATA_TYPE];
                        r_B_bank_nonzero[fk] <=
                            |B_bank_q[fk*IN_DATA_TYPE +: IN_DATA_TYPE];
                    end
                    state <= CORE_LOAD_CAPTURE;
                end
                CORE_LOAD_CAPTURE: begin
                    // A and B use independent banks, so one activation row and
                    // one weight column are captured in the same cycle.
                    if (load_A_active) begin
                        for (fk = 0; fk < FOLD_K; fk = fk + 1) begin
                            fold_dest = r_fold_A_base + fk;
                            if (fk < r_fold_logical_k) begin
                                fold_A_words[fold_dest] <=
                                    r_A_bank_stage[fk*IN_DATA_TYPE +: IN_DATA_TYPE];
                                fold_A_bits[fold_dest] <=
                                    r_A_bank_nonzero[fk];
                            end else begin
                                fold_A_words[fold_dest] <= 0;
                                fold_A_bits[fold_dest] <= 1'b0;
                            end
                        end
                        r_fold_A_base <= r_fold_A_base + FOLD_K;
                        if (load_A_remaining == 1)
                            load_A_active <= 1'b0;
                        else
                            load_A_remaining <= load_A_remaining - 1'b1;
                    end
                    if (load_B_active) begin
                        for (fk = 0; fk < FOLD_K; fk = fk + 1) begin
                            if (fk < r_fold_runtime_k) begin
                                fold_dest = r_fold_B_base + fk;
                                if (fk < r_fold_logical_k) begin
                                    fold_B_words[fold_dest] <=
                                        r_B_bank_stage[fk*IN_DATA_TYPE +: IN_DATA_TYPE];
                                    fold_B_bits[fold_dest] <=
                                        r_B_bank_nonzero[fk];
                                end else begin
                                    fold_B_words[fold_dest] <= 0;
                                    fold_B_bits[fold_dest] <= 1'b0;
                                end
                            end
                        end
                        r_fold_B_base <= r_fold_B_base + r_fold_runtime_k;
                        if (load_B_remaining == 1)
                            load_B_active <= 1'b0;
                        else
                            load_B_remaining <= load_B_remaining - 1'b1;
                    end
                    if (load_remaining == 1)
                        state <= CORE_START;
                    else begin
                        load_remaining <= load_remaining - 1'b1;
                        bank_read_addr_ctr <= bank_read_addr_ctr + MAX_FOLDS;
                        state <= CORE_LOAD;
                    end
                end
                CORE_START: begin
                    sigma_start <= 1'b1;
                    state <= CORE_WAIT;
                end
                CORE_WAIT: begin
                    if (sigma_error) begin
                        o_error <= 1'b1;
                        state <= CORE_DONE;
                    end else if (sigma_done) begin
                        if (fold_idx != 0) begin
                            if (PIPELINED_FOLD_ACCUM != 0) begin
                                accum_issue_addr <= 0;
                                accum_issue_row_index <= 0;
                                accum_issue_col_index <= 0;
                                accum_pending_addr <= 0;
                                accum_write_count <= 0;
                                accum_total <= r_M * r_N;
                                accum_read_pending <= 0;
                                accum_pending_last <= 0;
                                accum_pipe_valid <= 0;
                                accum_pipe_last <= 0;
                                state <= CORE_ACCUM_STREAM;
                            end else begin
                                accum_addr <= 0;
                                state <= CORE_ACCUM_REQ;
                            end
                        end else if (fold_idx + 1'b1 < fold_count) begin
                            fold_idx <= fold_idx + 1'b1;
                            state <= CORE_PREP;
                        end else begin
                            output_row <= 0;
                            state <= CORE_OUTPUT_REQ;
                        end
                    end
                end
                CORE_ACCUM_REQ: begin
                    // Synchronous C/fold bank address phase.  The BRAM outputs
                    // become valid for the pipeline in the following state.
                    state <= CORE_ACCUM_LAUNCH;
                end
                CORE_ACCUM_LAUNCH: begin
                    // Register both synchronous-BRAM outputs before the FP32
                    // align/compare stage. This removes BRAM clock-to-out and
                    // bank-routing delay from the adder's first logic stage.
                    fold_add_a_bus[0 +: OUT_DATA_TYPE] <=
                        C_bank_q[accum_col*OUT_DATA_TYPE +: OUT_DATA_TYPE];
                    fold_add_b_bus[0 +: OUT_DATA_TYPE] <=
                        fold_bank_q[accum_col*OUT_DATA_TYPE +: OUT_DATA_TYPE];
                    accum_wait_count <= 0;
                    state <= CORE_ACCUM_WAIT;
                end
                CORE_ACCUM_WAIT: begin
                    // One input-register edge plus the six-stage FP32 pipe.
                    // CORE_ACCUM writes only after the registered result is stable.
                    if (accum_wait_count == 5)
                        state <= CORE_ACCUM;
                    else
                        accum_wait_count <= accum_wait_count + 1'b1;
                end
                CORE_ACCUM: begin
                    // Only visit active columns.  Physical bank rows still use
                    // MAX_N addressing, so advance to the next row boundary
                    // explicitly instead of accumulating all 32 banks.
                    if ((accum_col + 1'b1 >= r_N) &&
                        (accum_row + 1'b1 >= r_M)) begin
                        if (fold_idx + 1'b1 < fold_count) begin
                            fold_idx <= fold_idx + 1'b1;
                            state <= CORE_PREP;
                        end else begin
                            output_row <= 0;
                            state <= CORE_OUTPUT_REQ;
                        end
                    end else if (accum_col + 1'b1 >= r_N) begin
                        accum_addr <= (accum_row + 1'b1) * MAX_N;
                        state <= CORE_ACCUM_REQ;
                    end else begin
                        accum_addr <= accum_addr + 1'b1;
                        state <= CORE_ACCUM_REQ;
                    end
                end
                CORE_ACCUM_STREAM: begin
                    // The FP32 adder accepts a new independent C element every
                    // cycle.  Address/valid metadata follows the seven edges
                    // from synchronous RAM capture to the registered result,
                    // reducing fold accumulation from ~9 cycles/element to 1.
                    for (accum_pipe_index = ACCUM_PIPE_LATENCY-1;
                         accum_pipe_index > 0;
                         accum_pipe_index = accum_pipe_index - 1) begin
                        accum_pipe_valid[accum_pipe_index] <=
                            accum_pipe_valid[accum_pipe_index-1];
                        accum_pipe_last[accum_pipe_index] <=
                            accum_pipe_last[accum_pipe_index-1];
                        accum_addr_pipe[accum_pipe_index] <=
                            accum_addr_pipe[accum_pipe_index-1];
                        accum_lane_count_pipe[accum_pipe_index] <=
                            accum_lane_count_pipe[accum_pipe_index-1];
                    end
                    accum_pipe_valid[0] <= 1'b0;
                    accum_pipe_last[0] <= 1'b0;
                    accum_lane_count_pipe[0] <= 0;

                    if (accum_read_pending) begin
                        for (accum_feed_lane = 0;
                             accum_feed_lane < FOLD_ACCUM_LANES;
                             accum_feed_lane = accum_feed_lane + 1) begin
                            if (accum_feed_lane < accum_pending_lane_count) begin
                                fold_add_a_bus[
                                    accum_feed_lane*OUT_DATA_TYPE +:
                                    OUT_DATA_TYPE] <= C_bank_q[
                                    (accum_pending_col + accum_feed_lane)*
                                    OUT_DATA_TYPE +: OUT_DATA_TYPE];
                                fold_add_b_bus[
                                    accum_feed_lane*OUT_DATA_TYPE +:
                                    OUT_DATA_TYPE] <= fold_bank_q[
                                    (accum_pending_col + accum_feed_lane)*
                                    OUT_DATA_TYPE +: OUT_DATA_TYPE];
                            end else begin
                                fold_add_a_bus[
                                    accum_feed_lane*OUT_DATA_TYPE +:
                                    OUT_DATA_TYPE] <= 0;
                                fold_add_b_bus[
                                    accum_feed_lane*OUT_DATA_TYPE +:
                                    OUT_DATA_TYPE] <= 0;
                            end
                        end
                        accum_pipe_valid[0] <= 1'b1;
                        accum_pipe_last[0] <= accum_pending_last;
                        accum_addr_pipe[0] <= accum_pending_addr;
                        accum_lane_count_pipe[0] <=
                            accum_pending_lane_count;
                    end

                    // The issue counter starts at zero, increments once per
                    // accepted address and stops exactly at accum_total.
                    // Equality is therefore equivalent to the old less-than
                    // test without placing a carry-chain comparator on every
                    // distributed result-bank read enable.
                    if (accum_issue_addr != accum_total) begin
                        accum_pending_addr <=
                            accum_issue_row_index * MAX_N +
                            accum_issue_col_index;
                        accum_pending_lane_count <=
                            accum_issue_lane_count;
                        accum_pending_last <=
                            (accum_issue_addr + accum_issue_lane_count >=
                             accum_total);
                        accum_read_pending <= 1'b1;
                        accum_issue_addr <= accum_issue_addr +
                                            accum_issue_lane_count;
                        if (accum_issue_col_index +
                            accum_issue_lane_count >= r_N) begin
                            accum_issue_col_index <= 0;
                            accum_issue_row_index <=
                                accum_issue_row_index + 1'b1;
                        end else begin
                            accum_issue_col_index <=
                                accum_issue_col_index +
                                accum_issue_lane_count;
                        end
                    end else begin
                        accum_read_pending <= 1'b0;
                    end

                    if (accum_pipe_valid[ACCUM_PIPE_LATENCY-1]) begin
                        if (accum_pipe_last[ACCUM_PIPE_LATENCY-1]) begin
                            accum_write_count <= 0;
                            accum_pipe_valid <= 0;
                            accum_pipe_last <= 0;
                            if (fold_idx + 1'b1 < fold_count) begin
                                fold_idx <= fold_idx + 1'b1;
                                state <= CORE_PREP;
                            end else begin
                                output_row <= 0;
                                state <= CORE_OUTPUT_REQ;
                            end
                        end else begin
                            accum_write_count <= accum_write_count + 1'b1;
                        end
                    end
                end
                CORE_OUTPUT_REQ: begin
                    state <= CORE_OUTPUT;
                end
                CORE_OUTPUT: begin
                    o_C_row <= 0;
                    for (output_col = 0; output_col < MAX_N; output_col = output_col + 1)
                        if (output_col < r_N)
                            o_C_row[output_col*OUT_DATA_TYPE +: OUT_DATA_TYPE] <=
                                C_bank_q[output_col*OUT_DATA_TYPE +: OUT_DATA_TYPE];
                    o_C_valid <= 1'b1;
                    if (output_row + 1'b1 >= r_M)
                        state <= CORE_DONE;
                    else begin
                        output_row <= output_row + 1'b1;
                        state <= CORE_OUTPUT_REQ;
                    end
                end
                CORE_DONE: begin
                    o_busy <= 0;
                    o_done <= 1'b1;
                    if (!i_start)
                        state <= CORE_IDLE;
                end
                default: state <= CORE_IDLE;
            endcase
        end
    end

    sigma_top #(
        .IN_DATA_TYPE(IN_DATA_TYPE), .OUT_DATA_TYPE(OUT_DATA_TYPE),
        .NUM_PES(NUM_PES), .LOG2_PES(LOG2_PES),
        .MESH_ROWS(MESH_ROWS), .MESH_COLS(MESH_COLS), .NUM_DPE(NUM_DPE),
        .MESH_ROUTE_WIDTH(MESH_ROUTE_WIDTH), .MESH_HOP_WIDTH(MESH_HOP_WIDTH),
        .MAX_M(MAX_M), .MAX_N(MAX_N), .MAX_K(FOLD_K),
        .RESULT_RAM_STYLE(RESULT_RAM_STYLE),
        .AUTO_BENES_CONFIG(AUTO_BENES_CONFIG),
        .BENES_CTRL_WIDTH(BENES_CTRL_WIDTH)
    ) u_sigma (
        .clk(clk), .rst(rst), .i_start(sigma_start),
        .i_M(r_M), .i_N(r_N), .i_K(r_fold_runtime_k),
        .i_B_mat(fold_B_mat), .i_A_mat(fold_A_mat),
        .i_A_bitmap(fold_A_bitmap), .i_B_bitmap(fold_B_bitmap),
        .i_benes_config(i_benes_config),
        .i_mesh_route_enable(i_mesh_route_enable),
        .i_mesh_route_config(i_mesh_route_config), .i_mesh_hops(i_mesh_hops),
        .o_done(sigma_done), .o_error(sigma_error),
        .o_C_row(sigma_C_row), .o_C_valid(sigma_C_valid)
    );

    assign o_config_active = (state == CORE_WAIT) &&
                             (u_sigma.state == 3) && (u_sigma.m_cnt < r_M);
    assign o_config_fold = fold_idx;
    assign o_config_tile = u_sigma.tile_idx;
    assign o_config_row = u_sigma.m_cnt;
endmodule
