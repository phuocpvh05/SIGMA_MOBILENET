`timescale 1ns / 1ps
//==========================================================================
// SIGMA wrapper with sparse stationary packing and parallel column tiling.
// A MESH_ROWS x MESH_COLS grid of Flex-DPEs forms one Flex-DPU. Each DPE owns a weight
// tile; the activation row is broadcast to all active DPEs in the group.
//==========================================================================

module sigma_top #(
    parameter IN_DATA_TYPE  = 16,
    parameter OUT_DATA_TYPE = 32,
    parameter NUM_PES       = 2,
    parameter LOG2_PES      = 1,
    parameter FAN_ADD_PIPE_STAGES = 6,
    parameter MESH_ROWS     = 4,
    parameter MESH_COLS     = 4,
    parameter NUM_DPE       = MESH_ROWS * MESH_COLS,
    parameter MESH_ROUTE_WIDTH = 15,
    parameter MESH_HOP_WIDTH = 3,
    parameter MAX_M         = 16,
    parameter MAX_N         = 16,
    parameter MAX_K         = 2,
    // Each output bank is only MAX_M x 32 bits.  Generic GEMM retains the
    // historical BRAM mapping; memory-heavy fixed-model tops can use LUTRAM.
    parameter RESULT_RAM_STYLE = "block",
    // Generate the verified two-lane sparse route locally from the staged
    // A/B bitmaps.  The default keeps the original externally supplied route
    // stream for the generic GEMM testbench; autonomous fixed-model tops can
    // enable this to avoid a large random-read metadata memory.
    parameter AUTO_BENES_CONFIG = 0,
    parameter BENES_CTRL_WIDTH = 2 * (2 * LOG2_PES - 1) * NUM_PES + NUM_PES
) (
    input  wire clk,
    input  wire rst,
    input  wire i_start,
    input  wire [7:0] i_M,
    input  wire [7:0] i_N,
    input  wire [7:0] i_K,
    input  wire [MAX_N*MAX_K*IN_DATA_TYPE-1:0] i_B_mat,
    input  wire [MAX_M*MAX_K*IN_DATA_TYPE-1:0] i_A_mat,
    input  wire [MAX_M*MAX_K-1:0] i_A_bitmap,
    input  wire [MAX_N*MAX_K-1:0] i_B_bitmap,
    input  wire [NUM_DPE*BENES_CTRL_WIDTH-1:0] i_benes_config,
    input  wire i_mesh_route_enable,
    input  wire [NUM_DPE*MESH_ROUTE_WIDTH-1:0] i_mesh_route_config,
    input  wire [NUM_DPE*MESH_HOP_WIDTH-1:0] i_mesh_hops,
    output reg  o_done,
    output reg  o_error,
    output reg  [MAX_N*OUT_DATA_TYPE-1:0] o_C_row,
    output reg  o_C_valid
);
    localparam ST_IDLE       = 4'd0;
    localparam ST_LOAD_STAT  = 4'd1;
    localparam ST_WAIT_PIPE  = 4'd2;
    localparam ST_STREAM     = 4'd3;
    localparam ST_COLLECT    = 4'd4;
    localparam ST_DONE       = 4'd5;
    localparam ST_OUTPUT     = 4'd6;
    localparam ST_OUTPUT_REQ = 4'd7;
    localparam ST_PREP_B     = 4'd8;
    localparam ST_PACK_B     = 4'd9;
    localparam ST_PACK_META  = 4'd10;
    localparam ST_PREP_DESC  = 4'd11;
    localparam ST_OUTPUT_MEM = 4'd12;
    // One extra cycle is introduced by the registered static mesh switch in
    // flexdpu before the original Flex-DPE pipeline.
    // 1 mesh register + Flex-DPE input/Benes/multiplier pipeline + the
    // LOG2_PES-level FAN. Each logical FAN level now has a multi-stage FP32
    // pipeline, while still accepting one row per cycle.
    // 14 front-end cycles includes the eight-stage BF16 multiplier. Keep the
    // launch tag aligned with the result-valid plane after the Fmax stage split.
    localparam integer RESULT_TAG_LATENCY =
        14 + FAN_ADD_PIPE_STAGES * LOG2_PES;

    // The original sparse frontend is retained bit-for-bit for the two-lane
    // LeNet/GEMM profile.  Wider CNN profiles execute one dense K=NUM_PES
    // dot-product per DPE.  This still uses the SIGMA stationary Benes/FAN
    // datapath, but avoids constructing a very large run-time sparse router
    // for the fixed on-chip networks.
    function [7:0] tile_capacity;
        input [7:0] k;
        begin
            if ((NUM_PES == 2) && (k == 1))
                tile_capacity = 2;
            else if ((NUM_PES == 2) && (k == 2))
                tile_capacity = 1;
            else if ((NUM_PES > 2) && (k == NUM_PES))
                tile_capacity = 1;
            else
                tile_capacity = 0;
        end
    endfunction

    function [7:0] tile_count_for;
        input [7:0] n;
        input [7:0] k;
        begin
            if ((NUM_PES == 2) && (k == 1))
                tile_count_for = (n + 1) >> 1;
            else if (((NUM_PES == 2) && (k == 2)) ||
                     ((NUM_PES > 2) && (k == NUM_PES)))
                tile_count_for = n;
            else
                tile_count_for = 0;
        end
    endfunction

    function [7:0] tile_base_for;
        input [7:0] tile;
        input [7:0] k;
        begin
            if ((NUM_PES == 2) && (k == 1))
                tile_base_for = tile << 1;
            else
                tile_base_for = tile;
        end
    endfunction

    function [1:0] regor2;
        input [1:0] bitmap;
        input [7:0] k;
        input [7:0] n;
        begin
            regor2 = 0;
            if (n != 0) begin
                if (k == 1)
                    regor2[0] = bitmap[0] | ((n > 1) & bitmap[1]);
                else
                    regor2 = bitmap;
            end
        end
    endfunction

    reg [3:0] state;
    reg [4:0] wait_cnt;
    reg [5:0] m_cnt;
    reg [7:0] result_count;
    // Base tile of the DPU group.  Kept as tile_idx for testbench/debug
    // compatibility; DPE d executes tile_idx+d.
    reg [7:0] tile_idx;
    reg [5:0] output_row;
    reg [7:0] prep_dpe;
    reg [7:0] prep_lane;
    reg [7:0] r_M, r_N, r_K;
    reg [7:0] r_tile_count;
    reg [MAX_N*MAX_K*IN_DATA_TYPE-1:0] r_B_shift;
    reg [MAX_M*MAX_K*IN_DATA_TYPE-1:0] r_A_mat;
    reg [MAX_M*MAX_K*IN_DATA_TYPE-1:0] r_A_shift;
    reg [MAX_M*MAX_K-1:0] r_A_bitmap;
    reg [MAX_M*MAX_K-1:0] r_A_bitmap_shift;
    reg [MAX_N*MAX_K-1:0] r_B_bitmap;
    reg r_mesh_route_enable;
    reg [NUM_DPE*MESH_ROUTE_WIDTH-1:0] r_mesh_route_config;
    reg [NUM_DPE*MESH_HOP_WIDTH-1:0] r_mesh_hops;
    reg [MAX_K*IN_DATA_TYPE-1:0] r_A_row_data;
    reg [MAX_K-1:0] r_A_row_bitmap;
    reg r_A_stage_valid;
    reg [NUM_PES*IN_DATA_TYPE-1:0] r_current_B_mat;
    reg [NUM_PES-1:0] r_current_B_bitmap;
    reg [IN_DATA_TYPE-1:0] r_B_packed_data [0:NUM_DPE*NUM_PES-1];
    reg [LOG2_PES-1:0] r_B_packed_tag [0:NUM_DPE*NUM_PES-1];
    reg r_tile_active [0:NUM_DPE-1];
    reg [7:0] r_tile_N [0:NUM_DPE-1];
    reg [7:0] r_tile_base [0:NUM_DPE-1];
    reg [NUM_PES-1:0] r_A_regor [0:NUM_DPE-1];
    reg [NUM_PES-1:0] r_B_stationary_bitmap [0:NUM_DPE-1];
    reg r_prep_active;
    reg [7:0] r_prep_tile_N;
    reg [7:0] r_prep_tile_base;
    reg [7:0] r_prep_valid_lanes;
    reg [NUM_PES-1:0] r_prep_A_regor;

    wire [7:0] w_tile_capacity = tile_capacity(i_K);
    wire [7:0] w_tile_count = r_tile_count;
    wire w_dimensions_valid = (i_M != 0) && (i_M <= MAX_M) &&
                              (i_N != 0) && (i_N <= MAX_N) &&
                              (i_K != 0) && (i_K <= MAX_K) &&
                              (w_tile_capacity != 0) && (NUM_DPE > 0) &&
                              (((NUM_PES == 2) && (LOG2_PES == 1) &&
                                (MAX_K == 2)) ||
                               ((NUM_PES > 2) &&
                                ((1 << LOG2_PES) == NUM_PES) &&
                                (MAX_K == NUM_PES) &&
                                (i_K == NUM_PES))) &&
                              (NUM_DPE == MESH_ROWS * MESH_COLS);

    wire [NUM_DPE-1:0] w_tile_active;
    wire [NUM_DPE*8-1:0] w_tile_N_all;
    wire [NUM_DPE*8-1:0] w_tile_base_all;
    wire [NUM_DPE*NUM_PES*IN_DATA_TYPE-1:0] w_B_packed_all;
    wire [NUM_DPE*NUM_PES*LOG2_PES-1:0] w_B_vn_all;
    wire [NUM_DPE*NUM_PES*IN_DATA_TYPE-1:0] w_A_compressed_all;

    // One shared B extractor replaces NUM_DPE copies of a variable 2048-bit
    // part-select. The old combinational mapping synthesized into hundreds of
    // thousands of LUT6/MUXF7 cells in the full 4x4 design.
    wire [7:0] prep_tile_index = tile_idx + prep_dpe;
    wire prep_active = prep_tile_index < w_tile_count;
    wire [7:0] prep_tile_base = tile_base_for(prep_tile_index, r_K);
    wire [7:0] prep_remaining = prep_active ? (r_N - prep_tile_base) : 0;
    wire [7:0] r_tile_capacity = tile_capacity(r_K);
    wire [7:0] prep_tile_N = !prep_active ? 0 :
        ((prep_remaining > r_tile_capacity) ? r_tile_capacity : prep_remaining);
    // The descriptor is registered before touching the wide stationary tile.
    // Keeping tile_idx/tile-count arithmetic out of the per-lane write-enable
    // cone is important once sixteen DPEs are placed across the device.
    wire prep_element_valid =
        r_prep_active && (prep_lane < r_prep_valid_lanes);

    wire [NUM_PES*IN_DATA_TYPE-1:0] shared_B_packed;
    wire [NUM_PES*LOG2_PES-1:0] shared_B_tags;
    wire [7:0] shared_B_count;
    generate
        if (NUM_PES == 2) begin : g_sparse_stationary_compactor
            stationary_compactor #(.DATA_TYPE(IN_DATA_TYPE))
            u_shared_stationary_compactor (
                .i_B_mat(r_current_B_mat),
                .i_B_bitmap(r_current_B_bitmap),
                .i_K(r_K), .i_N(r_prep_tile_N),
                .o_B_packed(shared_B_packed),
                .o_vn_sep(shared_B_tags),
                .o_packed_count(shared_B_count)
            );
        end else begin : g_dense_stationary_lanes
            // Identity Benes route: every lane owns one K element and all
            // lanes carry VN tag zero, so the FAN reduces the full dot product.
            assign shared_B_packed = r_current_B_mat;
            assign shared_B_tags = {NUM_PES*LOG2_PES{1'b0}};
            assign shared_B_count = NUM_PES;
        end
    endgenerate

    genvar td, bd;
    generate
        for (td = 0; td < NUM_DPE; td = td + 1) begin : g_tile_frontend
            wire d_active = r_tile_active[td];
            wire [7:0] d_tile_base = r_tile_base[td];
            wire [7:0] d_tile_N = r_tile_N[td];

            wire [NUM_PES*IN_DATA_TYPE-1:0] d_A_compressed;
            if (NUM_PES == 2) begin : g_sparse_activation_packer
                sparsity_packer #(.DATA_TYPE(IN_DATA_TYPE))
                u_sparsity_packer (
                    .i_data(r_A_row_data),
                    .i_A_bitmap(r_A_row_bitmap),
                    .i_REGOR(r_A_regor[td]),
                    .o_comp_data(d_A_compressed)
                );
            end else begin : g_dense_activation_lanes
                assign d_A_compressed = r_A_row_data;
            end

            wire [NUM_PES*IN_DATA_TYPE-1:0] d_B_packed;
            wire [NUM_PES*LOG2_PES-1:0] d_vn_sep;
            for (bd = 0; bd < NUM_PES; bd = bd + 1) begin : g_B_packed_lane
                assign d_B_packed[bd*IN_DATA_TYPE +: IN_DATA_TYPE] =
                    r_B_packed_data[td*NUM_PES + bd];
                assign d_vn_sep[bd*LOG2_PES +: LOG2_PES] =
                    r_B_packed_tag[td*NUM_PES + bd];
            end

            assign w_tile_active[td] = d_active;
            assign w_tile_N_all[td*8 +: 8] = d_tile_N;
            assign w_tile_base_all[td*8 +: 8] = d_tile_base;
            assign w_B_packed_all[td*NUM_PES*IN_DATA_TYPE +: NUM_PES*IN_DATA_TYPE] = d_B_packed;
            assign w_B_vn_all[td*NUM_PES*LOG2_PES +: NUM_PES*LOG2_PES] = d_vn_sep;
            assign w_A_compressed_all[
                td*NUM_PES*IN_DATA_TYPE +: NUM_PES*IN_DATA_TYPE] =
                d_A_compressed;
        end
    endgenerate

    reg r_data_valid;
    reg r_stationary;
    reg [NUM_DPE*NUM_PES*IN_DATA_TYPE-1:0] r_data_bus_all;
    reg [NUM_DPE*NUM_PES*LOG2_PES-1:0] r_vn_all;
    reg [NUM_DPE*BENES_CTRL_WIDTH-1:0] r_benes_stream;

    // The autonomous LeNet mapper only needs the two route patterns proven by
    // the exhaustive two-PE Benes test:
    //   K=2: 0x21 moves lane 1 into the empty lane-0 position when both
    //        stationary lanes are live.
    //   K=1: 0x20 multicasts the single activation to output column 1.
    // Build them from the fold-local registers already present in sigma_top.
    // This replaces MAX_M/MAX_N copies of full-K nonzero metadata and removes
    // the thousands-to-one mux trees that otherwise dominate FPGA LUT use.
    wire [NUM_DPE*BENES_CTRL_WIDTH-1:0] w_auto_benes_config;
    genvar auto_dpe;
    generate
        for (auto_dpe = 0; auto_dpe < NUM_DPE;
            auto_dpe = auto_dpe + 1) begin : g_auto_benes_config
            wire auto_pair_cross = r_tile_active[auto_dpe] &&
                !r_A_row_bitmap[0] && r_A_row_bitmap[1] &&
                r_B_stationary_bitmap[auto_dpe][0] &&
                r_B_stationary_bitmap[auto_dpe][1];
            wire auto_single_cross = r_tile_active[auto_dpe] &&
                (r_tile_N[auto_dpe] > 1) &&
                r_B_stationary_bitmap[auto_dpe][1];
            if (NUM_PES == 2) begin : g_sparse_auto_route
                assign w_auto_benes_config[
                    auto_dpe*BENES_CTRL_WIDTH +: BENES_CTRL_WIDTH] =
                    ((state != ST_STREAM) || !r_A_stage_valid) ? 6'h00 :
                    ((r_K == 1) ? (auto_single_cross ? 6'h20 : 6'h00) :
                                  (auto_pair_cross ? 6'h21 : 6'h00));
            end else begin : g_dense_identity_route
                assign w_auto_benes_config[
                    auto_dpe*BENES_CTRL_WIDTH +: BENES_CTRL_WIDTH] =
                    {BENES_CTRL_WIDTH{1'b0}};
            end
        end
    endgenerate

    reg next_data_valid;
    reg next_stationary;
    always @(*) begin
        next_data_valid = 1'b0;
        next_stationary = 1'b0;
        case (state)
            ST_LOAD_STAT: begin
                next_data_valid = 1'b1;
                next_stationary = 1'b1;
            end
            ST_STREAM: begin
                next_data_valid = r_A_stage_valid;
            end
            default: begin end
        endcase
    end

    // Wide payload registers are protected by r_data_valid/r_A_stage_valid;
    // only those validity bits need reset. This keeps the reset network out
    // of the 4x4 broadcast and activation-shift datapaths.
    always @(posedge clk) begin
        r_stationary <= next_stationary;
        // Payload is ignored whenever r_data_valid is low, so retain it
        // outside the two issue states.  The former default-zero assignment
        // placed the state decoder on every D/S pin of this 1024-bit register
        // and dominated routing at 300 MHz.
        if (state == ST_LOAD_STAT)
            r_data_bus_all <= w_B_packed_all;
        else if ((state == ST_STREAM) && r_A_stage_valid)
            r_data_bus_all <= w_A_compressed_all;
        if (AUTO_BENES_CONFIG != 0)
            r_benes_stream <= w_auto_benes_config;
        else
            r_benes_stream <= i_benes_config;
        if (state == ST_LOAD_STAT) begin
            // Reload activations for every output-tile group. Subsequent
            // fixed shifts replace a 16:1 row-select mux on the data path.
            r_A_shift <= r_A_mat;
            r_A_bitmap_shift <= r_A_bitmap;
        end else if (state == ST_STREAM && (m_cnt < r_M)) begin
            r_A_row_data <= r_A_shift[0 +: MAX_K*IN_DATA_TYPE];
            r_A_row_bitmap <= r_A_bitmap_shift[0 +: MAX_K];
            r_A_shift <= r_A_shift >> (MAX_K*IN_DATA_TYPE);
            r_A_bitmap_shift <= r_A_bitmap_shift >> MAX_K;
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            r_data_valid <= 1'b0;
            r_A_stage_valid <= 1'b0;
        end else begin
            r_data_valid <= next_data_valid;
            if (state == ST_STREAM && (m_cnt < r_M))
                r_A_stage_valid <= 1'b1;
            else
                r_A_stage_valid <= 1'b0;
        end
    end

    wire [NUM_DPE*NUM_PES-1:0] w_out_valid_all;
    wire [NUM_DPE*NUM_PES*OUT_DATA_TYPE-1:0] w_out_bus_all;
    wire [NUM_DPE*NUM_PES*LOG2_PES-1:0] w_out_vn_all;

    reg [RESULT_TAG_LATENCY-1:0] result_tag_pipe;
    wire real_row_launch = (state == ST_STREAM) && r_A_stage_valid;
    always @(posedge clk) begin
        if (rst || (state == ST_LOAD_STAT))
            result_tag_pipe <= 0;
        else
            result_tag_pipe <= {result_tag_pipe[RESULT_TAG_LATENCY-2:0], real_row_launch};
    end

    wire [NUM_DPE-1:0] w_dpe_C_valid;
    genvar od;
    generate
        for (od = 0; od < NUM_DPE; od = od + 1) begin : g_tile_output
            reg [NUM_PES-1:0] d_seen;
            reg d_valid;
            integer lane, check, vn;
            always @(*) begin
                d_seen = 0;
                for (lane = 0; lane < NUM_PES; lane = lane + 1) begin
                    if (w_out_valid_all[od*NUM_PES + lane]) begin
                        vn = w_out_vn_all[(od*NUM_PES+lane)*LOG2_PES +: LOG2_PES];
                        if (vn < w_tile_N_all[od*8 +: 8])
                            d_seen[vn] = 1'b1;
                    end
                end
                d_valid = w_tile_active[od];
                for (check = 0; check < NUM_PES; check = check + 1)
                    if ((check < w_tile_N_all[od*8 +: 8]) && !d_seen[check])
                        d_valid = 1'b0;
            end
            assign w_dpe_C_valid[od] = d_valid;
        end
    endgenerate

    reg raw_group_valid;
    integer valid_d;
    always @(*) begin
        raw_group_valid = 1'b1;
        for (valid_d = 0; valid_d < NUM_DPE; valid_d = valid_d + 1)
            if (w_tile_active[valid_d] && !w_dpe_C_valid[valid_d])
                raw_group_valid = 1'b0;
    end
    wire w_group_C_valid = raw_group_valid &&
                           result_tag_pipe[RESULT_TAG_LATENCY-1];

    // One narrow RAM per output column: every bank has exactly one write port
    // and one synchronous read port. This preserves tile-major stationary
    // reuse while avoiding the multi-write 3-D C store that Vivado dissolved
    // into 16k flip-flops.
    wire [MAX_N*OUT_DATA_TYPE-1:0] w_C_bank_q;
    localparam integer C_BANK_AW = $clog2(MAX_M);
    wire [C_BANK_AW-1:0] c_bank_wr_addr = result_count[C_BANK_AW-1:0];
    wire [C_BANK_AW-1:0] c_bank_rd_addr = output_row[C_BANK_AW-1:0];

    // Select one virtual-neuron result from the physical FAN lanes of a
    // known DPE.  Every call below receives constant-width, constant-position
    // slices, so it becomes a four-way 32-bit mux rather than a scatter into a
    // 128/1024-bit temporary row.
    function [OUT_DATA_TYPE:0] select_vn_result;
        input [NUM_PES-1:0] lane_valid;
        input [NUM_PES*OUT_DATA_TYPE-1:0] lane_data;
        input [NUM_PES*LOG2_PES-1:0] lane_tag;
        input [LOG2_PES-1:0] wanted_tag;
        integer select_lane;
        begin
            select_vn_result = 0;
            for (select_lane = 0; select_lane < NUM_PES;
                 select_lane = select_lane + 1) begin
                if (lane_valid[select_lane] &&
                    (lane_tag[select_lane*LOG2_PES +: LOG2_PES] == wanted_tag)) begin
                    select_vn_result[OUT_DATA_TYPE] = 1'b1;
                    select_vn_result[OUT_DATA_TYPE-1:0] =
                        lane_data[select_lane*OUT_DATA_TYPE +: OUT_DATA_TYPE];
                end
            end
        end
    endfunction

    // Decode every DPE/VN pair once and share it across all 16 result banks.
    // Without this layer, the K=1 and K=2 bank mappings instantiate many
    // nearly identical lane selectors; only 32 unique DPE/VN pairs exist.
    wire [NUM_DPE*NUM_PES*(OUT_DATA_TYPE+1)-1:0] w_vn_result_all;
    reg  [NUM_DPE*NUM_PES*(OUT_DATA_TYPE+1)-1:0] r_vn_result_all;
    reg r_group_C_valid;
    genvar decode_dpe, decode_vn;
    generate
        for (decode_dpe = 0; decode_dpe < NUM_DPE;
             decode_dpe = decode_dpe + 1) begin : g_decode_dpe
            for (decode_vn = 0; decode_vn < NUM_PES;
                 decode_vn = decode_vn + 1) begin : g_decode_vn
                assign w_vn_result_all[
                    (decode_dpe*NUM_PES+decode_vn)*(OUT_DATA_TYPE+1) +:
                    (OUT_DATA_TYPE+1)] = select_vn_result(
                        w_out_valid_all[decode_dpe*NUM_PES +: NUM_PES],
                        w_out_bus_all[decode_dpe*NUM_PES*OUT_DATA_TYPE +:
                                      NUM_PES*OUT_DATA_TYPE],
                        w_out_vn_all[decode_dpe*NUM_PES*LOG2_PES +:
                                     NUM_PES*LOG2_PES],
                        decode_vn[LOG2_PES-1:0]);
            end
        end
    endgenerate

    // Isolate the FAN/tag decoder from the C-bank write enables and BRAM
    // inputs.  This register is also a clean timing boundary between the
    // compute fabric and the result memory subsystem.
    always @(posedge clk) begin
        // The decoded payload is meaningful only when r_group_C_valid is set.
        // Leaving the 2112-bit data register reset-free avoids a large reset
        // tree and lets the placer pack it with neighbouring datapath logic.
        r_vn_result_all <= w_vn_result_all;
        if (rst || (state == ST_LOAD_STAT)) begin
            r_group_C_valid <= 1'b0;
        end else begin
            r_group_C_valid <= w_group_C_valid;
        end
    end

    genvar bank_col;
    generate
        for (bank_col = 0; bank_col < MAX_N; bank_col = bank_col + 1) begin : g_C_bank
            localparam integer K2_CAPACITY = NUM_PES / 2;
            localparam integer OWNER_K1 = bank_col / NUM_PES;
            localparam integer LOCAL_K1 = bank_col % NUM_PES;
            localparam integer OWNER_K2 = bank_col / K2_CAPACITY;
            localparam integer LOCAL_K2 = bank_col % K2_CAPACITY;
            localparam integer OWNER_WIDE = bank_col % NUM_DPE;
            localparam integer GROUP_WIDE = (bank_col / NUM_DPE) * NUM_DPE;

            wire [OUT_DATA_TYPE:0] result_k1 = r_vn_result_all[
                (OWNER_K1*NUM_PES+LOCAL_K1)*(OUT_DATA_TYPE+1) +:
                (OUT_DATA_TYPE+1)];
            wire [OUT_DATA_TYPE:0] result_k2 = r_vn_result_all[
                (OWNER_K2*NUM_PES+LOCAL_K2)*(OUT_DATA_TYPE+1) +:
                (OUT_DATA_TYPE+1)];
            wire [OUT_DATA_TYPE:0] result_wide = r_vn_result_all[
                (OWNER_WIDE*NUM_PES)*(OUT_DATA_TYPE+1) +:
                (OUT_DATA_TYPE+1)];

            reg [OUT_DATA_TYPE:0] selected_result;
            reg selected_group;
            always @(*) begin
                selected_result = 0;
                selected_group = 1'b0;
                case (r_K)
                    8'd1: begin
                        selected_result = result_k1;
                        selected_group = (tile_idx == 0);
                    end
                    8'd2: begin
                        selected_result = result_k2;
                        selected_group = (tile_idx == 0);
                    end
                    default: begin
                        selected_result = result_wide;
                        selected_group = (tile_idx == GROUP_WIDE);
                    end
                endcase
            end
            reg local_col_active;
            wire bank_wr_en =
                ((state == ST_STREAM) || (state == ST_COLLECT)) &&
                r_group_C_valid && (result_count < r_M) &&
                local_col_active && selected_group &&
                selected_result[OUT_DATA_TYPE];
            wire [OUT_DATA_TYPE-1:0] bank_wr_data =
                selected_result[OUT_DATA_TYPE-1:0];
            // Isolate the replicated result-valid/dimension decoder from the
            // distributed-RAM WE and data pins.  The pending write completes
            // one edge later, while ST_OUTPUT_REQ/ST_OUTPUT_MEM already leave
            // enough time before the first read.
            reg local_wr_en;
            reg [C_BANK_AW-1:0] local_wr_addr;
            reg [OUT_DATA_TYPE-1:0] local_wr_data;
            // Register the shared row address at each bank.  The old direct
            // output_row -> 16 RAM address fanout was dominated by routing and
            // appeared in the post-route critical paths.
            reg [C_BANK_AW-1:0] local_rd_addr;
            always @(posedge clk) begin
                if (rst) begin
                    local_wr_en <= 1'b0;
                    local_col_active <= 1'b0;
                end else begin
                    local_wr_en <= bank_wr_en;
                    // N is constant for the complete job. Capture the
                    // per-bank active bit at command acceptance instead of
                    // routing r_N into every distributed-RAM input cone.
                    if ((state == ST_IDLE) && i_start && w_dimensions_valid)
                        local_col_active <= (bank_col < i_N);
                    if (bank_wr_en) begin
                        local_wr_addr <= c_bank_wr_addr;
                        local_wr_data <= bank_wr_data;
                    end
                end
                if (state == ST_OUTPUT_REQ)
                    local_rd_addr <= c_bank_rd_addr;
            end
            sigma_sync_ram #(
                .WIDTH(OUT_DATA_TYPE), .DEPTH(MAX_M),
                .ADDR_WIDTH(C_BANK_AW), .RAM_STYLE(RESULT_RAM_STYLE)
            ) u_C_bank (
                .clk(clk), .wr_en(local_wr_en), .wr_addr(local_wr_addr),
                .wr_data(local_wr_data), .rd_en(state == ST_OUTPUT_MEM),
                .rd_addr(local_rd_addr),
                .rd_data(w_C_bank_q[bank_col*OUT_DATA_TYPE +: OUT_DATA_TYPE])
            );
        end
    endgenerate

    integer out_c, stream_d;
    always @(*) begin
        o_C_row = 0;
        o_C_valid = 1'b0;
        if ((state == ST_OUTPUT) && (output_row < r_M)) begin
            for (out_c = 0; out_c < MAX_N; out_c = out_c + 1)
                // Consumers already qualify columns with their registered N.
                // Do not put r_N on every bit of this 512-bit result bus: the
                // unused banks are architecturally ignored and masking them
                // here created the worst r_N -> sigma_C_row_q route.
                o_C_row[out_c*OUT_DATA_TYPE +: OUT_DATA_TYPE] =
                    w_C_bank_q[out_c*OUT_DATA_TYPE +: OUT_DATA_TYPE];
            o_C_valid = 1'b1;
        end
    end

    // Job payload storage has no architectural reset requirement.  Each item
    // is completely written after a valid start and before its first use;
    // state/valid registers below remain resettable and prevent stale payload
    // from being observed.  Keeping reset off these wide registers materially
    // reduces reset fanout and control-set pressure in the 4x4 implementation.
    always @(posedge clk) begin
        if ((state == ST_IDLE) && i_start && w_dimensions_valid) begin
            r_A_mat <= i_A_mat;
            r_A_bitmap <= i_A_bitmap;
            r_B_shift <= i_B_mat;
            r_B_bitmap <= i_B_bitmap;
            r_mesh_route_enable <= i_mesh_route_enable;
            r_mesh_route_config <= i_mesh_route_config;
            r_mesh_hops <= i_mesh_hops;
        end else if ((state == ST_PREP_DESC) && (NUM_PES != 2)) begin
            // Dense CNN folds arrive as one ordered NUM_PES-wide vector per
            // DPE.  Capture/advance the complete vector in one edge instead
            // of walking lanes serially.  Inactive tail DPEs are explicitly
            // zeroed so a short N tile cannot observe the following slice.
            if (prep_active) begin
                r_current_B_mat <=
                    r_B_shift[0 +: NUM_PES*IN_DATA_TYPE];
                r_current_B_bitmap <= r_B_bitmap[0 +: NUM_PES];
            end else begin
                r_current_B_mat <= 0;
                r_current_B_bitmap <= 0;
            end
            r_B_shift <= r_B_shift >> (NUM_PES*IN_DATA_TYPE);
            r_B_bitmap <= r_B_bitmap >> NUM_PES;
        end else if (state == ST_PREP_B) begin
            if (prep_element_valid) begin
                r_current_B_mat[prep_lane*IN_DATA_TYPE +: IN_DATA_TYPE] <=
                    r_B_shift[0 +: IN_DATA_TYPE];
                r_current_B_bitmap[prep_lane] <= r_B_bitmap[0];
            end else begin
                r_current_B_mat[prep_lane*IN_DATA_TYPE +: IN_DATA_TYPE] <= 0;
                r_current_B_bitmap[prep_lane] <= 1'b0;
            end
            r_B_shift <= r_B_shift >> IN_DATA_TYPE;
            r_B_bitmap <= r_B_bitmap >> 1;
        end

        if (state == ST_LOAD_STAT)
            r_vn_all <= w_B_vn_all;
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            o_done <= 1'b0;
            o_error <= 1'b0;
            wait_cnt <= 0;
            m_cnt <= 0;
            result_count <= 0;
            tile_idx <= 0;
            output_row <= 0;
            prep_dpe <= 0;
            prep_lane <= 0;
            r_M <= 0;
            r_N <= 0;
            r_K <= 0;
            r_tile_count <= 0;
            r_prep_active <= 1'b0;
            r_prep_tile_N <= 0;
            r_prep_tile_base <= 0;
            r_prep_valid_lanes <= 0;
            r_prep_A_regor <= 0;
        end else begin
            if (((state == ST_STREAM) || (state == ST_COLLECT)) &&
                r_group_C_valid && (result_count < r_M)) begin
                result_count <= result_count + 1'b1;
            end

            case (state)
                ST_IDLE: begin
                    o_done <= 1'b0;
                    o_error <= 1'b0;
                    wait_cnt <= 0;
                    m_cnt <= 0;
                    result_count <= 0;
                    tile_idx <= 0;
                    output_row <= 0;
                    if (i_start) begin
                        if (w_dimensions_valid) begin
                            r_M <= i_M;
                            r_N <= i_N;
                            r_K <= i_K;
                            r_tile_count <= tile_count_for(i_N, i_K);
                            prep_dpe <= 0;
                            prep_lane <= 0;
                            state <= ST_PREP_DESC;
                        end else begin
                            o_error <= 1'b1;
                            state <= ST_DONE;
                        end
                    end
                end

                ST_PREP_DESC: begin
                    // Pipeline the tile scheduler before the wide B lane
                    // registers. This removes tile_idx arithmetic from their
                    // D/CE paths without changing the stationary data layout.
                    r_prep_active <= prep_active;
                    r_prep_tile_N <= prep_tile_N;
                    r_prep_tile_base <= prep_tile_base;
                    // The sparse release profile permits only K=1 or K=2.
                    // Register its lane count at the descriptor boundary so
                    // the following wide B write enables see only a small
                    // comparator, not an inferred runtime DSP multiplier.
                    // Wider dense profiles bypass ST_PREP_B, so this value is
                    // deliberately limited to the verified two-PE schedule.
                    if (NUM_PES == 2) begin
                        if (r_K == 1)
                            r_prep_valid_lanes <= prep_tile_N;
                        else
                            r_prep_valid_lanes <= prep_tile_N << 1;
                    end else begin
                        r_prep_valid_lanes <= NUM_PES;
                    end
                    // The two-lane sparse path must inspect/pack its lanes
                    // serially.  Wide CNN folds are already dense, ordered
                    // vectors, so the per-lane walk only burns NUM_PES cycles
                    // without changing any bit.  Bypass it for MobileNet and
                    // AlexNet while preserving the exact LeNet/GEMM path.
                    if (NUM_PES == 2)
                        state <= ST_PREP_B;
                    else
                        state <= ST_PACK_META;
                end

                ST_PREP_B: begin
                    if (prep_lane + 1'b1 >= NUM_PES) begin
                        prep_lane <= 0;
                        state <= ST_PACK_META;
                    end else begin
                        prep_lane <= prep_lane + 1'b1;
                    end
                end

                ST_PACK_META: begin
                    // The stationary bitmap is complete only after all lanes
                    // arrive, so derive REGOR here from registered data.
                    if (NUM_PES == 2)
                        r_prep_A_regor <=
                            regor2(r_current_B_bitmap, r_K, prep_tile_N);
                    else
                        r_prep_A_regor <= {NUM_PES{1'b1}};
                    state <= ST_PACK_B;
                end

                ST_PACK_B: begin
                    r_tile_active[prep_dpe] <= r_prep_active;
                    r_tile_N[prep_dpe] <= r_prep_tile_N;
                    r_tile_base[prep_dpe] <= r_prep_tile_base;
                    r_A_regor[prep_dpe] <= r_prep_A_regor;
                    r_B_stationary_bitmap[prep_dpe] <= r_current_B_bitmap;
                    for (stream_d = 0; stream_d < NUM_PES;
                         stream_d = stream_d + 1) begin
                        r_B_packed_data[prep_dpe*NUM_PES + stream_d] <=
                            shared_B_packed[stream_d*IN_DATA_TYPE +: IN_DATA_TYPE];
                        r_B_packed_tag[prep_dpe*NUM_PES + stream_d] <=
                            shared_B_tags[stream_d*LOG2_PES +: LOG2_PES];
                    end
                    prep_lane <= 0;
                    if (prep_dpe + 1'b1 >= NUM_DPE) begin
                        prep_dpe <= 0;
                        state <= ST_LOAD_STAT;
                    end else begin
                        prep_dpe <= prep_dpe + 1'b1;
                        state <= ST_PREP_DESC;
                    end
                end

                ST_LOAD_STAT: begin
                    wait_cnt <= 0;
                    state <= ST_WAIT_PIPE;
                end

                ST_WAIT_PIPE: begin
                    if (wait_cnt == 7) begin
                        m_cnt <= 0;
                        state <= ST_STREAM;
                    end else
                        wait_cnt <= wait_cnt + 1'b1;
                end

                ST_STREAM: begin
                    if (m_cnt < r_M)
                        m_cnt <= m_cnt + 1'b1;
                    else if (!r_A_stage_valid) begin
                        m_cnt <= 0;
                        state <= ST_COLLECT;
                    end
                end

                ST_COLLECT: begin
                    if ((result_count >= r_M) ||
                        (r_group_C_valid && (result_count + 1'b1 >= r_M))) begin
                        result_count <= 0;
                        m_cnt <= 0;
                        if ((tile_idx + NUM_DPE) < w_tile_count) begin
                            tile_idx <= tile_idx + NUM_DPE;
                            prep_dpe <= 0;
                            prep_lane <= 0;
                            state <= ST_PREP_DESC;
                        end else begin
                            output_row <= 0;
                            state <= ST_OUTPUT_REQ;
                        end
                    end
                end

                ST_OUTPUT_REQ: begin
                    state <= ST_OUTPUT_MEM;
                end

                ST_OUTPUT_MEM: begin
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    if (output_row + 1'b1 >= r_M)
                        state <= ST_DONE;
                    else begin
                        output_row <= output_row + 1'b1;
                        state <= ST_OUTPUT_REQ;
                    end
                end

                ST_DONE: begin
                    o_done <= 1'b1;
                    if (!i_start)
                        state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    flexdpu #(
        .IN_DATA_TYPE(IN_DATA_TYPE),
        .OUT_DATA_TYPE(OUT_DATA_TYPE),
        .NUM_PES(NUM_PES),
        .LOG2_PES(LOG2_PES),
        .FAN_ADD_PIPE_STAGES(FAN_ADD_PIPE_STAGES),
        .MESH_ROWS(MESH_ROWS),
        .MESH_COLS(MESH_COLS),
        .NUM_DPE(NUM_DPE),
        .MESH_ROUTE_WIDTH(MESH_ROUTE_WIDTH),
        .MESH_HOP_WIDTH(MESH_HOP_WIDTH),
        .BENES_CTRL_WIDTH(BENES_CTRL_WIDTH)
    ) u_flexdpu (
        .clk(clk),
        .rst(rst),
        .i_data_valid(r_data_valid),
        .i_stationary(r_stationary),
        .i_mesh_route_enable(r_mesh_route_enable),
        .i_mesh_route_config(r_mesh_route_config),
        .i_mesh_hops(r_mesh_hops),
        .i_data_bus(r_data_bus_all),
        // Benes control is a row-synchronous stream.  Delay it by the same
        // input stage as the activation so each sparse row keeps its route.
        .i_benes_bus(r_benes_stream),
        .i_vn_seperator(r_vn_all),
        .o_data_valid(w_out_valid_all),
        .o_data_bus(w_out_bus_all),
        .o_vn_bus(w_out_vn_all)
    );
endmodule
