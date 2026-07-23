`timescale 1ns / 1ps
// Four-lane dot-product pipeline shared by the MobileNet fast paths.
//
// Each lane consumes four BF16 products per accepted fold.  The products use
// the same pipelined BF16 multiplier and SIGMA forwarding-adder network (FAN)
// as a Flex-DPE.  Depthwise uses three folds for K=10; M=1 pointwise layers
// use up to 64 folds for K<=256.  Four independent dot products are evaluated
// together without duplicating the full Benes distribution network or mesh.
module sigma_mobilenet_dot4 (
    input  wire         clk,
    input  wire         rst,
    input  wire         i_start,
    input  wire [9:0]   i_fold_count,
    input  wire         i_valid,
    input  wire [255:0] i_a_bus,
    input  wire [255:0] i_b_bus,
    output reg          o_busy,
    output reg          o_valid,
    output reg  [127:0] o_result_bus
);
    localparam LANES = 4;
    localparam PES = 4;
    localparam MULT_LATENCY = 8;
    localparam ADD_LATENCY = 6;
    // A feedback result is visible to this controller seven edges after its
    // operands are launched.  Eight independent stripes therefore let the
    // existing four adders accept one fold every clock without a RAW hazard.
    localparam ACC_PIPE_LATENCY = 7;
    localparam ACC_STRIPES = 8;

    wire [LANES*PES*16-1:0] product_bf16;
    reg  [MULT_LATENCY-1:0] mult_valid_pipe;

    genvar lane, pe;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            for (pe = 0; pe < PES; pe = pe + 1) begin : g_pe
                localparam integer SLOT = lane*PES + pe;
                bf16_mult_pipeline u_mult (
                    .clk(clk),
                    .a(i_a_bus[SLOT*16 +: 16]),
                    .b(i_b_bus[SLOT*16 +: 16]),
                    .out(product_bf16[SLOT*16 +: 16])
                );
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst || i_start)
            mult_valid_pipe <= 0;
        else
            mult_valid_pipe <= {mult_valid_pipe[MULT_LATENCY-2:0], i_valid};
    end

    wire [PES-1:0] fan_valid [0:LANES-1];
    wire [PES*32-1:0] fan_data [0:LANES-1];
    wire [PES*2-1:0] fan_tags [0:LANES-1];
    wire [PES*32-1:0] lane_products [0:LANES-1];

    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_fan
            for (pe = 0; pe < PES; pe = pe + 1) begin : g_expand
                assign lane_products[lane][pe*32 +: 32] =
                    {product_bf16[(lane*PES+pe)*16 +: 16], 16'h0000};
            end
            fan_network #(
                .DATA_TYPE(32), .NUM_PES(PES), .LOG2_PES(2),
                .ADD_PIPE_STAGES(ADD_LATENCY)
            ) u_fan (
                .clk(clk), .rst(rst),
                .i_valid(mult_valid_pipe[MULT_LATENCY-1]),
                .i_data_bus(lane_products[lane]),
                .i_vn_bus({PES*2{1'b0}}),
                .o_valid(fan_valid[lane]),
                .o_data_bus(fan_data[lane]),
                .o_vn_bus(fan_tags[lane])
            );
        end
    endgenerate

    localparam ST_IDLE = 3'd0;
    localparam ST_COLLECT = 3'd1;
    localparam ST_ACC_INIT = 3'd2;
    localparam ST_ACC_LAUNCH = 3'd3;
    localparam ST_ACC_WAIT = 3'd4;
    localparam ST_DONE = 3'd5;

    reg [2:0] state;
    reg [9:0] fold_count;
    reg [9:0] partial_count;
    // Final stripe reduction uses two adders per output lane.  The primary
    // adder is shared with interleaved fold accumulation; a second adder lets
    // two independent stripe pairs collapse in parallel after collection.
    // This reduces eight stripes in four add-latency rounds instead of seven
    // without duplicating the multiplier/FAN datapath.
    reg [2:0] reduce_index;
    reg [3:0] reduce_count;
    reg [31:0] stripe_accum [0:LANES*ACC_STRIPES-1];
    reg [31:0] accumulator [0:LANES-1];
    reg [31:0] add_a [0:LANES-1];
    reg [31:0] add_b [0:LANES-1];
    wire [31:0] add_out [0:LANES-1];
    reg [31:0] reduce2_a [0:LANES-1];
    reg [31:0] reduce2_b [0:LANES-1];
    wire [31:0] reduce2_out [0:LANES-1];
    reg reduce_second_valid;
    reg [3:0] add_wait;
    reg [ACC_PIPE_LATENCY-1:0] fold_add_valid_pipe;
    reg [2:0] fold_add_stripe_pipe [0:ACC_PIPE_LATENCY-1];
    integer capture_lane;
    integer add_pipe_index;

    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_fold_add
            fp32_add_pipeline u_fold_add (
                .clk(clk), .rst(rst),
                .a(add_a[lane]), .b(add_b[lane]), .out(add_out[lane])
            );
            fp32_add_pipeline u_reduce2_add (
                .clk(clk), .rst(rst),
                .a(reduce2_a[lane]), .b(reduce2_b[lane]),
                .out(reduce2_out[lane])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            partial_count <= 0;
            add_wait <= 0;
            o_busy <= 0;
            o_valid <= 0;
            o_result_bus <= 0;
            fold_add_valid_pipe <= 0;
            reduce_second_valid <= 0;
            for (add_pipe_index = 0; add_pipe_index < ACC_PIPE_LATENCY;
                 add_pipe_index = add_pipe_index + 1)
                fold_add_stripe_pipe[add_pipe_index] <= 0;
        end else begin
            o_valid <= 0;
            for (add_pipe_index = ACC_PIPE_LATENCY-1; add_pipe_index > 0;
                 add_pipe_index = add_pipe_index - 1) begin
                fold_add_valid_pipe[add_pipe_index] <=
                    fold_add_valid_pipe[add_pipe_index-1];
                fold_add_stripe_pipe[add_pipe_index] <=
                    fold_add_stripe_pipe[add_pipe_index-1];
            end
            fold_add_valid_pipe[0] <= 1'b0;

            if (fold_add_valid_pipe[ACC_PIPE_LATENCY-1]) begin
                for (capture_lane = 0; capture_lane < LANES;
                     capture_lane = capture_lane + 1)
                    stripe_accum[capture_lane*ACC_STRIPES +
                        fold_add_stripe_pipe[ACC_PIPE_LATENCY-1]] <=
                            add_out[capture_lane];
            end
            case (state)
                ST_IDLE: begin
                    o_busy <= 0;
                    if (i_start) begin
                        partial_count <= 0;
                        fold_count <= (i_fold_count == 0) ? 1 : i_fold_count;
                        fold_add_valid_pipe <= 0;
                        o_busy <= 1;
                        state <= ST_COLLECT;
                    end
                end

                ST_COLLECT: begin
                    // All four FANs receive the same valid schedule.  Lane 0
                    // therefore serves as the completion token for the group.
                    if (fan_valid[0][0]) begin
                        if (partial_count < ACC_STRIPES) begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1)
                                stripe_accum[capture_lane*ACC_STRIPES +
                                    partial_count[2:0]] <=
                                        fan_data[capture_lane][31:0];
                        end else begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1) begin
                                add_a[capture_lane] <=
                                    stripe_accum[capture_lane*ACC_STRIPES +
                                        partial_count[2:0]];
                                add_b[capture_lane] <=
                                    fan_data[capture_lane][31:0];
                            end
                            fold_add_valid_pipe[0] <= 1'b1;
                            fold_add_stripe_pipe[0] <= partial_count[2:0];
                        end
                        if (partial_count + 1 >= fold_count) begin
                            partial_count <= 0;
                            state <= ST_ACC_INIT;
                        end else
                            partial_count <= partial_count + 1'b1;
                    end
                end

                ST_ACC_INIT: begin
                    // Drain the last interleaved feedback operation before
                    // reducing at most eight stripe totals to one result.
                    if (fold_add_valid_pipe == 0) begin
                        for (capture_lane = 0; capture_lane < LANES;
                             capture_lane = capture_lane + 1)
                            accumulator[capture_lane] <=
                                stripe_accum[capture_lane*ACC_STRIPES];
                        reduce_count <= (fold_count < ACC_STRIPES) ?
                                        fold_count[3:0] : ACC_STRIPES;
                        reduce_index <= 0;
                        if (fold_count == 1) begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1)
                                o_result_bus[capture_lane*32 +: 32] <=
                                    stripe_accum[capture_lane*ACC_STRIPES];
                            state <= ST_DONE;
                        end else
                            state <= ST_ACC_LAUNCH;
                    end
                end

                ST_ACC_LAUNCH: begin
                    for (capture_lane = 0; capture_lane < LANES;
                         capture_lane = capture_lane + 1) begin
                        add_a[capture_lane] <=
                            stripe_accum[capture_lane*ACC_STRIPES +
                                         (reduce_index << 1)];
                        add_b[capture_lane] <=
                            stripe_accum[capture_lane*ACC_STRIPES +
                                         (reduce_index << 1) + 1'b1];
                        if (reduce_index + 1 < (reduce_count >> 1)) begin
                            reduce2_a[capture_lane] <=
                                stripe_accum[capture_lane*ACC_STRIPES +
                                             ((reduce_index + 1'b1) << 1)];
                            reduce2_b[capture_lane] <=
                                stripe_accum[capture_lane*ACC_STRIPES +
                                             ((reduce_index + 1'b1) << 1) + 1'b1];
                        end else begin
                            reduce2_a[capture_lane] <= 0;
                            reduce2_b[capture_lane] <= 0;
                        end
                    end
                    reduce_second_valid <=
                        (reduce_index + 1 < (reduce_count >> 1));
                    add_wait <= 0;
                    state <= ST_ACC_WAIT;
                end

                ST_ACC_WAIT: begin
                    if (add_wait == ADD_LATENCY) begin
                        for (capture_lane = 0; capture_lane < LANES;
                             capture_lane = capture_lane + 1) begin
                            stripe_accum[capture_lane*ACC_STRIPES +
                                         reduce_index] <= add_out[capture_lane];
                            if (reduce_second_valid)
                                stripe_accum[capture_lane*ACC_STRIPES +
                                             reduce_index + 1'b1] <=
                                                 reduce2_out[capture_lane];
                        end
                        if (reduce_index + 2 < (reduce_count >> 1)) begin
                            // The following pair batch reads untouched stripe
                            // slots, so launch it while capturing this batch
                            // instead of spending an extra FSM bubble.
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1) begin
                                add_a[capture_lane] <=
                                    stripe_accum[capture_lane*ACC_STRIPES +
                                        ((reduce_index + 2) << 1)];
                                add_b[capture_lane] <=
                                    stripe_accum[capture_lane*ACC_STRIPES +
                                        ((reduce_index + 2) << 1) + 1'b1];
                                if (reduce_index + 3 < (reduce_count >> 1)) begin
                                    reduce2_a[capture_lane] <=
                                        stripe_accum[capture_lane*ACC_STRIPES +
                                            ((reduce_index + 3) << 1)];
                                    reduce2_b[capture_lane] <=
                                        stripe_accum[capture_lane*ACC_STRIPES +
                                            ((reduce_index + 3) << 1) + 1'b1];
                                end else begin
                                    reduce2_a[capture_lane] <= 0;
                                    reduce2_b[capture_lane] <= 0;
                                end
                            end
                            reduce_second_valid <=
                                (reduce_index + 3 < (reduce_count >> 1));
                            reduce_index <= reduce_index + 2;
                            add_wait <= 0;
                            state <= ST_ACC_WAIT;
                        end else if (((reduce_count >> 1) +
                                      reduce_count[0]) == 1) begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1)
                                o_result_bus[capture_lane*32 +: 32] <=
                                    add_out[capture_lane];
                            state <= ST_DONE;
                        end else begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1) begin
                                if (reduce_count[0])
                                    stripe_accum[capture_lane*ACC_STRIPES +
                                                 (reduce_count >> 1)] <=
                                        stripe_accum[capture_lane*ACC_STRIPES +
                                                     reduce_count - 1'b1];
                            end
                            reduce_count <= (reduce_count >> 1) +
                                            reduce_count[0];
                            reduce_index <= 0;
                            state <= ST_ACC_LAUNCH;
                        end
                    end else
                        add_wait <= add_wait + 1'b1;
                end

                ST_DONE: begin
                    o_busy <= 0;
                    o_valid <= 1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
