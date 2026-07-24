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
    input  wire [6:0]   i_fold_count,
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
    reg [6:0] fold_count;
    reg [6:0] partial_count;
    reg [6:0] reduce_index;
    (* ram_style = "distributed" *) reg [31:0] partial [0:LANES*64-1];
    reg [31:0] accumulator [0:LANES-1];
    reg [31:0] add_a [0:LANES-1];
    reg [31:0] add_b [0:LANES-1];
    wire [31:0] add_out [0:LANES-1];
    reg [3:0] add_wait;
    integer capture_lane;

    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_fold_add
            fp32_add_pipeline u_fold_add (
                .clk(clk), .rst(rst),
                .a(add_a[lane]), .b(add_b[lane]), .out(add_out[lane])
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
        end else begin
            o_valid <= 0;
            case (state)
                ST_IDLE: begin
                    o_busy <= 0;
                    if (i_start) begin
                        partial_count <= 0;
                        fold_count <= (i_fold_count == 0) ? 1 : i_fold_count;
                        o_busy <= 1;
                        state <= ST_COLLECT;
                    end
                end

                ST_COLLECT: begin
                    // All four FANs receive the same valid schedule.  Lane 0
                    // therefore serves as the completion token for the group.
                    if (fan_valid[0][0]) begin
                        for (capture_lane = 0; capture_lane < LANES;
                            capture_lane = capture_lane + 1)
                            partial[capture_lane*64 + partial_count] <=
                                fan_data[capture_lane][31:0];
                        if (partial_count + 1 >= fold_count) begin
                            partial_count <= 0;
                            state <= ST_ACC_INIT;
                        end else
                            partial_count <= partial_count + 1'b1;
                    end
                end

                ST_ACC_INIT: begin
                    for (capture_lane = 0; capture_lane < LANES;
                         capture_lane = capture_lane + 1)
                        accumulator[capture_lane] <=
                            partial[capture_lane*64];
                    reduce_index <= 1;
                    if (fold_count == 1) begin
                        for (capture_lane = 0; capture_lane < LANES;
                             capture_lane = capture_lane + 1)
                            o_result_bus[capture_lane*32 +: 32] <=
                                partial[capture_lane*64];
                        state <= ST_DONE;
                    end else
                        state <= ST_ACC_LAUNCH;
                end

                ST_ACC_LAUNCH: begin
                    for (capture_lane = 0; capture_lane < LANES;
                         capture_lane = capture_lane + 1) begin
                        add_a[capture_lane] <= accumulator[capture_lane];
                        add_b[capture_lane] <=
                            partial[capture_lane*64 + reduce_index];
                    end
                    add_wait <= 0;
                    state <= ST_ACC_WAIT;
                end

                ST_ACC_WAIT: begin
                    if (add_wait == ADD_LATENCY) begin
                        if (reduce_index + 1 >= fold_count) begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1)
                                o_result_bus[capture_lane*32 +: 32] <=
                                    add_out[capture_lane];
                            state <= ST_DONE;
                        end else begin
                            for (capture_lane = 0; capture_lane < LANES;
                                 capture_lane = capture_lane + 1)
                                accumulator[capture_lane] <=
                                    add_out[capture_lane];
                            reduce_index <= reduce_index + 1'b1;
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
