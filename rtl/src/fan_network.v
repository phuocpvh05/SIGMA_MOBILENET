`timescale 1ns / 1ps
//==============================================================================
// Parameterized SIGMA Forwarding Adder Network (FAN).
//
// stationary_compactor places every virtual neuron in an aligned, power-of-two
// buddy block.  At tree level s, the two partial sums at the roots of adjacent
// 2^s sub-blocks are added only when their VN tags match; otherwise both values
// are forwarded.  This is the paper's add-or-forward behavior without the
// original generator's hard-coded wide-lane wiring.
//==============================================================================
module fan_network #(
    parameter DATA_TYPE = 32,
    parameter NUM_PES = 2,
    parameter LOG2_PES = 1,
    parameter ADD_PIPE_STAGES = 6
) (
    input  wire clk,
    input  wire rst,
    input  wire i_valid,
    input  wire [NUM_PES*DATA_TYPE-1:0] i_data_bus,
    input  wire [NUM_PES*LOG2_PES-1:0] i_vn_bus,
    output wire [NUM_PES-1:0] o_valid,
    output wire [NUM_PES*DATA_TYPE-1:0] o_data_bus,
    output wire [NUM_PES*LOG2_PES-1:0] o_vn_bus
);
    wire [NUM_PES*DATA_TYPE-1:0] level_data [0:LOG2_PES];
    wire [NUM_PES*LOG2_PES-1:0] level_tag [0:LOG2_PES];
    wire [NUM_PES-1:0] level_valid [0:LOG2_PES];
    reg [NUM_PES*DATA_TYPE-1:0] input_data;
    reg [NUM_PES*LOG2_PES-1:0] input_tag;
    reg [NUM_PES-1:0] input_valid;

    // Payload and tag are don't-care while valid is low, so only reset the
    // validity plane. Avoiding reset on the wide datapath greatly reduces
    // reset replication and permits efficient shift-register packing.
    always @(posedge clk) begin
        input_data <= i_data_bus;
        input_tag <= i_vn_bus;
    end
    always @(posedge clk) begin
        if (rst)
            input_valid <= 0;
        else
            input_valid <= {NUM_PES{i_valid}};
    end

    assign level_data[0] = input_data;
    assign level_tag[0] = input_tag;
    assign level_valid[0] = input_valid;

    genvar level, lane;
    generate
        for (level = 0; level < LOG2_PES; level = level + 1) begin : g_level
            localparam integer HALF = (1 << level);
            localparam integer GROUP = (2 << level);
            for (lane = 0; lane < NUM_PES; lane = lane + 1) begin : g_lane
                localparam integer OFFSET = lane % GROUP;
                if (OFFSET == 0) begin : g_root
                    localparam integer PARTNER = lane + HALF;
                    localparam integer LEFT_MASK = ((1 << HALF) - 1) << lane;
                    localparam integer RIGHT_MASK = ((1 << HALF) - 1) << PARTNER;
                    wire left_valid = level_valid[level][lane];
                    wire right_valid = level_valid[level][PARTNER];
                    wire left_complete =
                        (level_valid[level] & LEFT_MASK) == (1 << lane);
                    wire right_complete =
                        (level_valid[level] & RIGHT_MASK) == (1 << PARTNER);
                    wire tags_match =
                        level_tag[level][lane*LOG2_PES +: LOG2_PES] ==
                        level_tag[level][PARTNER*LOG2_PES +: LOG2_PES];
                    wire do_add = left_valid && right_valid && tags_match &&
                                  left_complete && right_complete;
                    wire [DATA_TYPE-1:0] sum;
                    wire [DATA_TYPE-1:0] delayed_left;
                    wire [LOG2_PES-1:0] delayed_tag;
                    wire delayed_left_valid;
                    wire delayed_do_add;
                    fp32_add_pipeline u_adder (
                        .clk(clk), .rst(rst),
                        .a(level_data[level][lane*DATA_TYPE +: DATA_TYPE]),
                        .b(level_data[level][PARTNER*DATA_TYPE +: DATA_TYPE]),
                        .out(sum)
                    );
                    fan_delay #(.WIDTH(DATA_TYPE), .STAGES(ADD_PIPE_STAGES)) u_left_delay (
                        .clk(clk), .rst(rst),
                        .i_data(level_data[level][lane*DATA_TYPE +: DATA_TYPE]),
                        .o_data(delayed_left)
                    );
                    fan_delay #(.WIDTH(LOG2_PES), .STAGES(ADD_PIPE_STAGES)) u_tag_delay (
                        .clk(clk), .rst(rst),
                        .i_data(level_tag[level][lane*LOG2_PES +: LOG2_PES]),
                        .o_data(delayed_tag)
                    );
                    fan_delay #(.WIDTH(1), .STAGES(ADD_PIPE_STAGES),
                                .RESETTABLE(1)) u_valid_delay (
                        .clk(clk), .rst(rst), .i_data(left_valid),
                        .o_data(delayed_left_valid)
                    );
                    fan_delay #(.WIDTH(1), .STAGES(ADD_PIPE_STAGES),
                                .RESETTABLE(1)) u_add_delay (
                        .clk(clk), .rst(rst), .i_data(do_add),
                        .o_data(delayed_do_add)
                    );
                    assign level_data[level+1][lane*DATA_TYPE +: DATA_TYPE] =
                        delayed_do_add ? sum : delayed_left;
                    assign level_tag[level+1][lane*LOG2_PES +: LOG2_PES] = delayed_tag;
                    assign level_valid[level+1][lane] = delayed_left_valid;
                end else if (OFFSET == HALF) begin : g_partner_root
                    localparam integer LEFT = lane - HALF;
                    localparam integer LEFT_MASK = ((1 << HALF) - 1) << LEFT;
                    localparam integer RIGHT_MASK = ((1 << HALF) - 1) << lane;
                    wire left_valid = level_valid[level][LEFT];
                    wire this_valid = level_valid[level][lane];
                    wire left_complete =
                        (level_valid[level] & LEFT_MASK) == (1 << LEFT);
                    wire right_complete =
                        (level_valid[level] & RIGHT_MASK) == (1 << lane);
                    wire tags_match =
                        level_tag[level][LEFT*LOG2_PES +: LOG2_PES] ==
                        level_tag[level][lane*LOG2_PES +: LOG2_PES];
                    wire consumed = left_valid && this_valid && tags_match &&
                                    left_complete && right_complete;
                    fan_delay #(.WIDTH(DATA_TYPE), .STAGES(ADD_PIPE_STAGES)) u_data_delay (
                        .clk(clk), .rst(rst),
                        .i_data(level_data[level][lane*DATA_TYPE +: DATA_TYPE]),
                        .o_data(level_data[level+1][lane*DATA_TYPE +: DATA_TYPE])
                    );
                    fan_delay #(.WIDTH(LOG2_PES), .STAGES(ADD_PIPE_STAGES)) u_tag_delay (
                        .clk(clk), .rst(rst),
                        .i_data(level_tag[level][lane*LOG2_PES +: LOG2_PES]),
                        .o_data(level_tag[level+1][lane*LOG2_PES +: LOG2_PES])
                    );
                    fan_delay #(.WIDTH(1), .STAGES(ADD_PIPE_STAGES),
                                .RESETTABLE(1)) u_valid_delay (
                        .clk(clk), .rst(rst), .i_data(this_valid && !consumed),
                        .o_data(level_valid[level+1][lane])
                    );
                end else begin : g_forward
                    fan_delay #(.WIDTH(DATA_TYPE), .STAGES(ADD_PIPE_STAGES)) u_data_delay (
                        .clk(clk), .rst(rst),
                        .i_data(level_data[level][lane*DATA_TYPE +: DATA_TYPE]),
                        .o_data(level_data[level+1][lane*DATA_TYPE +: DATA_TYPE])
                    );
                    fan_delay #(.WIDTH(LOG2_PES), .STAGES(ADD_PIPE_STAGES)) u_tag_delay (
                        .clk(clk), .rst(rst),
                        .i_data(level_tag[level][lane*LOG2_PES +: LOG2_PES]),
                        .o_data(level_tag[level+1][lane*LOG2_PES +: LOG2_PES])
                    );
                    fan_delay #(.WIDTH(1), .STAGES(ADD_PIPE_STAGES),
                                .RESETTABLE(1)) u_valid_delay (
                        .clk(clk), .rst(rst), .i_data(level_valid[level][lane]),
                        .o_data(level_valid[level+1][lane])
                    );
                end
            end
        end
    endgenerate

    assign o_valid = level_valid[LOG2_PES];
    assign o_data_bus = level_data[LOG2_PES];
    assign o_vn_bus = level_tag[LOG2_PES];
endmodule


module fan_delay #(
    parameter WIDTH = 1,
    parameter STAGES = 4,
    parameter RESETTABLE = 0
) (
    input  wire clk,
    input  wire rst,
    input  wire [WIDTH-1:0] i_data,
    output wire [WIDTH-1:0] o_data
);
    reg [WIDTH-1:0] pipe [0:STAGES-1];
    integer stage;
    generate
        if (RESETTABLE) begin : g_resettable_delay
            always @(posedge clk) begin
                if (rst) begin
                    for (stage = 0; stage < STAGES; stage = stage + 1)
                        pipe[stage] <= 0;
                end else begin
                    pipe[0] <= i_data;
                    for (stage = 1; stage < STAGES; stage = stage + 1)
                        pipe[stage] <= pipe[stage-1];
                end
            end
        end else begin : g_payload_delay
            always @(posedge clk) begin
            pipe[0] <= i_data;
            for (stage = 1; stage < STAGES; stage = stage + 1)
                pipe[stage] <= pipe[stage-1];
            end
        end
    endgenerate
    assign o_data = pipe[STAGES-1];
endmodule
