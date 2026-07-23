`timescale 1ns / 1ps
//==========================================================================
// Parameterized Flex-DPU: a 2D grid of Flex-DPE compute nodes.
// The generated hierarchy preserves row/column coordinates for the static
// mesh NoC used by SIGMA and for later Chipyard physical integration.
//==========================================================================

module flexdpu #(
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
    parameter MAX_MESH_HOPS = MESH_ROWS + MESH_COLS - 2,
    parameter BENES_CTRL_WIDTH = 2 * (2 * LOG2_PES - 1) * NUM_PES + NUM_PES
) (
    input  wire clk,
    input  wire rst,
    input  wire i_data_valid,
    input  wire i_stationary,
    input  wire i_mesh_route_enable,
    input  wire [NUM_DPE*MESH_ROUTE_WIDTH-1:0] i_mesh_route_config,
    input  wire [NUM_DPE*MESH_HOP_WIDTH-1:0] i_mesh_hops,
    input  wire [NUM_DPE*NUM_PES*IN_DATA_TYPE-1:0] i_data_bus,
    input  wire [NUM_DPE*BENES_CTRL_WIDTH-1:0] i_benes_bus,
    input  wire [NUM_DPE*NUM_PES*LOG2_PES-1:0] i_vn_seperator,
    output wire [NUM_DPE*NUM_PES-1:0] o_data_valid,
    output wire [NUM_DPE*NUM_PES*OUT_DATA_TYPE-1:0] o_data_bus,
    output wire [NUM_DPE*NUM_PES*LOG2_PES-1:0] o_vn_bus
);
    localparam integer MESH_DATA_WIDTH = NUM_PES * IN_DATA_TYPE;
    wire [NUM_DPE*MESH_DATA_WIDTH-1:0] mesh_data;
    wire [NUM_DPE-1:0] mesh_valid;
    wire [NUM_DPE*MESH_ROUTE_WIDTH-1:0] local_route_config;
    wire [NUM_DPE*MESH_ROUTE_WIDTH-1:0] mesh_route_config =
        i_mesh_route_enable ? i_mesh_route_config : local_route_config;
    reg [NUM_DPE-1:0] r_stationary_pipe [0:MAX_MESH_HOPS];
    reg [NUM_DPE*BENES_CTRL_WIDTH-1:0] r_benes_pipe [0:MAX_MESH_HOPS];
    reg [NUM_DPE*NUM_PES*LOG2_PES-1:0] r_vn_pipe [0:MAX_MESH_HOPS];
    integer control_stage;

    // Current mapped tiles are injected locally at their assigned DPE.  The
    // payload still traverses the same registered static switch used for mesh
    // multicast/hop routing; later mappings may replace this local route.
    genvar route_node;
    generate
        for (route_node = 0; route_node < NUM_DPE; route_node = route_node + 1) begin : g_local_routes
            assign local_route_config[route_node*MESH_ROUTE_WIDTH +: MESH_ROUTE_WIDTH] =
                15'b111_111_111_111_000;
        end
    endgenerate

    sigma_mesh_noc #(
        .MESH_ROWS(MESH_ROWS), .MESH_COLS(MESH_COLS),
        .DATA_WIDTH(MESH_DATA_WIDTH)
    ) u_data_mesh (
        .clk(clk), .rst(rst),
        .i_local_data(i_data_bus),
        .i_local_valid({NUM_DPE{i_data_valid}}),
        .i_route_config(mesh_route_config),
        .o_local_data(mesh_data),
        .o_local_valid(mesh_valid)
    );

    // These controls are consumed only with the separately reset mesh_valid
    // stream. Their payload pipeline therefore needs no reset, avoiding a
    // wide, high-fanout reset tree across the complete 4x4 mesh.
    always @(posedge clk) begin
        r_stationary_pipe[0] <= {NUM_DPE{i_stationary}};
        r_benes_pipe[0] <= i_benes_bus;
        r_vn_pipe[0] <= i_vn_seperator;
        for (control_stage = 1; control_stage <= MAX_MESH_HOPS;
             control_stage = control_stage + 1) begin
            r_stationary_pipe[control_stage] <= r_stationary_pipe[control_stage-1];
            r_benes_pipe[control_stage] <= r_benes_pipe[control_stage-1];
            r_vn_pipe[control_stage] <= r_vn_pipe[control_stage-1];
        end
    end

    genvar mesh_row, mesh_col;
    generate
        for (mesh_row = 0; mesh_row < MESH_ROWS; mesh_row = mesh_row + 1) begin : g_mesh_row
            for (mesh_col = 0; mesh_col < MESH_COLS; mesh_col = mesh_col + 1) begin : g_mesh_col
                localparam integer DPE_ID = mesh_row * MESH_COLS + mesh_col;
                wire [MESH_HOP_WIDTH-1:0] requested_hops =
                    i_mesh_route_enable ?
                    i_mesh_hops[DPE_ID*MESH_HOP_WIDTH +: MESH_HOP_WIDTH] : 0;
                wire [MESH_HOP_WIDTH-1:0] control_hops =
                    (requested_hops <= MAX_MESH_HOPS) ? requested_hops : MAX_MESH_HOPS;
                flexdpe #(
                    .IN_DATA_TYPE(IN_DATA_TYPE),
                    .OUT_DATA_TYPE(OUT_DATA_TYPE),
                    .NUM_PES(NUM_PES),
                    .LOG2_PES(LOG2_PES),
                    .FAN_ADD_PIPE_STAGES(FAN_ADD_PIPE_STAGES)
                ) u_flexdpe (
                    .clk(clk),
                    .rst(rst),
                    .i_data_valid(mesh_valid[DPE_ID]),
                    .i_data_bus(mesh_data[DPE_ID*NUM_PES*IN_DATA_TYPE +: NUM_PES*IN_DATA_TYPE]),
                    .i_stationary(r_stationary_pipe[control_hops][DPE_ID]),
                    .i_dest_bus({NUM_PES*LOG2_PES{1'b0}}),
                    .i_benes_bus(r_benes_pipe[control_hops]
                        [DPE_ID*BENES_CTRL_WIDTH +: BENES_CTRL_WIDTH]),
                    .i_vn_seperator(r_vn_pipe[control_hops]
                        [DPE_ID*NUM_PES*LOG2_PES +: NUM_PES*LOG2_PES]),
                    .o_data_valid(o_data_valid[DPE_ID*NUM_PES +: NUM_PES]),
                    .o_data_bus(o_data_bus[DPE_ID*NUM_PES*OUT_DATA_TYPE +: NUM_PES*OUT_DATA_TYPE]),
                    .o_vn_bus(o_vn_bus[DPE_ID*NUM_PES*LOG2_PES +: NUM_PES*LOG2_PES])
                );
            end
        end
    endgenerate
endmodule
