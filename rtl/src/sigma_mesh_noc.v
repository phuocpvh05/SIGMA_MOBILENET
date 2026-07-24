`timescale 1ns / 1ps
//==========================================================================
// Parameterized bidirectional 2D static mesh NoC for composing Flex-DPEs.
// Local payloads may be injected/ejected at every DPE node.  Routing is fixed
// for a mapped GEMM, so the fabric needs no packet header or dynamic arbiter.
//==========================================================================

module sigma_mesh_noc #(
    parameter MESH_ROWS = 4,
    parameter MESH_COLS = 4,
    parameter DATA_WIDTH = 32,
    parameter NUM_PORTS = 5,
    parameter SEL_WIDTH = 3,
    parameter NUM_NODES = MESH_ROWS * MESH_COLS,
    parameter ROUTE_WIDTH = NUM_PORTS * SEL_WIDTH
) (
    input  wire clk,
    input  wire rst,
    input  wire [NUM_NODES*DATA_WIDTH-1:0] i_local_data,
    input  wire [NUM_NODES-1:0] i_local_valid,
    input  wire [NUM_NODES*ROUTE_WIDTH-1:0] i_route_config,
    output wire [NUM_NODES*DATA_WIDTH-1:0] o_local_data,
    output wire [NUM_NODES-1:0] o_local_valid
);
    localparam PORT_LOCAL = 0;
    localparam PORT_NORTH = 1;
    localparam PORT_EAST  = 2;
    localparam PORT_SOUTH = 3;
    localparam PORT_WEST  = 4;

    wire [NUM_NODES*NUM_PORTS*DATA_WIDTH-1:0] node_in_data;
    wire [NUM_NODES*NUM_PORTS-1:0] node_in_valid;
    wire [NUM_NODES*NUM_PORTS*DATA_WIDTH-1:0] node_out_data;
    wire [NUM_NODES*NUM_PORTS-1:0] node_out_valid;

    genvar row, col;
    generate
        for (row = 0; row < MESH_ROWS; row = row + 1) begin : g_row
            for (col = 0; col < MESH_COLS; col = col + 1) begin : g_col
                localparam integer NODE = row * MESH_COLS + col;
                localparam integer NORTH_NODE = (row-1) * MESH_COLS + col;
                localparam integer EAST_NODE  = row * MESH_COLS + col + 1;
                localparam integer SOUTH_NODE = (row+1) * MESH_COLS + col;
                localparam integer WEST_NODE  = row * MESH_COLS + col - 1;

                assign node_in_data[(NODE*NUM_PORTS+PORT_LOCAL)*DATA_WIDTH +: DATA_WIDTH] =
                    i_local_data[NODE*DATA_WIDTH +: DATA_WIDTH];
                assign node_in_valid[NODE*NUM_PORTS+PORT_LOCAL] = i_local_valid[NODE];

                if (row > 0) begin : g_north_link
                    assign node_in_data[(NODE*NUM_PORTS+PORT_NORTH)*DATA_WIDTH +: DATA_WIDTH] =
                        node_out_data[(NORTH_NODE*NUM_PORTS+PORT_SOUTH)*DATA_WIDTH +: DATA_WIDTH];
                    assign node_in_valid[NODE*NUM_PORTS+PORT_NORTH] =
                        node_out_valid[NORTH_NODE*NUM_PORTS+PORT_SOUTH];
                end else begin : g_no_north
                    assign node_in_data[(NODE*NUM_PORTS+PORT_NORTH)*DATA_WIDTH +: DATA_WIDTH] = 0;
                    assign node_in_valid[NODE*NUM_PORTS+PORT_NORTH] = 1'b0;
                end

                if (col + 1 < MESH_COLS) begin : g_east_link
                    assign node_in_data[(NODE*NUM_PORTS+PORT_EAST)*DATA_WIDTH +: DATA_WIDTH] =
                        node_out_data[(EAST_NODE*NUM_PORTS+PORT_WEST)*DATA_WIDTH +: DATA_WIDTH];
                    assign node_in_valid[NODE*NUM_PORTS+PORT_EAST] =
                        node_out_valid[EAST_NODE*NUM_PORTS+PORT_WEST];
                end else begin : g_no_east
                    assign node_in_data[(NODE*NUM_PORTS+PORT_EAST)*DATA_WIDTH +: DATA_WIDTH] = 0;
                    assign node_in_valid[NODE*NUM_PORTS+PORT_EAST] = 1'b0;
                end

                if (row + 1 < MESH_ROWS) begin : g_south_link
                    assign node_in_data[(NODE*NUM_PORTS+PORT_SOUTH)*DATA_WIDTH +: DATA_WIDTH] =
                        node_out_data[(SOUTH_NODE*NUM_PORTS+PORT_NORTH)*DATA_WIDTH +: DATA_WIDTH];
                    assign node_in_valid[NODE*NUM_PORTS+PORT_SOUTH] =
                        node_out_valid[SOUTH_NODE*NUM_PORTS+PORT_NORTH];
                end else begin : g_no_south
                    assign node_in_data[(NODE*NUM_PORTS+PORT_SOUTH)*DATA_WIDTH +: DATA_WIDTH] = 0;
                    assign node_in_valid[NODE*NUM_PORTS+PORT_SOUTH] = 1'b0;
                end

                if (col > 0) begin : g_west_link
                    assign node_in_data[(NODE*NUM_PORTS+PORT_WEST)*DATA_WIDTH +: DATA_WIDTH] =
                        node_out_data[(WEST_NODE*NUM_PORTS+PORT_EAST)*DATA_WIDTH +: DATA_WIDTH];
                    assign node_in_valid[NODE*NUM_PORTS+PORT_WEST] =
                        node_out_valid[WEST_NODE*NUM_PORTS+PORT_EAST];
                end else begin : g_no_west
                    assign node_in_data[(NODE*NUM_PORTS+PORT_WEST)*DATA_WIDTH +: DATA_WIDTH] = 0;
                    assign node_in_valid[NODE*NUM_PORTS+PORT_WEST] = 1'b0;
                end

                sigma_mesh_switch #(.DATA_WIDTH(DATA_WIDTH)) u_switch (
                    .clk(clk),
                    .rst(rst),
                    .i_data(node_in_data[NODE*NUM_PORTS*DATA_WIDTH +: NUM_PORTS*DATA_WIDTH]),
                    .i_valid(node_in_valid[NODE*NUM_PORTS +: NUM_PORTS]),
                    .i_route(i_route_config[NODE*ROUTE_WIDTH +: ROUTE_WIDTH]),
                    .o_data(node_out_data[NODE*NUM_PORTS*DATA_WIDTH +: NUM_PORTS*DATA_WIDTH]),
                    .o_valid(node_out_valid[NODE*NUM_PORTS +: NUM_PORTS])
                );

                assign o_local_data[NODE*DATA_WIDTH +: DATA_WIDTH] =
                    node_out_data[(NODE*NUM_PORTS+PORT_LOCAL)*DATA_WIDTH +: DATA_WIDTH];
                assign o_local_valid[NODE] = node_out_valid[NODE*NUM_PORTS+PORT_LOCAL];
            end
        end
    endgenerate
endmodule

