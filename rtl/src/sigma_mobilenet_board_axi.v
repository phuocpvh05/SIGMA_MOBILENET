`timescale 1ns / 1ps
// AXI4-Lite/JTAG board shell for the autonomous MobileNet top.
// Address map (byte addresses):
//   0x0000 CONTROL  W bit0=start, bit1=clear done/error latch
//   0x0004 STATUS   R busy/done/error/ready, layer, prediction
//   0x0008 CYCLES   R accelerator cycles for the last/current inference
//   0x000C ID       R 0x5349474d ("SIGM")
//   0x0010 LOAD     R generic SIGMA tile-loader cycles
//   0x0014 CORE     R generic 4x4 SIGMA core cycles
//   0x0018 POST     R result/readback/residual/post-process cycles
//   0x001C DEPTHWISE R optimized four-spatial-lane depthwise cycles
//   0x0020 POINTWISE R optimized four-output M=1 pointwise cycles
//   0x0024 CLOCK_HZ R configured PL clock frequency (300,000,000)
//   0x0028 PROFILE  R implementation profile/version ("MOB6")
//   0x002C IMAGE_COUNT R accepted image words since CONTROL.clear
//   0x0030 IMAGE_HASH  R order-sensitive hash of accepted BF16 image words
//   0x0034 RUN_COUNT   R completed hardware inferences since reset
//   0x1000..0x1C3C IMAGE[0..783], BF16 in write-data[15:0]
module sigma_mobilenet_board_axi #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 16,
    parameter WEIGHT_INIT_FILE = "mobilenet_onchip_bf16_wide.mem"
) (
    input  wire s_axi_aclk,
    input  wire s_axi_aresetn,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire s_axi_awvalid,
    output wire s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire s_axi_wvalid,
    output wire s_axi_wready,
    output reg  [1:0] s_axi_bresp,
    output reg  s_axi_bvalid,
    input  wire s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire s_axi_arvalid,
    output wire s_axi_arready,
    output reg  [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output reg  [1:0] s_axi_rresp,
    output reg  s_axi_rvalid,
    input  wire s_axi_rready
);
    localparam IMAGE_BASE = 16'h1000;
    localparam IMAGE_LAST = IMAGE_BASE + 16'd3132;

    reg aw_pending, w_pending;
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_q;
    reg [C_S_AXI_DATA_WIDTH-1:0] wdata_q;
    reg [(C_S_AXI_DATA_WIDTH/8)-1:0] wstrb_q;
    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready = !w_pending && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;

    reg image_wr_en, start_pulse;
    reg [9:0] image_wr_addr;
    reg [15:0] image_wr_data;
    reg done_latched, error_latched;
    reg [10:0] image_write_count;
    reg [31:0] image_hash;
    reg [31:0] inference_count;
    wire accel_busy, accel_done, accel_error;
    wire [3:0] prediction;
    wire [31:0] cycles;
    wire [31:0] load_cycles, core_cycles, post_cycles;
    wire [31:0] depthwise_cycles, pointwise_cycles;
    wire [7:0] layer;

    sigma_mobilenet_onchip_top #(
        .WEIGHT_INIT_FILE(WEIGHT_INIT_FILE),
        .CORE_NUM_PES(4),
        .CORE_LOG2_PES(2)
    ) u_accelerator (
        .clk(s_axi_aclk),
        .rst(!s_axi_aresetn),
        .i_image_wr_en(image_wr_en),
        .i_image_wr_addr(image_wr_addr),
        .i_image_wr_data(image_wr_data),
        .i_start(start_pulse),
        .o_busy(accel_busy),
        .o_done(accel_done),
        .o_prediction(prediction),
        .o_error(accel_error),
        .o_cycles(cycles),
        .o_load_cycles(load_cycles),
        .o_core_cycles(core_cycles),
        .o_post_cycles(post_cycles),
        .o_depthwise_cycles(depthwise_cycles),
        .o_pointwise_cycles(pointwise_cycles),
        .o_layer(layer)
    );

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_pending <= 0;
            w_pending <= 0;
            s_axi_bvalid <= 0;
            s_axi_bresp <= 0;
            s_axi_rvalid <= 0;
            s_axi_rresp <= 0;
            s_axi_rdata <= 0;
            image_wr_en <= 0;
            start_pulse <= 0;
            done_latched <= 0;
            error_latched <= 0;
            image_write_count <= 0;
            image_hash <= 0;
            inference_count <= 0;
        end else begin
            image_wr_en <= 0;
            start_pulse <= 0;
            if (accel_done) begin
                done_latched <= 1;
                inference_count <= inference_count + 1'b1;
            end
            if (accel_error)
                error_latched <= 1;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_pending <= 1;
                awaddr_q <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_pending <= 1;
                wdata_q <= s_axi_wdata;
                wstrb_q <= s_axi_wstrb;
            end

            if (aw_pending && w_pending && !s_axi_bvalid) begin
                aw_pending <= 0;
                w_pending <= 0;
                s_axi_bvalid <= 1;
                s_axi_bresp <= 2'b00;
                if (awaddr_q == 16'h0000) begin
                    if (wstrb_q[0] && wdata_q[1]) begin
                        done_latched <= 0;
                        error_latched <= 0;
                        image_write_count <= 0;
                        image_hash <= 0;
                    end
                    if (wstrb_q[0] && wdata_q[0] && !accel_busy) begin
                        start_pulse <= 1;
                        done_latched <= 0;
                        error_latched <= 0;
                    end else if (wstrb_q[0] && wdata_q[0])
                        s_axi_bresp <= 2'b10;
                end else if ((awaddr_q >= IMAGE_BASE) &&
                             (awaddr_q <= IMAGE_LAST) &&
                             (awaddr_q[1:0] == 0)) begin
                    if (!accel_busy && wstrb_q[0] && wstrb_q[1]) begin
                        image_wr_en <= 1;
                        image_wr_addr <= (awaddr_q - IMAGE_BASE) >> 2;
                        image_wr_data <= wdata_q[15:0];
                        image_write_count <= image_write_count + 1'b1;
                        // Rotate then XOR.  This is deliberately cheap in RTL
                        // but catches missing, repeated, reordered and stale
                        // AXI image writes before an inference is launched.
                        image_hash <= {image_hash[30:0], image_hash[31]} ^
                                      {16'd0, wdata_q[15:0]};
                    end else
                        s_axi_bresp <= 2'b10;
                end else
                    s_axi_bresp <= 2'b10;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 0;

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                s_axi_rresp <= 2'b00;
                case (s_axi_araddr)
                    16'h0000: s_axi_rdata <= 0;
                    16'h0004: s_axi_rdata <= {
                        12'd0, prediction, layer, 4'd0, 1'b1,
                        error_latched, done_latched, accel_busy};
                    16'h0008: s_axi_rdata <= cycles;
                    16'h000c: s_axi_rdata <= 32'h5349474d;
                    16'h0010: s_axi_rdata <= load_cycles;
                    16'h0014: s_axi_rdata <= core_cycles;
                    16'h0018: s_axi_rdata <= post_cycles;
                    16'h001c: s_axi_rdata <= depthwise_cycles;
                    16'h0020: s_axi_rdata <= pointwise_cycles;
                    16'h0024: s_axi_rdata <= 32'd300000000;
                    // MOB6: pipelined model ROM and replicable per-bank
                    // address registers. Software rejects older marginal
                    // bitstream cannot be benchmarked accidentally.
                    16'h0028: s_axi_rdata <= 32'h4d4f4236;
                    16'h002c: s_axi_rdata <= {21'd0, image_write_count};
                    16'h0030: s_axi_rdata <= image_hash;
                    16'h0034: s_axi_rdata <= inference_count;
                    default: begin
                        s_axi_rdata <= 0;
                        s_axi_rresp <= 2'b10;
                    end
                endcase
            end
            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 0;
        end
    end
endmodule
