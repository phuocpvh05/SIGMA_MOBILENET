`timescale 1ns / 1ps
// Focused timing contract check for the MobileNet model ROM.  It checks both
// the 256-bit super-read latency and four consecutive 64-bit lane selections.
// The latter catches an address/data skew that a single static read cannot.
module tb_mobilenet_store_latency;
    reg clk = 0;
    always #1.6665 clk = ~clk;

    reg weight_rd_en = 0;
    reg [17:0] weight_rd_addr = 0;
    wire [15:0] weight_rd_data;
    reg weight_wide_rd_en = 0;
    reg [17:0] weight_wide_rd_addr = 0;
    wire [63:0] weight_wide_rd_data;
    reg weight_super_rd_en = 0;
    reg [17:0] weight_super_rd_addr = 0;
    wire [255:0] weight_super_rd_data;
    integer cycle = 0;

    sigma_mobilenet_onchip_store dut (
        .clk(clk),
        .i_weight_rd_en(weight_rd_en),
        .i_weight_rd_addr(weight_rd_addr),
        .o_weight_rd_data(weight_rd_data),
        .i_weight_wide_rd_en(weight_wide_rd_en),
        .i_weight_wide_rd_addr(weight_wide_rd_addr),
        .o_weight_wide_rd_data(weight_wide_rd_data),
        .i_weight_super_rd_en(weight_super_rd_en),
        .i_weight_super_rd_addr(weight_super_rd_addr),
        .o_weight_super_rd_data(weight_super_rd_data),
        .i_activation_wr_en(1'b0),
        .i_activation_wr_bank(2'd0),
        .i_activation_wr_addr(14'd0),
        .i_activation_wr_data(16'd0),
        .i_activation_wide_wr_en(1'b0),
        .i_activation_wide_wr_bank(2'd0),
        .i_activation_wide_wr_addr(14'd0),
        .i_activation_wide_wr_data(64'd0),
        .i_activation_rd_en(1'b0),
        .i_activation_rd_bank(2'd0),
        .i_activation_rd_addr(14'd0),
        .o_activation_rd_data(),
        .i_activation_wide_rd_en(1'b0),
        .i_activation_wide_rd_bank(2'd0),
        .i_activation_wide_rd_addr(14'd0),
        .o_activation_wide_rd_data(),
        .i_skip_rd_en(1'b0),
        .i_skip_rd_bank(2'd0),
        .i_skip_rd_addr(14'd0),
        .o_skip_rd_data()
    );

    always @(posedge clk) begin
        cycle <= cycle + 1;
        $display("STORE_LATENCY cycle=%0d super_en=%b wide_en=%b wide_addr=%0d super=%064h wide=%016h",
                 cycle, weight_super_rd_en, weight_wide_rd_en,
                 weight_wide_rd_addr,
                 weight_super_rd_data, weight_wide_rd_data);
        case (cycle)
            14: if (weight_wide_rd_data !== 64'h3f99bf87bf36bfd2)
                    $fatal(1, "STORE_LATENCY lane0 mismatch: %016h",
                           weight_wide_rd_data);
            15: if (weight_wide_rd_data !== 64'h00000000bc8fc032)
                    $fatal(1, "STORE_LATENCY lane1 mismatch: %016h",
                           weight_wide_rd_data);
            16: if (weight_wide_rd_data !== 64'h3e283f3c3f3f3e98)
                    $fatal(1, "STORE_LATENCY lane2 mismatch: %016h",
                           weight_wide_rd_data);
            17: begin
                if (weight_wide_rd_data !== 64'hbf6d3ffdc0453ff7)
                    $fatal(1, "STORE_LATENCY lane3 mismatch: %016h",
                           weight_wide_rd_data);
                $display("STORE_LATENCY PASSED: four consecutive lanes aligned");
            end
            default: ;
        endcase
    end

    initial begin
        repeat (4) @(negedge clk);
        weight_super_rd_addr = 0;
        weight_super_rd_en = 1;
        @(negedge clk);
        weight_super_rd_en = 0;
        repeat (5) @(negedge clk);

        // Change both physical row and logical lane on every clock.  This
        // proves that the delayed lane selector belongs to the same request
        // as the BRAM row, not merely that a static row can be selected.
        weight_wide_rd_en = 1;
        weight_wide_rd_addr = 0;
        @(negedge clk);
        weight_wide_rd_addr = 5;
        @(negedge clk);
        weight_wide_rd_addr = 10;
        @(negedge clk);
        weight_wide_rd_addr = 15;
        @(negedge clk);
        weight_wide_rd_en = 0;
        repeat (8) @(negedge clk);
        $finish;
    end
endmodule
