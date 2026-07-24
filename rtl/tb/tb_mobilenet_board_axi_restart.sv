`timescale 1ns / 1ps
// Board-path regression: exercise the real AXI4-Lite shell, transfer
// diagnostics and two back-to-back inferences with different predictions.
module tb_mobilenet_board_axi_restart;
    localparam IMAGE_WORDS = 784;
    reg clk = 0;
    reg resetn = 0;
    always #1.6665 clk = ~clk;

    reg [15:0] awaddr = 0;
    reg awvalid = 0;
    wire awready;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 0;
    reg wvalid = 0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 1;
    reg [15:0] araddr = 0;
    reg arvalid = 0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 1;

    reg [15:0] image0 [0:IMAGE_WORDS-1];
    reg [15:0] image1 [0:IMAGE_WORDS-1];
    string image0_file, image1_file;
    integer index;
    reg [31:0] reference_cycles = 0;

`ifdef POST_SYNTH_NETLIST
    // Functional netlist emitted by the MobileNet OOC synthesis run.  This
    // path checks the exact optimized logic and embedded BRAM INIT contents
    // that are later stitched into the PS block design.
    sigma_mobilenet_ps_bd_sigma_mobilenet_0 dut (
        .s_axi_awprot(3'b000), .s_axi_arprot(3'b000),
`else
    sigma_mobilenet_board_axi dut (
`endif
        .s_axi_aclk(clk), .s_axi_aresetn(resetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid),
        .s_axi_awready(awready), .s_axi_wdata(wdata),
        .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid),
        .s_axi_wready(wready), .s_axi_bresp(bresp),
        .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata),
        .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready)
    );

    task automatic axi_write(input [15:0] address, input [31:0] data);
        integer aw_done, w_done;
        begin
            aw_done = 0;
            w_done = 0;
            @(negedge clk);
            awaddr = address;
            awvalid = 1;
            wdata = data;
            wstrb = 4'hf;
            wvalid = 1;
            while (!aw_done || !w_done) begin
                @(posedge clk);
                if (!aw_done && awready) begin
                    awvalid <= 0;
                    aw_done = 1;
                end
                if (!w_done && wready) begin
                    wvalid <= 0;
                    w_done = 1;
                end
            end
            while (!bvalid) @(posedge clk);
            if (bresp != 0) begin
                $display("MOBILENET_AXI_RESTART FAILED write addr=%h bresp=%b", address, bresp);
                $finish;
            end
            @(posedge clk);
        end
    endtask

    task automatic axi_read(input [15:0] address, output [31:0] data);
        begin
            @(negedge clk);
            araddr = address;
            arvalid = 1;
            while (!arready) @(posedge clk);
            @(posedge clk);
            arvalid <= 0;
            while (!rvalid) @(posedge clk);
            data = rdata;
            if (rresp != 0) begin
                $display("MOBILENET_AXI_RESTART FAILED read addr=%h rresp=%b", address, rresp);
                $finish;
            end
            @(posedge clk);
        end
    endtask

    task automatic run_image(input integer which, input [3:0] expected);
        reg [15:0] word_value;
        reg [31:0] expected_hash;
        reg [31:0] value;
        reg [31:0] run_before;
        integer polls;
        begin
            axi_read(16'h0034, run_before);
            axi_write(16'h0000, 32'd2);
            expected_hash = 0;
            for (index = 0; index < IMAGE_WORDS; index = index + 1) begin
                word_value = which ? image1[index] : image0[index];
                expected_hash = {expected_hash[30:0], expected_hash[31]} ^
                                {16'd0, word_value};
                axi_write(16'h1000 + index*4, {16'd0, word_value});
            end
            axi_read(16'h002c, value);
            if (value != IMAGE_WORDS) begin
                $display("MOBILENET_AXI_RESTART FAILED image_count=%0d", value);
                $finish;
            end
            axi_read(16'h0030, value);
            if (value != expected_hash) begin
                $display("MOBILENET_AXI_RESTART FAILED image_hash=%h expected=%h", value, expected_hash);
                $finish;
            end
            axi_write(16'h0000, 32'd1);
            value = 0;
            polls = 0;
            while (!value[1] && polls < 30000) begin
                repeat (64) @(posedge clk);
                axi_read(16'h0004, value);
                polls = polls + 1;
            end
            if (!value[1] || value[2] || value[19:16] != expected) begin
                $display("MOBILENET_AXI_RESTART FAILED status=%h expected=%0d", value, expected);
                $finish;
            end
            axi_read(16'h0008, value);
            if ((value < 1170000) || (value > 1200000)) begin
                $display("MOBILENET_AXI_RESTART FAILED cycles out of range=%0d", value);
                $finish;
            end
            if (reference_cycles == 0)
                reference_cycles = value;
            else if (value != reference_cycles) begin
                $display("MOBILENET_AXI_RESTART FAILED nondeterministic cycles=%0d expected=%0d",
                         value, reference_cycles);
                $finish;
            end
            axi_read(16'h0034, value);
            if (value != run_before + 1) begin
                $display("MOBILENET_AXI_RESTART FAILED run_count=%0d expected=%0d", value, run_before + 1);
                $finish;
            end
            $display("MOBILENET_AXI_RESTART run=%0d prediction=%0d cycles=%0d hash=%h PASS",
                     value, expected, reference_cycles, expected_hash);
        end
    endtask

    initial begin
        if (!$value$plusargs("IMAGE0=%s", image0_file) ||
            !$value$plusargs("IMAGE1=%s", image1_file)) begin
            $display("MOBILENET_AXI_RESTART FAILED IMAGE0 and IMAGE1 are required");
            $finish;
        end
        $readmemh(image0_file, image0);
        $readmemh(image1_file, image1);
        repeat (20) @(posedge clk);
        resetn <= 1;
        repeat (10) @(posedge clk);
        axi_read(16'h0028, wdata);
        if (wdata != 32'h4d4f4237) begin
            $display("MOBILENET_AXI_RESTART FAILED profile=%h", wdata);
            $finish;
        end
        run_image(0, 7);
        run_image(1, 2);
        $display("MOBILENET_AXI_RESTART PASSED two different back-to-back inferences");
        $finish;
    end
endmodule
