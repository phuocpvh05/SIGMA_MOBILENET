`timescale 1ns / 1ps
// Local model/activation store for MobileNetV2-0.25.  The 490,900-byte BF16
// model image is initialized into Block RAM with the bitstream.  Three small
// activation banks use UltraRAM: they are runtime-written (no initialization
// requirement), remain fully on-chip, and avoid both BRAM exhaustion and a
// very large distributed-RAM address fanout on the 5EV.
module sigma_mobilenet_onchip_store #(
    parameter WEIGHT_WORDS = 245450,
    parameter WEIGHT_WIDE_WORDS = 64632,
    parameter WEIGHT_AW = 18,
    parameter BANK_WORDS = 12288,
    parameter BANK_AW = 14,
    // WEIGHT_INIT_FILE is retained for source compatibility with the MOB2
    // wrappers.  MOB6 uses four independently initialized 64-bit BRAM banks.
    parameter WEIGHT_INIT_FILE = "mobilenet_onchip_bf16_wide.mem",
    parameter WEIGHT_BANK0_INIT_FILE = "mobilenet_onchip_bf16_bank0.mem",
    parameter WEIGHT_BANK1_INIT_FILE = "mobilenet_onchip_bf16_bank1.mem",
    parameter WEIGHT_BANK2_INIT_FILE = "mobilenet_onchip_bf16_bank2.mem",
    parameter WEIGHT_BANK3_INIT_FILE = "mobilenet_onchip_bf16_bank3.mem"
) (
    input  wire clk,
    input  wire i_weight_rd_en,
    input  wire [WEIGHT_AW-1:0] i_weight_rd_addr,
    output wire [15:0] o_weight_rd_data,
    input  wire i_weight_wide_rd_en,
    input  wire [WEIGHT_AW-1:0] i_weight_wide_rd_addr,
    output wire [63:0] o_weight_wide_rd_data,
    input  wire i_weight_super_rd_en,
    input  wire [WEIGHT_AW-1:0] i_weight_super_rd_addr,
    output wire [255:0] o_weight_super_rd_data,

    input  wire i_activation_wr_en,
    input  wire [1:0] i_activation_wr_bank,
    input  wire [BANK_AW-1:0] i_activation_wr_addr,
    input  wire [15:0] i_activation_wr_data,
    input  wire i_activation_wide_wr_en,
    input  wire [1:0] i_activation_wide_wr_bank,
    input  wire [BANK_AW-1:0] i_activation_wide_wr_addr,
    input  wire [63:0] i_activation_wide_wr_data,

    input  wire i_activation_rd_en,
    input  wire [1:0] i_activation_rd_bank,
    input  wire [BANK_AW-1:0] i_activation_rd_addr,
    output wire [15:0] o_activation_rd_data,
    input  wire i_activation_wide_rd_en,
    input  wire [1:0] i_activation_wide_rd_bank,
    input  wire [BANK_AW-1:0] i_activation_wide_rd_addr,
    output wire [63:0] o_activation_wide_rd_data,

    input  wire i_skip_rd_en,
    input  wire [1:0] i_skip_rd_bank,
    input  wire [BANK_AW-1:0] i_skip_rd_addr,
    output wire [15:0] o_skip_rd_data
);
    // Four logical 64-bit weight rows share one physical address.  Use XPM
    // SPROMs in synthesis so the initialized model is unambiguously mapped to
    // RAMB36E2 blocks.  A plain inferred array was mapped by Vivado 2025.1 to
    // >200k LUTs/61k MUXF7s despite ram_style="block".  The behavioral arrays
    // remain under !SYNTHESIS so RTL simulation stays vendor-independent.
    // Generic/depthwise reads select one 64-bit lane; an aligned M=1 request
    // consumes all four lanes (256 bits) in parallel.  Capacity, address
    // semantics are unchanged.  MOB6 gives the ROM three registered read
    // cycles.  The extra XPM pipeline register breaks the physical
    // RAMB36-to-consumer path that had only 19 ps of routed slack in MOB5.
    localparam WEIGHT_PHYSICAL_ROWS = (WEIGHT_WIDE_WORDS + 3) / 4;
    localparam WEIGHT_PHYSICAL_AW = $clog2(WEIGHT_PHYSICAL_ROWS);
    localparam WEIGHT_BANK_BITS = 64 * WEIGHT_PHYSICAL_ROWS;
    wire weight_any_rd_en = i_weight_super_rd_en ||
                            i_weight_wide_rd_en || i_weight_rd_en;
    wire [WEIGHT_PHYSICAL_AW-1:0] weight_phys_rd_addr =
        i_weight_super_rd_en ? (i_weight_super_rd_addr >> 2) :
        i_weight_wide_rd_en  ? (i_weight_wide_rd_addr  >> 2) :
                               (i_weight_rd_addr       >> 4);
    // Isolate each deep ROM cascade from the controller request mux with its
    // own physical register bank.  A single shared address register used by
    // MOB4 drove all 128 RAMB36 address pins.  That route consumed 96.7% of
    // the 300 MHz period and left only 25 ps post-route setup margin, which
    // was not stable on the physical board even though timing barely passed.
    //
    // KEEP is intentional: the four copies have identical D/CE logic, so
    // Vivado would otherwise merge them back into one high-fanout register.
    // DONT_TOUCH is deliberately not used in MOB6.  MAX_FANOUT=8 allows
    // synthesis/physical optimization to replicate each bank-local register
    // near the RAMB36 address pins instead of forcing one register to span all
    // 32 BRAMs in that bank.
    (* keep = "true", max_fanout = 8 *)
    reg weight_req_en_bank0_q;
    (* keep = "true", max_fanout = 8 *)
    reg weight_req_en_bank1_q;
    (* keep = "true", max_fanout = 8 *)
    reg weight_req_en_bank2_q;
    (* keep = "true", max_fanout = 8 *)
    reg weight_req_en_bank3_q;
    (* keep = "true", max_fanout = 8 *)
    reg [WEIGHT_PHYSICAL_AW-1:0] weight_phys_rd_addr_bank0_q;
    (* keep = "true", max_fanout = 8 *)
    reg [WEIGHT_PHYSICAL_AW-1:0] weight_phys_rd_addr_bank1_q;
    (* keep = "true", max_fanout = 8 *)
    reg [WEIGHT_PHYSICAL_AW-1:0] weight_phys_rd_addr_bank2_q;
    (* keep = "true", max_fanout = 8 *)
    reg [WEIGHT_PHYSICAL_AW-1:0] weight_phys_rd_addr_bank3_q;
    wire [63:0] weight_bank0_q, weight_bank1_q;
    wire [63:0] weight_bank2_q, weight_bank3_q;

`ifdef SYNTHESIS
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(WEIGHT_PHYSICAL_AW),
        .AUTO_SLEEP_TIME(0),
        // A four-block hard cascade reduces the external soft mux.  MOB6's
        // third read-latency stage registers the cascade output before it
        // reaches the depthwise/pointwise caches.
        .CASCADE_HEIGHT(4),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(WEIGHT_BANK0_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("false"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(WEIGHT_BANK_BITS),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(64),
        .READ_LATENCY_A(3),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_weight_bank0 (
        .clka(clk), .rsta(1'b0), .ena(weight_req_en_bank0_q),
        .regcea(1'b1), .addra(weight_phys_rd_addr_bank0_q),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .douta(weight_bank0_q), .sbiterra(), .dbiterra(), .sleep(1'b0)
    );

    xpm_memory_sprom #(
        .ADDR_WIDTH_A(WEIGHT_PHYSICAL_AW),
        .AUTO_SLEEP_TIME(0),
        .CASCADE_HEIGHT(4),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(WEIGHT_BANK1_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("false"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(WEIGHT_BANK_BITS),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(64),
        .READ_LATENCY_A(3),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_weight_bank1 (
        .clka(clk), .rsta(1'b0), .ena(weight_req_en_bank1_q),
        .regcea(1'b1), .addra(weight_phys_rd_addr_bank1_q),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .douta(weight_bank1_q), .sbiterra(), .dbiterra(), .sleep(1'b0)
    );

    xpm_memory_sprom #(
        .ADDR_WIDTH_A(WEIGHT_PHYSICAL_AW),
        .AUTO_SLEEP_TIME(0),
        .CASCADE_HEIGHT(4),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(WEIGHT_BANK2_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("false"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(WEIGHT_BANK_BITS),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(64),
        .READ_LATENCY_A(3),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_weight_bank2 (
        .clka(clk), .rsta(1'b0), .ena(weight_req_en_bank2_q),
        .regcea(1'b1), .addra(weight_phys_rd_addr_bank2_q),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .douta(weight_bank2_q), .sbiterra(), .dbiterra(), .sleep(1'b0)
    );

    xpm_memory_sprom #(
        .ADDR_WIDTH_A(WEIGHT_PHYSICAL_AW),
        .AUTO_SLEEP_TIME(0),
        .CASCADE_HEIGHT(4),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(WEIGHT_BANK3_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("false"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(WEIGHT_BANK_BITS),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(64),
        .READ_LATENCY_A(3),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_weight_bank3 (
        .clka(clk), .rsta(1'b0), .ena(weight_req_en_bank3_q),
        .regcea(1'b1), .addra(weight_phys_rd_addr_bank3_q),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .douta(weight_bank3_q), .sbiterra(), .dbiterra(), .sleep(1'b0)
    );
`else
    reg [63:0] weight_bank0 [0:WEIGHT_PHYSICAL_ROWS-1];
    reg [63:0] weight_bank1 [0:WEIGHT_PHYSICAL_ROWS-1];
    reg [63:0] weight_bank2 [0:WEIGHT_PHYSICAL_ROWS-1];
    reg [63:0] weight_bank3 [0:WEIGHT_PHYSICAL_ROWS-1];
    reg [63:0] weight_bank0_sim_raw_q, weight_bank1_sim_raw_q;
    reg [63:0] weight_bank2_sim_raw_q, weight_bank3_sim_raw_q;
    reg [63:0] weight_bank0_sim_mid_q, weight_bank1_sim_mid_q;
    reg [63:0] weight_bank2_sim_mid_q, weight_bank3_sim_mid_q;
    reg [63:0] weight_bank0_sim_out_q, weight_bank1_sim_out_q;
    reg [63:0] weight_bank2_sim_out_q, weight_bank3_sim_out_q;
    reg weight_sim_valid_d1, weight_sim_valid_d2;
    assign weight_bank0_q = weight_bank0_sim_out_q;
    assign weight_bank1_q = weight_bank1_sim_out_q;
    assign weight_bank2_q = weight_bank2_sim_out_q;
    assign weight_bank3_q = weight_bank3_sim_out_q;

    initial begin
        if (WEIGHT_BANK0_INIT_FILE != "")
            $readmemh(WEIGHT_BANK0_INIT_FILE, weight_bank0);
        if (WEIGHT_BANK1_INIT_FILE != "")
            $readmemh(WEIGHT_BANK1_INIT_FILE, weight_bank1);
        if (WEIGHT_BANK2_INIT_FILE != "")
            $readmemh(WEIGHT_BANK2_INIT_FILE, weight_bank2);
        if (WEIGHT_BANK3_INIT_FILE != "")
            $readmemh(WEIGHT_BANK3_INIT_FILE, weight_bank3);
    end

    always @(posedge clk) begin
        weight_sim_valid_d1 <= weight_req_en_bank0_q;
        weight_sim_valid_d2 <= weight_sim_valid_d1;
        if (weight_req_en_bank0_q)
            weight_bank0_sim_raw_q <=
                weight_bank0[weight_phys_rd_addr_bank0_q];
        if (weight_req_en_bank1_q)
            weight_bank1_sim_raw_q <=
                weight_bank1[weight_phys_rd_addr_bank1_q];
        if (weight_req_en_bank2_q)
            weight_bank2_sim_raw_q <=
                weight_bank2[weight_phys_rd_addr_bank2_q];
        if (weight_req_en_bank3_q)
            weight_bank3_sim_raw_q <=
                weight_bank3[weight_phys_rd_addr_bank3_q];
        if (weight_sim_valid_d1) begin
            weight_bank0_sim_mid_q <= weight_bank0_sim_raw_q;
            weight_bank1_sim_mid_q <= weight_bank1_sim_raw_q;
            weight_bank2_sim_mid_q <= weight_bank2_sim_raw_q;
            weight_bank3_sim_mid_q <= weight_bank3_sim_raw_q;
        end
        if (weight_sim_valid_d2) begin
            weight_bank0_sim_out_q <= weight_bank0_sim_mid_q;
            weight_bank1_sim_out_q <= weight_bank1_sim_mid_q;
            weight_bank2_sim_out_q <= weight_bank2_sim_mid_q;
            weight_bank3_sim_out_q <= weight_bank3_sim_mid_q;
        end
    end
`endif
    // Interleave every logical activation bank over four physical URAM
    // lanes.  Scalar accesses retain their original address semantics, while
    // depthwise layers can read/write four adjacent HWC channels in one
    // clock.  Four 3072-word lanes carry exactly the same data as the former
    // single 12288-word array; the banking changes bandwidth, not capacity.
    localparam BANK_LANE_WORDS = (BANK_WORDS + 3) / 4;
    (* ram_style = "ultra" *) reg [15:0] activation0_l0 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation0_l1 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation0_l2 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation0_l3 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation1_l0 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation1_l1 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation1_l2 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation1_l3 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation2_l0 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation2_l1 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation2_l2 [0:BANK_LANE_WORDS-1];
    (* ram_style = "ultra" *) reg [15:0] activation2_l3 [0:BANK_LANE_WORDS-1];

    reg [63:0] bank0_q, bank1_q, bank2_q;
    reg [1:0] bank0_lane_q, bank1_lane_q, bank2_lane_q;
    reg [1:0] activation_wide_bank_q;
    reg [1:0] weight_wide_lane_d1, weight_wide_lane_d2,
              weight_wide_lane_d3, weight_wide_lane_q;
    reg [1:0] weight_lane_d1, weight_lane_d2, weight_lane_d3,
              weight_lane_q;
    wire [63:0] weight_wide_q = (weight_wide_lane_q == 0) ? weight_bank0_q :
                                  (weight_wide_lane_q == 1) ? weight_bank1_q :
                                  (weight_wide_lane_q == 2) ? weight_bank2_q :
                                                                  weight_bank3_q;
    assign o_weight_super_rd_data = {weight_bank3_q, weight_bank2_q,
                                      weight_bank1_q, weight_bank0_q};
    assign o_weight_wide_rd_data = weight_wide_q;
    assign o_weight_rd_data =
        weight_wide_q[weight_lane_q*16 +: 16];
    wire primary_rd_en = i_activation_wide_rd_en || i_activation_rd_en;
    wire [1:0] primary_rd_bank = i_activation_wide_rd_en ?
        i_activation_wide_rd_bank : i_activation_rd_bank;
    wire [BANK_AW-1:0] primary_rd_addr = i_activation_wide_rd_en ?
        i_activation_wide_rd_addr : i_activation_rd_addr;
    wire bank0_primary = primary_rd_en && (primary_rd_bank == 0);
    wire bank1_primary = primary_rd_en && (primary_rd_bank == 1);
    wire bank2_primary = primary_rd_en && (primary_rd_bank == 2);
    wire bank0_skip = i_skip_rd_en && (i_skip_rd_bank == 0);
    wire bank1_skip = i_skip_rd_en && (i_skip_rd_bank == 1);
    wire bank2_skip = i_skip_rd_en && (i_skip_rd_bank == 2);
    wire [BANK_AW-1:0] bank0_rd_addr = bank0_primary ? primary_rd_addr : i_skip_rd_addr;
    wire [BANK_AW-1:0] bank1_rd_addr = bank1_primary ? primary_rd_addr : i_skip_rd_addr;
    wire [BANK_AW-1:0] bank2_rd_addr = bank2_primary ? primary_rd_addr : i_skip_rd_addr;
    assign o_activation_wide_rd_data = (activation_wide_bank_q == 0) ? bank0_q :
                                           (activation_wide_bank_q == 1) ? bank1_q : bank2_q;
    assign o_activation_rd_data = (i_activation_rd_bank == 0) ?
        bank0_q[bank0_lane_q*16 +: 16] :
        (i_activation_rd_bank == 1) ? bank1_q[bank1_lane_q*16 +: 16] :
                                     bank2_q[bank2_lane_q*16 +: 16];
    assign o_skip_rd_data = (i_skip_rd_bank == 0) ?
        bank0_q[bank0_lane_q*16 +: 16] :
        (i_skip_rd_bank == 1) ? bank1_q[bank1_lane_q*16 +: 16] :
                               bank2_q[bank2_lane_q*16 +: 16];

    always @(posedge clk) begin
        weight_req_en_bank0_q <= weight_any_rd_en;
        weight_req_en_bank1_q <= weight_any_rd_en;
        weight_req_en_bank2_q <= weight_any_rd_en;
        weight_req_en_bank3_q <= weight_any_rd_en;
        if (weight_any_rd_en) begin
            weight_phys_rd_addr_bank0_q <= weight_phys_rd_addr;
            weight_phys_rd_addr_bank1_q <= weight_phys_rd_addr;
            weight_phys_rd_addr_bank2_q <= weight_phys_rd_addr;
            weight_phys_rd_addr_bank3_q <= weight_phys_rd_addr;
        end

        // Match the selector to the input address stage plus the three-cycle
        // MOB6 XPM read pipeline.
        weight_wide_lane_q <= weight_wide_lane_d3;
        weight_wide_lane_d3 <= weight_wide_lane_d2;
        weight_wide_lane_d2 <= weight_wide_lane_d1;
        weight_lane_q <= weight_lane_d3;
        weight_lane_d3 <= weight_lane_d2;
        weight_lane_d2 <= weight_lane_d1;
        if (i_weight_super_rd_en) begin
            weight_wide_lane_d1 <= 0;
            weight_lane_d1 <= 0;
        end else if (i_weight_wide_rd_en) begin
            weight_wide_lane_d1 <= i_weight_wide_rd_addr[1:0];
            weight_lane_d1 <= 0;
        end else if (i_weight_rd_en) begin
            weight_wide_lane_d1 <= i_weight_rd_addr[3:2];
            weight_lane_d1 <= i_weight_rd_addr[1:0];
        end

        if (i_activation_wide_rd_en)
            activation_wide_bank_q <= i_activation_wide_rd_bank;

        if (i_activation_wide_wr_en) begin
            case (i_activation_wide_wr_bank)
                0: begin
                    activation0_l0[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[ 0 +: 16];
                    activation0_l1[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[16 +: 16];
                    activation0_l2[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[32 +: 16];
                    activation0_l3[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[48 +: 16];
                end
                1: begin
                    activation1_l0[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[ 0 +: 16];
                    activation1_l1[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[16 +: 16];
                    activation1_l2[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[32 +: 16];
                    activation1_l3[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[48 +: 16];
                end
                2: begin
                    activation2_l0[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[ 0 +: 16];
                    activation2_l1[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[16 +: 16];
                    activation2_l2[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[32 +: 16];
                    activation2_l3[i_activation_wide_wr_addr >> 2] <= i_activation_wide_wr_data[48 +: 16];
                end
                default: ;
            endcase
        end else if (i_activation_wr_en) begin
            case (i_activation_wr_bank)
                0: case (i_activation_wr_addr[1:0])
                    0: activation0_l0[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    1: activation0_l1[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    2: activation0_l2[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    default: activation0_l3[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                endcase
                1: case (i_activation_wr_addr[1:0])
                    0: activation1_l0[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    1: activation1_l1[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    2: activation1_l2[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    default: activation1_l3[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                endcase
                2: case (i_activation_wr_addr[1:0])
                    0: activation2_l0[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    1: activation2_l1[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    2: activation2_l2[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                    default: activation2_l3[i_activation_wr_addr >> 2] <= i_activation_wr_data;
                endcase
                default: ;
            endcase
        end

        if (bank0_primary || bank0_skip) begin
            bank0_q[ 0 +: 16] <= activation0_l0[bank0_rd_addr >> 2];
            bank0_q[16 +: 16] <= activation0_l1[bank0_rd_addr >> 2];
            bank0_q[32 +: 16] <= activation0_l2[bank0_rd_addr >> 2];
            bank0_q[48 +: 16] <= activation0_l3[bank0_rd_addr >> 2];
            bank0_lane_q <= bank0_rd_addr[1:0];
        end
        if (bank1_primary || bank1_skip) begin
            bank1_q[ 0 +: 16] <= activation1_l0[bank1_rd_addr >> 2];
            bank1_q[16 +: 16] <= activation1_l1[bank1_rd_addr >> 2];
            bank1_q[32 +: 16] <= activation1_l2[bank1_rd_addr >> 2];
            bank1_q[48 +: 16] <= activation1_l3[bank1_rd_addr >> 2];
            bank1_lane_q <= bank1_rd_addr[1:0];
        end
        if (bank2_primary || bank2_skip) begin
            bank2_q[ 0 +: 16] <= activation2_l0[bank2_rd_addr >> 2];
            bank2_q[16 +: 16] <= activation2_l1[bank2_rd_addr >> 2];
            bank2_q[32 +: 16] <= activation2_l2[bank2_rd_addr >> 2];
            bank2_q[48 +: 16] <= activation2_l3[bank2_rd_addr >> 2];
            bank2_lane_q <= bank2_rd_addr[1:0];
        end

    end
endmodule
