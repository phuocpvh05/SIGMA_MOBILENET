`timescale 1ns / 1ps
//==============================================================================
// BF16 multiplier
//
// Format: sign[15], exponent[14:7], fraction[6:0].  Results are rounded to
// nearest, ties to even.  The registered wrapper keeps the original module
// interface used by the Flex-DPE.
//==============================================================================

module multiplier(
    input  wire        clk,
    input  wire [15:0] A,
    input  wire [15:0] B,
    output reg  [15:0] O
);
    wire [15:0] product;
    bf16_mult_pipeline u_multiplier_pipe (
        .clk(clk), .a(A), .b(B), .out(product)
    );
    always @(*) O = product;
endmodule


// Eight-stage BF16 multiplier used by each physical PE. Only the unsigned 8x8
// mantissa product is mapped to DSP48; exponent, normalization and rounding
// stay in nearby LUTs instead of being chained through many DSP ALUs.
module bf16_mult_pipeline(
    input  wire        clk,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] out
);
    function signed [5:0] highest_set_bit16_pipe;
        input [15:0] value;
        begin
            if      (value[15]) highest_set_bit16_pipe = 15;
            else if (value[14]) highest_set_bit16_pipe = 14;
            else if (value[13]) highest_set_bit16_pipe = 13;
            else if (value[12]) highest_set_bit16_pipe = 12;
            else if (value[11]) highest_set_bit16_pipe = 11;
            else if (value[10]) highest_set_bit16_pipe = 10;
            else if (value[9])  highest_set_bit16_pipe = 9;
            else if (value[8])  highest_set_bit16_pipe = 8;
            else if (value[7])  highest_set_bit16_pipe = 7;
            else if (value[6])  highest_set_bit16_pipe = 6;
            else if (value[5])  highest_set_bit16_pipe = 5;
            else if (value[4])  highest_set_bit16_pipe = 4;
            else if (value[3])  highest_set_bit16_pipe = 3;
            else if (value[2])  highest_set_bit16_pipe = 2;
            else if (value[1])  highest_set_bit16_pipe = 1;
            else if (value[0])  highest_set_bit16_pipe = 0;
            else                highest_set_bit16_pipe = -1;
        end
    endfunction

    wire [7:0] a_exp_eff = (a[14:7] == 0) ? 8'd1 : a[14:7];
    wire [7:0] b_exp_eff = (b[14:7] == 0) ? 8'd1 : b[14:7];
    wire [7:0] a_mant = (a[14:7] == 0) ? {1'b0, a[6:0]} : {1'b1, a[6:0]};
    wire [7:0] b_mant = (b[14:7] == 0) ? {1'b0, b[6:0]} : {1'b1, b[6:0]};
    wire a_nan = (a[14:7] == 8'hff) && (a[6:0] != 0);
    wire b_nan = (b[14:7] == 8'hff) && (b[6:0] != 0);
    wire a_inf = (a[14:7] == 8'hff) && (a[6:0] == 0);
    wire b_inf = (b[14:7] == 8'hff) && (b[6:0] == 0);
    wire a_zero = (a[14:0] == 0);
    wire b_zero = (b[14:0] == 0);

    // Stage 0: decode and register the DSP operands.  At 400 MHz the old
    // Benes-register -> BF16 decode -> DSP multiply path was the replicated
    // post-route critical path.  Keeping this boundary explicit lets Vivado
    // absorb the mantissa registers into the DSP A/B input registers while
    // preserving one accepted operand pair per clock.
    reg s0_special;
    reg [15:0] s0_special_out;
    reg s0_sign;
    reg [7:0] s0_a_exp_eff;
    reg [7:0] s0_b_exp_eff;
    reg [7:0] s0_a_mant;
    reg [7:0] s0_b_mant;
    always @(posedge clk) begin
        s0_special <= 1'b0;
        s0_special_out <= 0;
        if (a_nan || b_nan || ((a_inf && b_zero) || (b_inf && a_zero))) begin
            s0_special <= 1'b1;
            s0_special_out <= 16'h7fc0;
        end else if (a_inf || b_inf) begin
            s0_special <= 1'b1;
            s0_special_out <= {a[15] ^ b[15], 8'hff, 7'h00};
        end else if (a_zero || b_zero) begin
            s0_special <= 1'b1;
            s0_special_out <= {a[15] ^ b[15], 15'h0000};
        end
        s0_sign <= a[15] ^ b[15];
        s0_a_exp_eff <= a_exp_eff;
        s0_b_exp_eff <= b_exp_eff;
        s0_a_mant <= a_mant;
        s0_b_mant <= b_mant;
    end

    // Stage 1: registered DSP mantissa product and exponent addition.
    reg s1_special;
    reg [15:0] s1_special_out;
    reg s1_sign;
    reg signed [10:0] s1_exp_base;
    (* use_dsp = "yes" *) reg [15:0] s1_product;
    always @(posedge clk) begin
        s1_special <= s0_special;
        s1_special_out <= s0_special_out;
        s1_sign <= s0_sign;
        s1_exp_base <= $signed({1'b0, s0_a_exp_eff}) +
                       $signed({1'b0, s0_b_exp_eff}) - 11'sd127;
        s1_product <= s0_a_mant * s0_b_mant;
    end

    // Stage 2: leading-one detection.  Keep the priority encoder isolated
    // from exponent correction: at 350 MHz their old combined DSP->CARRY path
    // was the only post-route setup violation replicated across the 32 PEs.
    reg s2_special;
    reg [15:0] s2_special_out;
    reg s2_sign;
    reg [15:0] s2_product;
    reg signed [10:0] s2_exp_base;
    reg signed [5:0] s2_leading;
    always @(posedge clk) begin
        s2_special <= s1_special;
        s2_special_out <= s1_special_out;
        s2_sign <= s1_sign;
        s2_product <= s1_product;
        s2_leading <= highest_set_bit16_pipe(s1_product);
        s2_exp_base <= s1_exp_base;
    end

    // Stage 2B: exponent correction from registered leading-bit metadata.
    // Product/special payloads advance in parallel so arithmetic and exception
    // behaviour remain bit-exact while the multiplier still accepts one input
    // pair every clock.
    reg s2b_special;
    reg [15:0] s2b_special_out;
    reg s2b_sign;
    reg [15:0] s2b_product;
    reg signed [10:0] s2b_exp_work;
    reg signed [5:0] s2b_leading;
    always @(posedge clk) begin
        s2b_special <= s2_special;
        s2b_special_out <= s2_special_out;
        s2b_sign <= s2_sign;
        s2b_product <= s2_product;
        s2b_leading <= s2_leading;
        s2b_exp_work <= s2_exp_base + s2_leading - 11'sd14;
    end

    // Stage 3: perform only leading-dependent alignment and retain the
    // discarded bits.  The previous implementation also compared, rounded,
    // renormalized and updated the exponent in this stage; that created the
    // final path replicated across the 32-PE array.
    reg [8:0] shifted_next;
    reg [31:0] remainder_next;
    reg [31:0] halfway_next;
    reg signed [5:0] shift_next;
    reg round_enable_next;
    always @(*) begin
        shifted_next = 0;
        remainder_next = 0;
        halfway_next = 0;
        shift_next = 0;
        round_enable_next = 1'b0;
        if (s2b_leading >= 7) begin
            shift_next = s2b_leading - 7;
            shifted_next = s2b_product >> shift_next;
            if (shift_next > 0) begin
                round_enable_next = 1'b1;
                remainder_next = s2b_product & ((32'd1 << shift_next) - 1'b1);
                halfway_next = 32'd1 << (shift_next - 1'b1);
            end
        end else if (s2b_leading >= 0) begin
            shifted_next = s2b_product << (7 - s2b_leading);
        end
    end

    reg s3a_special;
    reg [15:0] s3a_special_out;
    reg s3a_sign;
    reg [8:0] s3a_shifted;
    reg [31:0] s3a_remainder;
    reg [31:0] s3a_halfway;
    reg s3a_round_enable;
    reg signed [10:0] s3a_exp_work;
    always @(posedge clk) begin
        s3a_special <= s2b_special;
        s3a_special_out <= s2b_special_out;
        s3a_sign <= s2b_sign;
        s3a_shifted <= shifted_next;
        s3a_remainder <= remainder_next;
        s3a_halfway <= halfway_next;
        s3a_round_enable <= round_enable_next;
        s3a_exp_work <= s2b_exp_work;
    end

    // Stage 4 input logic: round-to-nearest-even, then handle the possible
    // carry into a new hidden bit. This is now independent of the barrel
    // alignment stage above and has its own registered destination.
    reg [8:0] normalized_next;
    reg signed [10:0] exp_next;
    always @(*) begin
        normalized_next = s3a_shifted;
        exp_next = s3a_exp_work;
        if (s3a_round_enable &&
            ((s3a_remainder > s3a_halfway) ||
             ((s3a_remainder == s3a_halfway) && normalized_next[0]))) begin
            normalized_next = normalized_next + 1'b1;
        end
        if (normalized_next[8]) begin
            normalized_next = normalized_next >> 1;
            exp_next = exp_next + 1'b1;
        end
    end

    reg s3_special;
    reg [15:0] s3_special_out;
    reg s3_sign;
    reg [8:0] s3_significand;
    reg signed [10:0] s3_exp_work;
    always @(posedge clk) begin
        s3_special <= s3a_special;
        s3_special_out <= s3a_special_out;
        s3_sign <= s3a_sign;
        s3_significand <= normalized_next;
        s3_exp_work <= exp_next;
    end

    // Stage 5: classify the packed result and perform only the small
    // subnormal shift/mask. Rounding/packing is registered separately.
    localparam [2:0] BF_DIRECT = 3'd0;
    localparam [2:0] BF_SUBNORMAL = 3'd1;
    reg [2:0] s4_class;
    reg [15:0] s4_direct_out;
    reg s4_sign;
    reg [8:0] s4_subnormal_value;
    reg [31:0] s4_subnormal_remainder;
    reg [31:0] s4_subnormal_halfway;
    reg signed [11:0] subnormal_shift;
    always @(posedge clk) begin
        s4_class <= BF_DIRECT;
        s4_direct_out <= 0;
        s4_sign <= s3_sign;
        s4_subnormal_value <= 0;
        s4_subnormal_remainder <= 0;
        s4_subnormal_halfway <= 0;
        if (s3_special) begin
            s4_direct_out <= s3_special_out;
        end else if (s3_exp_work >= 255) begin
            s4_direct_out <= {s3_sign, 8'hff, 7'h00};
        end else if (s3_exp_work > 0) begin
            s4_direct_out <=
                {s3_sign, s3_exp_work[7:0], s3_significand[6:0]};
        end else begin
            subnormal_shift = 1 - s3_exp_work;
            if (subnormal_shift >= 9) begin
                s4_direct_out <= {s3_sign, 15'h0000};
            end else begin
                s4_class <= BF_SUBNORMAL;
                s4_subnormal_value <= s3_significand >> subnormal_shift;
                s4_subnormal_remainder <= s3_significand &
                    ((32'd1 << subnormal_shift) - 1'b1);
                s4_subnormal_halfway <=
                    32'd1 << (subnormal_shift - 1'b1);
            end
        end
    end

    // Stage 6: subnormal round-to-nearest-even and final packing.
    reg [8:0] rounded_subnormal;
    always @(posedge clk) begin
        if (s4_class == BF_SUBNORMAL) begin
            rounded_subnormal = s4_subnormal_value;
            if ((s4_subnormal_remainder > s4_subnormal_halfway) ||
                ((s4_subnormal_remainder == s4_subnormal_halfway) &&
                 rounded_subnormal[0]))
                rounded_subnormal = rounded_subnormal + 1'b1;
            if (rounded_subnormal[7])
                out <= {s4_sign, 8'h01, 7'h00};
            else
                out <= {s4_sign, 8'h00, rounded_subnormal[6:0]};
        end else begin
            out <= s4_direct_out;
        end
    end
endmodule


module gMultiplier(
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] out
);
    reg        result_sign;
    reg [7:0]  a_exp_eff;
    reg [7:0]  b_exp_eff;
    reg [7:0]  a_mant;
    reg [7:0]  b_mant;
    reg [8:0]  significand;
    reg [8:0]  rounded_value;
    reg [31:0] remainder;
    reg [31:0] halfway;

    integer exp_work;
    integer leading_bit;
    integer shift_amount;
    integer subnormal_shift;
    wire [7:0] a_mant_dsp = (a[14:7] == 0) ? {1'b0, a[6:0]} : {1'b1, a[6:0]};
    wire [7:0] b_mant_dsp = (b[14:7] == 0) ? {1'b0, b[6:0]} : {1'b1, b[6:0]};
    // Combinational golden/reference implementation used by arithmetic tests.
    // The physical Flex-DPE uses bf16_mult_pipeline above.
    wire [15:0] mantissa_product;
    assign mantissa_product = a_mant_dsp * b_mant_dsp;
    function integer highest_set_bit16;
        input [15:0] value;
        begin
            if      (value[15]) highest_set_bit16 = 15;
            else if (value[14]) highest_set_bit16 = 14;
            else if (value[13]) highest_set_bit16 = 13;
            else if (value[12]) highest_set_bit16 = 12;
            else if (value[11]) highest_set_bit16 = 11;
            else if (value[10]) highest_set_bit16 = 10;
            else if (value[9])  highest_set_bit16 = 9;
            else if (value[8])  highest_set_bit16 = 8;
            else if (value[7])  highest_set_bit16 = 7;
            else if (value[6])  highest_set_bit16 = 6;
            else if (value[5])  highest_set_bit16 = 5;
            else if (value[4])  highest_set_bit16 = 4;
            else if (value[3])  highest_set_bit16 = 3;
            else if (value[2])  highest_set_bit16 = 2;
            else if (value[1])  highest_set_bit16 = 1;
            else if (value[0])  highest_set_bit16 = 0;
            else                highest_set_bit16 = -1;
        end
    endfunction

    always @(*) begin
        out             = 16'h0000;
        result_sign     = a[15] ^ b[15];
        a_exp_eff       = (a[14:7] == 0) ? 8'd1 : a[14:7];
        b_exp_eff       = (b[14:7] == 0) ? 8'd1 : b[14:7];
        a_mant          = (a[14:7] == 0) ? {1'b0, a[6:0]} : {1'b1, a[6:0]};
        b_mant          = (b[14:7] == 0) ? {1'b0, b[6:0]} : {1'b1, b[6:0]};
        significand     = 9'd0;
        rounded_value   = 9'd0;
        remainder       = 32'd0;
        halfway         = 32'd0;
        exp_work        = 0;
        leading_bit     = -1;
        shift_amount    = 0;
        subnormal_shift = 0;

        // NaNs take precedence.  Return a canonical quiet NaN.
        if (((a[14:7] == 8'hff) && (a[6:0] != 0)) ||
            ((b[14:7] == 8'hff) && (b[6:0] != 0))) begin
            out = 16'h7fc0;
        end else if (((a[14:7] == 8'hff) && (a[6:0] == 0) && (b[14:0] == 0)) ||
                     ((b[14:7] == 8'hff) && (b[6:0] == 0) && (a[14:0] == 0))) begin
            // infinity multiplied by zero is invalid.
            out = 16'h7fc0;
        end else if ((a[14:7] == 8'hff) || (b[14:7] == 8'hff)) begin
            out = {result_sign, 8'hff, 7'h00};
        end else if ((a[14:0] == 0) || (b[14:0] == 0)) begin
            out = {result_sign, 15'h0000};
        end else begin
            leading_bit = highest_set_bit16(mantissa_product);

            // Normalize the product so the hidden bit is significand[7].
            exp_work = a_exp_eff + b_exp_eff - 127 + leading_bit - 14;
            if (leading_bit >= 7) begin
                shift_amount = leading_bit - 7;
                significand = mantissa_product >> shift_amount;
                if (shift_amount > 0) begin
                    remainder = mantissa_product & ((32'd1 << shift_amount) - 1);
                    halfway   = 32'd1 << (shift_amount - 1);
                    if ((remainder > halfway) ||
                        ((remainder == halfway) && significand[0]))
                        significand = significand + 1'b1;
                end
            end else begin
                significand = mantissa_product << (7 - leading_bit);
            end

            // Rounding can create a carry into a new leading bit.
            if (significand[8]) begin
                significand = significand >> 1;
                exp_work = exp_work + 1;
            end

            if (exp_work >= 255) begin
                out = {result_sign, 8'hff, 7'h00};
            end else if (exp_work > 0) begin
                out = {result_sign, exp_work[7:0], significand[6:0]};
            end else begin
                // Convert the normalized significand into a BF16 subnormal.
                subnormal_shift = 1 - exp_work;
                if (subnormal_shift >= 9) begin
                    out = {result_sign, 15'h0000};
                end else begin
                    rounded_value = significand >> subnormal_shift;
                    remainder = significand & ((32'd1 << subnormal_shift) - 1);
                    halfway   = 32'd1 << (subnormal_shift - 1);
                    if ((remainder > halfway) ||
                        ((remainder == halfway) && rounded_value[0]))
                        rounded_value = rounded_value + 1'b1;

                    // A rounded subnormal may become the minimum normal.
                    if (rounded_value[7])
                        out = {result_sign, 8'h01, 7'h00};
                    else
                        out = {result_sign, 8'h00, rounded_value[6:0]};
                end
            end
        end
    end
endmodule
