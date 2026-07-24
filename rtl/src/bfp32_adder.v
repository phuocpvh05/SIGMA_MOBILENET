`timescale 1ns / 1ps
//==============================================================================
// IEEE-754 single-precision adder used by the FAN.
//
// The combinational core carries guard, round, and sticky bits and rounds to
// nearest, ties to even.  The registered wrapper preserves the original FAN
// pipeline interface.
//==============================================================================

module adder32(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] A,
    input  wire [31:0] B,
    output reg  [31:0] O
);
    wire [31:0] sum;

    generalAdder u_adder_comb (
        .a   (A),
        .b   (B),
        .out (sum)
    );

    always @(posedge clk) begin
        if (rst)
            O <= 32'd0;
        else
            O <= sum;
    end
endmodule


// Six-stage, one-result-per-cycle FP32 adder for the timing-critical FAN.
// The arithmetic and special-case behavior matches generalAdder, but exponent
// selection, alignment/addition, normalization and rounding are separated by
// registers so a complete IEEE-754 operation is not placed in one FPGA cycle.
module fp32_add_pipeline(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] out
);
    function [26:0] shift_right_sticky_pipe;
        input [26:0] value;
        input [7:0] amount;
        reg [26:0] shifted;
        reg [26:0] discarded_mask;
        begin
            shifted = 27'd0;
            discarded_mask = 27'd0;
            if (amount == 0) begin
                shifted = value;
            end else if (amount >= 27) begin
                shifted[0] = |value;
            end else begin
                shifted = value >> amount;
                discarded_mask = (27'd1 << amount) - 1'b1;
                shifted[0] = shifted[0] | (|(value & discarded_mask));
            end
            shift_right_sticky_pipe = shifted;
        end
    endfunction

    function [4:0] highest_set_bit27_pipe;
        input [26:0] value;
        begin
            if      (value[26]) highest_set_bit27_pipe = 5'd26;
            else if (value[25]) highest_set_bit27_pipe = 5'd25;
            else if (value[24]) highest_set_bit27_pipe = 5'd24;
            else if (value[23]) highest_set_bit27_pipe = 5'd23;
            else if (value[22]) highest_set_bit27_pipe = 5'd22;
            else if (value[21]) highest_set_bit27_pipe = 5'd21;
            else if (value[20]) highest_set_bit27_pipe = 5'd20;
            else if (value[19]) highest_set_bit27_pipe = 5'd19;
            else if (value[18]) highest_set_bit27_pipe = 5'd18;
            else if (value[17]) highest_set_bit27_pipe = 5'd17;
            else if (value[16]) highest_set_bit27_pipe = 5'd16;
            else if (value[15]) highest_set_bit27_pipe = 5'd15;
            else if (value[14]) highest_set_bit27_pipe = 5'd14;
            else if (value[13]) highest_set_bit27_pipe = 5'd13;
            else if (value[12]) highest_set_bit27_pipe = 5'd12;
            else if (value[11]) highest_set_bit27_pipe = 5'd11;
            else if (value[10]) highest_set_bit27_pipe = 5'd10;
            else if (value[9])  highest_set_bit27_pipe = 5'd9;
            else if (value[8])  highest_set_bit27_pipe = 5'd8;
            else if (value[7])  highest_set_bit27_pipe = 5'd7;
            else if (value[6])  highest_set_bit27_pipe = 5'd6;
            else if (value[5])  highest_set_bit27_pipe = 5'd5;
            else if (value[4])  highest_set_bit27_pipe = 5'd4;
            else if (value[3])  highest_set_bit27_pipe = 5'd3;
            else if (value[2])  highest_set_bit27_pipe = 5'd2;
            else if (value[1])  highest_set_bit27_pipe = 5'd1;
            else               highest_set_bit27_pipe = 5'd0;
        end
    endfunction

    // Stage 1: special cases, magnitude selection and exponent difference.
    reg        s1_special;
    reg [31:0] s1_special_out;
    reg        s1_result_sign;
    reg        s1_same_sign;
    reg [8:0]  s1_exp;
    reg [7:0]  s1_exp_diff;
    reg [26:0] s1_large_ext;
    reg [26:0] s1_small_ext;

    wire a_sign = a[31];
    wire b_sign = b[31];
    wire [7:0] a_exp_eff = (a[30:23] == 0) ? 8'd1 : a[30:23];
    wire [7:0] b_exp_eff = (b[30:23] == 0) ? 8'd1 : b[30:23];
    wire [23:0] a_mant = (a[30:23] == 0) ? {1'b0, a[22:0]} : {1'b1, a[22:0]};
    wire [23:0] b_mant = (b[30:23] == 0) ? {1'b0, b[22:0]} : {1'b1, b[22:0]};
    wire a_is_nan = (a[30:23] == 8'hff) && (a[22:0] != 0);
    wire b_is_nan = (b[30:23] == 8'hff) && (b[22:0] != 0);
    wire a_is_inf = (a[30:23] == 8'hff) && (a[22:0] == 0);
    wire b_is_inf = (b[30:23] == 8'hff) && (b[22:0] == 0);
    wire a_is_zero = (a[30:0] == 0);
    wire b_is_zero = (b[30:0] == 0);
    wire a_larger = (a_exp_eff > b_exp_eff) ||
                    ((a_exp_eff == b_exp_eff) && (a_mant >= b_mant));

    // Datapath registers intentionally have no reset. Their values are
    // ignored until the separately reset FAN/fold valid control is asserted.
    // This avoids replicating a high-fanout reset across every FP32 adder.
    always @(posedge clk) begin
        s1_special <= 1'b0;
        s1_special_out <= 0;
        if (a_is_nan || b_is_nan ||
            (a_is_inf && b_is_inf && (a_sign != b_sign))) begin
            s1_special <= 1'b1;
            s1_special_out <= 32'h7fc00000;
        end else if (a_is_inf) begin
            s1_special <= 1'b1;
            s1_special_out <= a;
        end else if (b_is_inf) begin
            s1_special <= 1'b1;
            s1_special_out <= b;
        end else if (a_is_zero) begin
            s1_special <= 1'b1;
            s1_special_out <= b;
        end else if (b_is_zero) begin
            s1_special <= 1'b1;
            s1_special_out <= a;
        end
        s1_same_sign <= (a_sign == b_sign);
        if (a_larger) begin
            s1_result_sign <= a_sign;
            s1_exp <= {1'b0, a_exp_eff};
            s1_exp_diff <= a_exp_eff - b_exp_eff;
            s1_large_ext <= {a_mant, 3'b000};
            s1_small_ext <= {b_mant, 3'b000};
        end else begin
            s1_result_sign <= b_sign;
            s1_exp <= {1'b0, b_exp_eff};
            s1_exp_diff <= b_exp_eff - a_exp_eff;
            s1_large_ext <= {b_mant, 3'b000};
            s1_small_ext <= {a_mant, 3'b000};
        end
    end

    // Stage 2: exponent alignment. Keeping the variable sticky shift separate
    // from the 28-bit add/subtract removes the remaining ~2.8 ns FAN path.
    reg        s1a_special;
    reg [31:0] s1a_special_out;
    reg        s1a_result_sign;
    reg        s1a_same_sign;
    reg [8:0]  s1a_exp;
    reg [26:0] s1a_large_ext;
    reg [26:0] s1a_aligned_small;
    always @(posedge clk) begin
        s1a_special <= s1_special;
        s1a_special_out <= s1_special_out;
        s1a_result_sign <= s1_result_sign;
        s1a_same_sign <= s1_same_sign;
        s1a_exp <= s1_exp;
        s1a_large_ext <= s1_large_ext;
        s1a_aligned_small <=
            shift_right_sticky_pipe(s1_small_ext, s1_exp_diff);
    end

    // Stage 3: significand add/subtract.
    reg        s2_special;
    reg [31:0] s2_special_out;
    reg        s2_result_sign;
    reg [8:0]  s2_exp;
    reg [26:0] s2_normalized;
    wire [27:0] s1_add_result =
        {1'b0, s1a_large_ext} + {1'b0, s1a_aligned_small};
    always @(posedge clk) begin
        s2_special <= s1a_special;
        s2_special_out <= s1a_special_out;
        s2_result_sign <= s1a_result_sign;
        if (s1a_same_sign) begin
            if (s1_add_result[27]) begin
                s2_normalized <= {s1_add_result[27:2],
                                  s1_add_result[1] | s1_add_result[0]};
                s2_exp <= s1a_exp + 1'b1;
            end else begin
                s2_normalized <= s1_add_result[26:0];
                s2_exp <= s1a_exp;
            end
        end else begin
            s2_normalized <= s1a_large_ext - s1a_aligned_small;
            s2_exp <= s1a_exp;
        end
    end

    // Stage 4: leading-zero detection.  Registering the shift count here
    // separates the priority encoder from the wide normalization barrel
    // shifter; their former combined path was the last 300 MHz violation.
    reg        s2a_special;
    reg [31:0] s2a_special_out;
    reg        s2a_result_sign;
    reg [8:0]  s2a_exp;
    reg [26:0] s2a_normalized;
    reg [5:0]  s2a_normalize_shift;
    reg        s2a_zero;
    always @(posedge clk) begin
        s2a_special <= s2_special;
        s2a_special_out <= s2_special_out;
        s2a_result_sign <= s2_result_sign;
        s2a_exp <= s2_exp;
        s2a_normalized <= s2_normalized;
        s2a_normalize_shift <=
            6'd26 - highest_set_bit27_pipe(s2_normalized);
        s2a_zero <= (s2_normalized == 0);
    end

    // Stage 5: left normalization and exponent adjustment.
    reg        s3_special;
    reg [31:0] s3_special_out;
    reg        s3_result_sign;
    reg [8:0]  s3_exp;
    reg [26:0] s3_normalized;
    always @(posedge clk) begin
        s3_special <= s2a_special;
        s3_special_out <= s2a_special_out;
        s3_result_sign <= s2a_result_sign;
        if (s2a_zero) begin
            s3_normalized <= 0;
            s3_exp <= 0;
        end else begin
            if (s2a_normalize_shift < s2a_exp) begin
                s3_normalized <= s2a_normalized << s2a_normalize_shift;
                s3_exp <= s2a_exp - s2a_normalize_shift;
            end else begin
                s3_normalized <= s2a_normalized << (s2a_exp - 1'b1);
                s3_exp <= 0;
            end
        end
    end

    // Stage 6: round-to-nearest-even and pack.
    reg [24:0] rounded_mant;
    reg [8:0] rounded_exp;
    always @(posedge clk) begin
        if (s3_special) begin
            out <= s3_special_out;
        end else if (s3_normalized == 0) begin
            out <= 0;
        end else begin
            rounded_mant = {1'b0, s3_normalized[26:3]};
            if (s3_normalized[2] &&
                (s3_normalized[1] || s3_normalized[0] || s3_normalized[3]))
                rounded_mant = rounded_mant + 1'b1;
            rounded_exp = s3_exp;
            if (s3_exp == 0) begin
                if (rounded_mant[23])
                    out <= {s3_result_sign, 8'h01, 23'h000000};
                else
                    out <= {s3_result_sign, 8'h00, rounded_mant[22:0]};
            end else begin
                if (rounded_mant[24]) begin
                    rounded_mant = rounded_mant >> 1;
                    rounded_exp = s3_exp + 1'b1;
                end
                if (rounded_exp >= 255)
                    out <= {s3_result_sign, 8'hff, 23'h000000};
                else
                    out <= {s3_result_sign, rounded_exp[7:0], rounded_mant[22:0]};
            end
        end
    end
endmodule


module generalAdder(
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] out
);
    reg        a_sign;
    reg        b_sign;
    reg        result_sign;
    reg [7:0]  a_exp_eff;
    reg [7:0]  b_exp_eff;
    reg [23:0] a_mant;
    reg [23:0] b_mant;
    reg [26:0] a_ext;
    reg [26:0] b_ext;
    reg [26:0] large_ext;
    reg [26:0] small_ext;
    reg [26:0] aligned_small;
    reg [26:0] normalized;
    reg [27:0] add_result;
    reg [24:0] rounded_mant;

    integer exp_work;
    integer exp_diff;
    integer leading_bit;
    integer normalize_shift;
    function integer highest_set_bit27;
        input [26:0] value;
        begin
            if      (value[26]) highest_set_bit27 = 26;
            else if (value[25]) highest_set_bit27 = 25;
            else if (value[24]) highest_set_bit27 = 24;
            else if (value[23]) highest_set_bit27 = 23;
            else if (value[22]) highest_set_bit27 = 22;
            else if (value[21]) highest_set_bit27 = 21;
            else if (value[20]) highest_set_bit27 = 20;
            else if (value[19]) highest_set_bit27 = 19;
            else if (value[18]) highest_set_bit27 = 18;
            else if (value[17]) highest_set_bit27 = 17;
            else if (value[16]) highest_set_bit27 = 16;
            else if (value[15]) highest_set_bit27 = 15;
            else if (value[14]) highest_set_bit27 = 14;
            else if (value[13]) highest_set_bit27 = 13;
            else if (value[12]) highest_set_bit27 = 12;
            else if (value[11]) highest_set_bit27 = 11;
            else if (value[10]) highest_set_bit27 = 10;
            else if (value[9])  highest_set_bit27 = 9;
            else if (value[8])  highest_set_bit27 = 8;
            else if (value[7])  highest_set_bit27 = 7;
            else if (value[6])  highest_set_bit27 = 6;
            else if (value[5])  highest_set_bit27 = 5;
            else if (value[4])  highest_set_bit27 = 4;
            else if (value[3])  highest_set_bit27 = 3;
            else if (value[2])  highest_set_bit27 = 2;
            else if (value[1])  highest_set_bit27 = 1;
            else if (value[0])  highest_set_bit27 = 0;
            else                highest_set_bit27 = -1;
        end
    endfunction

    function [26:0] shift_right_sticky;
        input [26:0] value;
        input integer amount;
        reg sticky;
        reg [26:0] shifted;
        reg [26:0] discarded_mask;
        begin
            sticky = 1'b0;
            shifted = 27'd0;
            discarded_mask = 27'd0;
            if (amount <= 0) begin
                shifted = value;
            end else if (amount >= 27) begin
                shifted = 27'd0;
                shifted[0] = |value;
            end else begin
                shifted = value >> amount;
                // A fixed-width barrel mask avoids a variable-bound loop,
                // which otherwise becomes very expensive when replicated
                // across every FAN adder during FPGA synthesis.
                discarded_mask = (27'd1 << amount) - 1'b1;
                sticky = |(value & discarded_mask);
                shifted[0] = shifted[0] | sticky;
            end
            shift_right_sticky = shifted;
        end
    endfunction

    always @(*) begin
        out             = 32'd0;
        a_sign          = a[31];
        b_sign          = b[31];
        result_sign     = 1'b0;
        a_exp_eff       = (a[30:23] == 0) ? 8'd1 : a[30:23];
        b_exp_eff       = (b[30:23] == 0) ? 8'd1 : b[30:23];
        a_mant          = (a[30:23] == 0) ? {1'b0, a[22:0]} : {1'b1, a[22:0]};
        b_mant          = (b[30:23] == 0) ? {1'b0, b[22:0]} : {1'b1, b[22:0]};
        a_ext           = {a_mant, 3'b000};
        b_ext           = {b_mant, 3'b000};
        large_ext       = 27'd0;
        small_ext       = 27'd0;
        aligned_small   = 27'd0;
        normalized      = 27'd0;
        add_result      = 28'd0;
        rounded_mant    = 25'd0;
        exp_work        = 0;
        exp_diff        = 0;
        leading_bit     = -1;
        normalize_shift = 0;

        // NaN propagation and infinity arithmetic.
        if (((a[30:23] == 8'hff) && (a[22:0] != 0)) ||
            ((b[30:23] == 8'hff) && (b[22:0] != 0))) begin
            out = 32'h7fc00000;
        end else if ((a[30:23] == 8'hff) && (b[30:23] == 8'hff) &&
                     (a[22:0] == 0) && (b[22:0] == 0) &&
                     (a_sign != b_sign)) begin
            out = 32'h7fc00000;
        end else if ((a[30:23] == 8'hff) && (a[22:0] == 0)) begin
            out = a;
        end else if ((b[30:23] == 8'hff) && (b[22:0] == 0)) begin
            out = b;
        end else if (a[30:0] == 0) begin
            out = b;
        end else if (b[30:0] == 0) begin
            out = a;
        end else begin
            // Select the larger magnitude operand before exponent alignment.
            if ((a_exp_eff > b_exp_eff) ||
                ((a_exp_eff == b_exp_eff) && (a_mant >= b_mant))) begin
                large_ext   = a_ext;
                small_ext   = b_ext;
                exp_work    = a_exp_eff;
                exp_diff    = a_exp_eff - b_exp_eff;
                result_sign = a_sign;
            end else begin
                large_ext   = b_ext;
                small_ext   = a_ext;
                exp_work    = b_exp_eff;
                exp_diff    = b_exp_eff - a_exp_eff;
                result_sign = b_sign;
            end

            aligned_small = shift_right_sticky(small_ext, exp_diff);

            if (a_sign == b_sign) begin
                add_result = {1'b0, large_ext} + {1'b0, aligned_small};
                result_sign = a_sign;
                if (add_result[27]) begin
                    normalized = add_result[27:1];
                    normalized[0] = normalized[0] | add_result[0];
                    exp_work = exp_work + 1;
                end else begin
                    normalized = add_result[26:0];
                end
            end else begin
                normalized = large_ext - aligned_small;
            end

            if (normalized == 0) begin
                out = 32'd0;
            end else begin
                // Normalize subtraction results and sums of subnormal values.
                leading_bit = highest_set_bit27(normalized);

                if (leading_bit < 26) begin
                    normalize_shift = 26 - leading_bit;
                    if (normalize_shift < exp_work) begin
                        normalized = normalized << normalize_shift;
                        exp_work = exp_work - normalize_shift;
                    end else begin
                        // Stop at the subnormal exponent instead of underflowing.
                        normalized = normalized << (exp_work - 1);
                        exp_work = 0;
                    end
                end

                // Round using guard/round/sticky bits.
                rounded_mant = {1'b0, normalized[26:3]};
                if (normalized[2] &&
                    (normalized[1] || normalized[0] || normalized[3]))
                    rounded_mant = rounded_mant + 1'b1;

                if (exp_work == 0) begin
                    if (rounded_mant[23])
                        out = {result_sign, 8'h01, 23'h000000};
                    else
                        out = {result_sign, 8'h00, rounded_mant[22:0]};
                end else begin
                    if (rounded_mant[24]) begin
                        rounded_mant = rounded_mant >> 1;
                        exp_work = exp_work + 1;
                    end

                    if (exp_work >= 255)
                        out = {result_sign, 8'hff, 23'h000000};
                    else
                        out = {result_sign, exp_work[7:0], rounded_mant[22:0]};
                end
            end
        end
    end
endmodule
