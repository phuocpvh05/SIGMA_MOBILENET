# MobileNet MNIST-10K robustness benchmark

This benchmark runs the same 10,000 MNIST labels under three deterministic
input conditions on both Cortex-A53 and the SIGMA FPGA accelerator:

- `clean`: original MNIST test images.
- `broken`: two short segments are removed from real ink pixels.
- `poor`: up to two-pixel translation, contrast loss, Gaussian noise and one
  local erased region.

The degradation seeds are fixed at 2026 (`broken`) and 2027 (`poor`). The
generator records SHA-256 hashes in
`generated/mobilenet_mnist10k_manifest.json`, so the test is reproducible.

## Run from Vitis GUI

No Vivado synthesis, implementation, bitstream generation, XSA export or
platform rebuild is required. The RTL and hardware platform are unchanged;
only the bare-metal ELF and its embedded test images change.

1. Select the `sigma_mobilenet_a53_benchmark` application component.
2. Use **Clean** on that application component. This is required because the
   assembler `.incbin` files are external binary dependencies.
3. Click **Build**. The ELF contains three 15,680,000-byte BF16 image sets plus
   one common 10,000-byte label set.
4. Open the Genesys ZU UART at 115200 baud (the previously verified COM port).
5. Use the existing hardware launch configuration and click **Run**.

The UART prints progress and independent accuracy/performance results for
`clean`, `broken` and `poor`, followed by a three-row robustness summary. With
the previously measured board time, all three conditions should take roughly
6-7 minutes after the ELF has been downloaded through JTAG.

Visible PNG evidence decoded from the exact BF16 board payloads is stored in
`pic/mobilenet_mnist10k_conditions/`.
