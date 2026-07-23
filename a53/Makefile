CC ?= gcc
CFLAGS ?= -O3 -mcpu=cortex-a53 -std=c11 -Wall -Wextra -Wpedantic -Igenerated
TARGET := sigma_mobilenet_bench
SOURCES := sigma_mobilenet_linux.c sigma_mobilenet_reference.c

.PHONY: all generate clean

all: generate $(TARGET)

generate:
	python3 generate_mobilenet_c_artifacts.py

$(TARGET): $(SOURCES) sigma_mobilenet_reference.h generated/mobilenet_layers_generated.h
	$(CC) $(CFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET) sigma_mobilenet_a53_vs_fpga.json
