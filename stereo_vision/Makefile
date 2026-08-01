CC = gcc

# define params (-Wall: show warnings, -g: add debug info, -I : srearch from root)                       
# compile params
CFLAGS = -Wall -O2 -fPIC \
	-I. \
	-IPSO/include \
	-IDIC/include \
	-Itools/interpolation/include

LDFLAGS = -lm -shared -Wl,--export-all-symbols      # link params

BUILD_DIR            =      build

OUTPUT_PSO           =      PSO.dll
OUTPUT_INTERP        =      cubic_interp.dll
OUTPUT_ICGN          =      ICGN.dll

TARGET_PSO           =      build/$(OUTPUT_PSO)
TARGET_INTERP        =      build/$(OUTPUT_INTERP)
TARGET_ICGN          =      build/$(OUTPUT_ICGN)

# define all source (find .c)
SRCS_PSO             =      $(wildcard PSO/src/core/PSO_core.c) \
                            $(wildcard PSO/src/core/PSO.c) \
							$(wildcard PSO/src/core/ICGN_core.c) \
                            $(wildcard PSO/src/factory/PSO_factory.c) \
                            $(wildcard PSO/src/interp/interpolation_core.c) \
                            $(wildcard PSO/src/system/sys_maneger.c) \
                            $(wildcard PSO/src/system/system_comm.c) \

HEADERS_PSO          =      $(wildcard PSO/include/*.h) \
                            $(wildcard DIC/include/*.h) \
                            $(wildcard tools/interpolation/include/*.h)

SRCS_INTERP          =      $(wildcard tools/interpolation/src/cubic_interp.c)
HEADERS_INTERP       =      $(wildcard tools/interpolation/include/*.h)
SRCS_ICGN            =      $(wildcard DIC/ICGN_subset_warping.c)
HEADERS_ICGN         =      $(wildcard DIC/include/*.h)

.PHONY: check
check:
	@echo $(SRCS_PSO)

.PHONY: prepare
prepare:
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

.PHONY: all
all: prepare $(TARGET_PSO) $(TARGET_INTERP) $(TARGET_ICGN)

$(TARGET_PSO): $(SRCS_PSO)
	$(CC) $(CFLAGS) $(SRCS_PSO) -o $(TARGET_PSO) $(LDFLAGS)

$(TARGET_INTERP): $(SRCS_INTERP)
	$(CC) $(CFLAGS) $(SRCS_INTERP) -o $(TARGET_INTERP) $(LDFLAGS)

$(TARGET_ICGN): $(SRCS_ICGN)
	$(CC) $(CFLAGS) $(SRCS_ICGN) -o $(TARGET_ICGN) $(LDFLAGS)

.PHONY: clean
clean:
	@if exist $(BUILD_DIR) del /f /q $(BUILD_DIR)\*.dll


# $^: source files
# $@: target files