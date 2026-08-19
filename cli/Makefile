CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=

CORE_DIR = ../r-package/fexactnet/src
TARGET = fisher-network
TEST_TARGET = test-core
SOURCES = fisher_network_cli.c $(CORE_DIR)/fisher_network_core.c
HEADERS = $(CORE_DIR)/fisher_network_core.h $(CORE_DIR)/version.h

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) $(SOURCES) -o $@ $(LDFLAGS) -lm

$(TEST_TARGET): test_core.c $(CORE_DIR)/fisher_network_core.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) test_core.c \
		$(CORE_DIR)/fisher_network_core.c -o $@ $(LDFLAGS) -lm

test: $(TARGET) $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	$(RM) $(TARGET) fisher-network.exe $(TEST_TARGET) test-core.exe
