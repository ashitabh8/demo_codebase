CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -I.
LDFLAGS = -lm

.PHONY: all clean run

all: run_model

run_model: main.c model.c model.h weights.h nn_ops_float.h nn_ops_int8.h true_labels.h
	$(CC) $(CFLAGS) -o run_model main.c model.c $(LDFLAGS)

run: run_model
	./run_model

clean:
	rm -f run_model
