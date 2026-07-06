CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
LDFLAGS ?= -lm

all: acids_infer

acids_infer: main.c model.c acids_mel_preprocess.c
	$(CC) $(CFLAGS) -o $@ main.c model.c acids_mel_preprocess.c $(LDFLAGS)

clean:
	rm -f acids_infer
