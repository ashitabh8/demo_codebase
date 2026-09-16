CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
LDFLAGS ?= -lm

KISSFFT_DIR := kissfft
KISSFFT_SRCS := $(KISSFFT_DIR)/kiss_fft.c $(KISSFFT_DIR)/kiss_fftr.c
INCLUDES := -I.

all: spectral_infer spectral_txt_to_bin

spectral_infer: main.c model.c spectral_scaler.c acids_spectral_preprocess.c $(KISSFFT_SRCS) acids_samples.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ main.c model.c spectral_scaler.c acids_spectral_preprocess.c $(KISSFFT_SRCS) $(LDFLAGS)

spectral_txt_to_bin: spectral_txt_to_bin.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ spectral_txt_to_bin.c $(LDFLAGS)

clean:
	rm -f spectral_infer spectral_txt_to_bin libmodel_w8.so libspectral_scaler.so libacids_spectral_preprocess.so
