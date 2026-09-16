/*
 * Convert ACIDS sample .txt (header + floats) to raw .bin for spectral_infer.
 *
 * Expected txt layout:
 *   sample_name
 *   label_id
 *   RAW_AUDIO_STREAM
 *   1600
 *   <1600 floats, one per line>
 *
 * usage: ./spectral_txt_to_bin <input.txt> <output.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acids_spectral_preprocess.h"

#define MAX_LINE 4096

static int parse_count_line(const char *line, int *count)
{
    if (sscanf(line, "%d", count) != 1) {
        return -1;
    }
    if (*count != ACIDS_SPECTRAL_INPUT_SAMPLES) {
        fprintf(stderr, "expected count %d, got %d\n", ACIDS_SPECTRAL_INPUT_SAMPLES, *count);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    FILE *in;
    FILE *out;
    char line[MAX_LINE];
    float values[ACIDS_SPECTRAL_INPUT_SAMPLES];
    int expected_count = 0;
    int value_count = 0;
    int line_no = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.txt> <output.bin>\n", argv[0]);
        return 1;
    }

    in = fopen(argv[1], "r");
    if (in == NULL) {
        fprintf(stderr, "failed to open input: %s\n", argv[1]);
        return 1;
    }

    if (fgets(line, sizeof(line), in) == NULL) {
        fprintf(stderr, "empty input file\n");
        fclose(in);
        return 1;
    }
    line_no += 1;

    if (fgets(line, sizeof(line), in) == NULL) {
        fprintf(stderr, "missing label id line\n");
        fclose(in);
        return 1;
    }
    line_no += 1;

    if (fgets(line, sizeof(line), in) == NULL) {
        fprintf(stderr, "missing layout line\n");
        fclose(in);
        return 1;
    }
    line_no += 1;
    if (strncmp(line, "RAW_AUDIO_STREAM", 16) != 0) {
        fprintf(stderr, "expected layout RAW_AUDIO_STREAM, got: %s", line);
        fclose(in);
        return 1;
    }

    if (fgets(line, sizeof(line), in) == NULL) {
        fprintf(stderr, "missing count line\n");
        fclose(in);
        return 1;
    }
    line_no += 1;
    line[strcspn(line, "\r\n")] = '\0';
    if (parse_count_line(line, &expected_count) != 0) {
        fclose(in);
        return 1;
    }

    while (fgets(line, sizeof(line), in) != NULL) {
        line_no += 1;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        if (value_count >= ACIDS_SPECTRAL_INPUT_SAMPLES) {
            fprintf(stderr, "too many values at line %d\n", line_no);
            fclose(in);
            return 1;
        }
        values[value_count] = (float)strtod(line, NULL);
        value_count += 1;
    }
    fclose(in);

    if (value_count != ACIDS_SPECTRAL_INPUT_SAMPLES) {
        fprintf(
            stderr,
            "expected %d values, got %d\n",
            ACIDS_SPECTRAL_INPUT_SAMPLES,
            value_count
        );
        return 1;
    }

    out = fopen(argv[2], "wb");
    if (out == NULL) {
        fprintf(stderr, "failed to open output: %s\n", argv[2]);
        return 1;
    }
    if (fwrite(values, sizeof(float), ACIDS_SPECTRAL_INPUT_SAMPLES, out)
        != (size_t)ACIDS_SPECTRAL_INPUT_SAMPLES) {
        fprintf(stderr, "failed to write output\n");
        fclose(out);
        return 1;
    }
    fclose(out);
    return 0;
}
