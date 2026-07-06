/*
 * Convert ACIDS sample .txt (header + floats) to raw .bin for acids_infer.
 *
 * usage: ./acids_txt_to_bin <input.txt> <output.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acids_mel_preprocess.h"

#define RAW_INPUT_SIZE (ACIDS_RAW_AUDIO_CHANNELS * ACIDS_NUM_SEGMENTS * ACIDS_RAW_SEG_SAMPLES)
#define MAX_LINE 4096

static int parse_shape_line(const char *line, int *c, int *seg, int *t)
{
    if (sscanf(line, "%d,%d,%d", c, seg, t) != 3) {
        return -1;
    }
    if (*c != ACIDS_RAW_AUDIO_CHANNELS || *seg != ACIDS_NUM_SEGMENTS || *t != ACIDS_RAW_SEG_SAMPLES) {
        fprintf(stderr, "expected shape 3,7,25, got %s\n", line);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    FILE *in;
    FILE *out;
    char line[MAX_LINE];
    float values[RAW_INPUT_SIZE];
    int c = 0;
    int seg = 0;
    int t = 0;
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
    if (strncmp(line, "RAW_AUDIO_CHW", 13) != 0) {
        fprintf(stderr, "expected layout RAW_AUDIO_CHW, got: %s", line);
        fclose(in);
        return 1;
    }

    if (fgets(line, sizeof(line), in) == NULL) {
        fprintf(stderr, "missing shape line\n");
        fclose(in);
        return 1;
    }
    line_no += 1;
    line[strcspn(line, "\r\n")] = '\0';
    if (parse_shape_line(line, &c, &seg, &t) != 0) {
        fclose(in);
        return 1;
    }

    while (fgets(line, sizeof(line), in) != NULL) {
        line_no += 1;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        if (value_count >= RAW_INPUT_SIZE) {
            fprintf(stderr, "too many values at line %d\n", line_no);
            fclose(in);
            return 1;
        }
        values[value_count] = (float)strtod(line, NULL);
        value_count += 1;
    }
    fclose(in);

    if (value_count != RAW_INPUT_SIZE) {
        fprintf(stderr, "expected %d values, got %d\n", RAW_INPUT_SIZE, value_count);
        return 1;
    }

    out = fopen(argv[2], "wb");
    if (out == NULL) {
        fprintf(stderr, "failed to open output: %s\n", argv[2]);
        return 1;
    }
    if (fwrite(values, sizeof(float), RAW_INPUT_SIZE, out) != RAW_INPUT_SIZE) {
        fprintf(stderr, "failed to write output\n");
        fclose(out);
        return 1;
    }
    fclose(out);
    return 0;
}
