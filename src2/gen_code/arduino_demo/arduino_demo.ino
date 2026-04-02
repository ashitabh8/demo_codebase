#include <Arduino.h>

#include "model.h"
#include "demo_samples.h"

// Build-time knobs.
// If your generated model outputs a different count, override MODEL_OUTPUT_SIZE.
#ifndef MODEL_OUTPUT_SIZE
#define MODEL_OUTPUT_SIZE 3
#endif

static float output_buf[MODEL_OUTPUT_SIZE];
static bool running = false;
static int sample_cursor = 0;
static unsigned long last_emit_ms = 0;
static const unsigned long kEmitIntervalMs = 120;

static void print_ready() {
  Serial.println("ready src=arduino commands=START,STOP,RESET,STATUS");
}

static void print_status() {
  Serial.print("status src=arduino running=");
  Serial.print(running ? 1 : 0);
  Serial.print(" cursor=");
  Serial.print(sample_cursor);
  Serial.print(" total=");
  Serial.println(DEMO_NUM_SAMPLES);
}

static void send_prediction_line(int sample_id, int pred, float inf_ms) {
  Serial.print("sample_");
  Serial.print(sample_id);
  Serial.print(" src=arduino pred=");
  Serial.print(pred);
  Serial.print(" target=");
  Serial.print(DEMO_TARGETS[sample_id]);
  Serial.print(" inf_ms=");
  Serial.print(inf_ms, 3);
  Serial.print(" logits=");
  for (int i = 0; i < MODEL_OUTPUT_SIZE; ++i) {
    if (i > 0) Serial.print(",");
    Serial.print(output_buf[i], 6);
  }
  Serial.println();
}

static int argmax_logits() {
  int best = 0;
  for (int i = 1; i < MODEL_OUTPUT_SIZE; ++i) {
    if (output_buf[i] > output_buf[best]) best = i;
  }
  return best;
}

static void run_one_sample() {
  if (sample_cursor >= DEMO_NUM_SAMPLES) {
    running = false;
    Serial.println("done src=arduino");
    return;
  }

  unsigned long t0 = micros();
  model_forward(DEMO_SAMPLES[sample_cursor], output_buf);
  unsigned long t1 = micros();
  float inf_ms = (float)(t1 - t0) / 1000.0f;

  int pred = argmax_logits();
  send_prediction_line(sample_cursor, pred, inf_ms);
  sample_cursor += 1;
}

static void handle_command(const String& cmd_raw) {
  String cmd = cmd_raw;
  cmd.trim();
  cmd.toUpperCase();
  if (cmd == "START") {
    running = true;
    Serial.println("ack src=arduino cmd=START");
    return;
  }
  if (cmd == "STOP") {
    running = false;
    Serial.println("ack src=arduino cmd=STOP");
    return;
  }
  if (cmd == "RESET") {
    running = false;
    sample_cursor = 0;
    Serial.println("ack src=arduino cmd=RESET");
    return;
  }
  if (cmd == "STATUS") {
    print_status();
    return;
  }
  if (cmd.length() > 0) {
    Serial.print("error src=arduino unknown_cmd=");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(5);
  }
  print_ready();
  print_status();
}

void loop() {
  while (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    handle_command(line);
  }

  if (!running) {
    delay(10);
    return;
  }

  unsigned long now = millis();
  if (now - last_emit_ms >= kEmitIntervalMs) {
    last_emit_ms = now;
    run_one_sample();
  }
}
