PYTHON ?= python

YAML ?= /home/misra8/demo_codebase/src2/data/ACIDS.yaml
CHECKPOINT ?=
MODEL_NAME ?= student_audio_deepsense_dw_simple_tiny_w8a8
FLOAT_MODEL_NAME ?= student_audio_deepsense_dw_simple_tiny
NUM_SAMPLES ?= 50
CALIB_BATCHES ?= 50
SEED ?= 1234
TOL ?= 0.5

.PHONY: help quick full compile export build run compare-dir compare-file clean

help:
	@echo "Targets:"
	@echo "  make quick                     # C-only check using saved expected outputs"
	@echo "  make full                      # Compile + export + C run + compare"
	@echo "  make compile                   # Generate W8A8 C from checkpoint"
	@echo "  make export                    # Export test data and expected outputs"
	@echo "  make build                     # Build test_inference binary"
	@echo "  make run                       # Run C inference to c_outputs.txt"
	@echo "  make compare-dir               # Compare with test_data/pytorch_output_*.txt"
	@echo "  make compare-file              # Compare with expected_outputs.txt"
	@echo "  make clean                     # Remove run artifacts"
	@echo ""
	@echo "Override variables, e.g.:"
	@echo "  make full CHECKPOINT=/path/to/best_model.pth NUM_SAMPLES=50"

quick:
	$(PYTHON) run_validation.py quick-c-check --tolerance $(TOL)

full:
	$(PYTHON) run_validation.py full-checkpoint-check \
		--yaml_path $(YAML) \
		--model_name $(MODEL_NAME) \
		--float_model_name $(FLOAT_MODEL_NAME) \
		--calib_batches $(CALIB_BATCHES) \
		--num_samples $(NUM_SAMPLES) \
		--seed $(SEED) \
		--tolerance $(TOL) \
		$(if $(CHECKPOINT),--checkpoint_path $(CHECKPOINT),)

compile:
	$(PYTHON) compile_w8a8.py \
		--yaml_path $(YAML) \
		--model_name $(MODEL_NAME) \
		--float_model_name $(FLOAT_MODEL_NAME) \
		--calib_batches $(CALIB_BATCHES) \
		$(if $(CHECKPOINT),--checkpoint_path $(CHECKPOINT),)

export:
	$(PYTHON) export_test_data_w8a8.py \
		--yaml_path $(YAML) \
		--model_name $(MODEL_NAME) \
		--num_samples $(NUM_SAMPLES) \
		--seed $(SEED) \
		$(if $(CHECKPOINT),--checkpoint_path $(CHECKPOINT),)

build:
	gcc -O2 -I generated/ -o test_inference generated/model.c test_main.c -lm

run:
	./test_inference > c_outputs.txt

compare-dir:
	$(PYTHON) compare_outputs.py \
		--c_output_path c_outputs.txt \
		--reference_dir test_data \
		--report_csv_path comparison_report.csv \
		--tolerance $(TOL)

compare-file:
	$(PYTHON) compare_outputs.py \
		--c_output_path c_outputs.txt \
		--reference_file expected_outputs.txt \
		--report_csv_path comparison_report.csv \
		--tolerance $(TOL)

clean:
	rm -f test_inference c_outputs.txt comparison_report.csv
