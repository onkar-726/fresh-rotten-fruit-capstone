# Convenience wrappers - every target is just a command from the README.
PY ?= python3
PROFILE ?= configs/default.json

.PHONY: install test test-fast data check-gpu train-cnn train-mobilenet train evaluate compare analysis app package manual

install:            ## laptop / general install
	$(PY) -m pip install -r requirements.txt
test:               ## full test-suite (CPU, synthetic data; ~10 min)
	$(PY) -m pytest tests -q
test-fast:          ## unit tests only
	$(PY) -m pytest tests -q --ignore=tests/test_pipeline_smoke.py --ignore=tests/test_scripts_help.py
data:               ## split + verification (needs the dataset; see README)
	$(PY) scripts/prepare_data.py --audit
check-gpu:
	$(PY) scripts/check_gpu.py
train-cnn:
	$(PY) scripts/train.py --model baseline_cnn --config $(PROFILE)
train-mobilenet:
	$(PY) scripts/train.py --model mobilenet_v2 --config $(PROFILE)
train: train-cnn train-mobilenet
evaluate:
	$(PY) scripts/evaluate.py
compare:
	$(PY) scripts/compare.py
analysis:
	$(PY) scripts/error_analysis.py --run baseline_cnn && $(PY) scripts/error_analysis.py --run mobilenet_v2
app:
	$(PY) app/app.py
package:
	$(PY) scripts/package_results.py --out dgx_results.zip
manual:
	$(PY) tools/build_manual.py
