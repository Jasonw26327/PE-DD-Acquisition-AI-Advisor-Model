.PHONY: setup test corpus run clean

setup:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt pytest
	python preflight.py

test:
	pytest tests/ -v

corpus:
	python src/corpus_builder.py --n 720 --pairs 90 --out-prefix data/corpus_rebuilt

run:
	python src/testbed.py --stage all

# Same phases, one at a time, for machines with a wall-clock cap.
run-staged:
	python src/testbed.py --stage baseline
	python src/testbed.py --stage teacher --teacher-steps 150 --lr 3e-3
	python src/testbed.py --stage adversarial
	python src/testbed.py --stage extract --budgets 2 4 6 8 16 32 \
		--student-steps 120 --lr 3e-3

clean:
	rm -rf artifacts run_* metrics.json .pytest_cache __pycache__
