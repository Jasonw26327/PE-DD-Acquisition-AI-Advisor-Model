.PHONY: setup test corpus run run-teacher run-student defense clean

setup:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt pytest
	python preflight.py

test:
	pytest tests/ -v

corpus:
	python src/corpus_builder.py --n 720 --pairs 90 --adversarial 100

run:
	python src/testbed.py --stage all

run-teacher:
	python src/testbed.py --stage teacher --teacher-steps 150 --lr 3e-3

run-student:
	python src/testbed.py --stage extract --budgets 2 4 6 8 16 32 --student-steps 120 --lr 3e-3

defense:
	python src/testbed.py --stage extract --budgets 8 --defense-rate 0.15

clean:
	rm -rf artifacts run_* metrics.json .pytest_cache __pycache__
