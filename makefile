setup:
	make init
	make test
	make doc

init:
	pip install -r requirements.txt

test:
	python3 app/tests/test_suite.py

cover:
	coverage run --source=analysis,app,independent,relational app/tests/test_suite.py
	COVERALLS_REPO_TOKEN=zyRrjmUTrnhmBAuVn3OQQwIpvFv3AOHNs coveralls

doc:
	pdoc --overwrite --html app/config.py --html-dir docs/
	pdoc --overwrite --html independent/scripts/classification.py --html-dir docs/
	pdoc --overwrite --html independent/scripts/content_features.py --html-dir docs/
	pdoc --overwrite --html independent/scripts/graph_features.py --html-dir docs/
	pdoc --overwrite --html independent/scripts/independent.py --html-dir docs/
	pdoc --overwrite --html independent/scripts/relational_features.py --html-dir docs/
	pdoc --overwrite --html relational/scripts/comments.py --html-dir docs/
	pdoc --overwrite --html relational/scripts/generator.py --html-dir docs/
	pdoc --overwrite --html relational/scripts/pred_builder.py --html-dir docs/
	pdoc --overwrite --html relational/scripts/psl.py --html-dir docs/
	pdoc --overwrite --html relational/scripts/relational.py --html-dir docs/
	pdoc --overwrite --html relational/scripts/tuffy.py --html-dir docs/
	pdoc --overwrite --html analysis/analysis.py --html-dir docs/
	pdoc --overwrite --html analysis/connections.py --html-dir docs/
	pdoc --overwrite --html analysis/evaluation.py --html-dir docs/
	pdoc --overwrite --html analysis/interpretability.py --html-dir docs/
	pdoc --overwrite --html analysis/label.py --html-dir docs/
	pdoc --overwrite --html analysis/purity.py --html-dir docs/
	pdoc --overwrite --html analysis/util.py --html-dir docs/
