.PHONY: install lint test docker-build docker-run 

install: ## install lfire with the default pip
	pip install -e .
	
lint: ## check style with flake8
	flake8 pylfire tests

test:  ## run tests with the default Python
	python -m pytest tests

docker-build:  ## build a docker image suitable for running LFIRE
	docker build --rm -t pylfire .

docker-run:  ## run a docker container with a lfire directory and publish port 8888 for Jupyter
	docker run --rm -v ${PWD}:/pylfire -w /pylfire -it -p 8888:8888 pylfire jupyter notebook --ip 0.0.0.0 --no-browser
