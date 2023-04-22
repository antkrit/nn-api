# target: test - run tests from tests/ folder
test:
	pytest --cov-report term-missing --cov api/ --doctest-modules

# target: lint - run code style tests for api/ folder
lint:
	pylint api/

# target: format - run automatic code styler (black)
format:
	black api/ -l 80

# target: isort - sort imports
isort:
	isort api/ --profile black -l 80
