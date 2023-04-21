# target: test - run tests from tests/ folder
test:
	pytest --cov-report term-missing --cov --doctest-modules

# target: lint - run code style tests for api/ folder
lint:
	pylint api/ --rcfile .pylintrc

# target: format - run automatic code styler (black)
format:
	black api/ -l 80

# target: isort - sort imports
isort:
	isort api/ --profile black -l 80

# target: build - build `api-module-image` .docker image
build:
	docker compose --project-name nn-api build

# target: celery - run celery service (using .env variables)
celery:
	celery -A api.celery_service.worker worker -l info --pool=solo
