project=boson_higgs_app
dockerfile=docker/.Dockerfile-server

jupyter:
	poetry run jupyter lab

data-pulling:
	poetry run python scripts/data_pulling.py

data-full-preprocessing:
	poetry run python scripts/features_handling.py

train-model:
	poetry run python scripts/trigger_experiment.py

api-server-poetry:
	poetry run uvicorn api.main:app --reload --workers 1 --host 0.0.0.0 --port 8008

docker-build-server:
	docker build -f ${dockerfile} -t ${project} .

docker-run-server:
	docker run -p 8008:8008 ${project}

batch-prediction:
	poetry run python batch_prediction/model.py --csv_file $(csv_file)