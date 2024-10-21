# Makefile


setup:
	 bash  ./setup.sh

# Cible pour entraîner le modèle
train:
	python src/main.py train \
		--input_filename=data/raw/train.csv \
		--model_function=$(MODEL_FUNCTION) \
		--model_dump_filename=models/$(MODEL_NAME).json
# Cible pour évaluer le modèle
evaluate:
	python src/main.py evaluate \
		--input_filename=data/raw/train.csv \
		--model_function=$(MODEL_FUNCTION)
# Cible pour faire des prédictions
predict:
	python src/main.py predict \
		--input_filename=data/raw/test.csv \
		--model_dump_filename=models/$(MODEL_NAME).json\
		--output_filename=data/processed/predictions_$(MODEL_NAME).json

validate:
	python src/main.py validate \
		--input_filename=data/raw/test.csv \
		--model_dump_filename=models/$(MODEL_NAME).json
.PHONY: train evaluate predict validate setup
