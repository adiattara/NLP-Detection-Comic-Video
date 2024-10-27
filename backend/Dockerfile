
FROM python:3.9


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

COPY .env /code/.env

COPY init.sh /code/init.sh

RUN chmod +x /code/init.sh
RUN /code/init.sh

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Télécharger les stopwords de NLTK
RUN python -c "import nltk; nltk.download('stopwords')"

# Télécharger le modèle français de spaCy
RUN python -m spacy download fr_core_news_sm

COPY  *.py /code/


