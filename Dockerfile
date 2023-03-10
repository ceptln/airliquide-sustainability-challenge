FROM python:3.9

WORKDIR /app

COPY poetry.lock pyproject.toml /app/
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

COPY . /app/

CMD ["poetry", "run", "streamlit", "run", "h2_station_distributor/Home.py"]
