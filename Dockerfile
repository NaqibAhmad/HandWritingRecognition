From python:3.10.3 as build
FROM tensorflow/tensorflow:2.17.0
WORKDIR /app


COPY ./mlruns/ ./mlruns/
COPY ./requirements.txt ./requirements.txt
COPY ./app.py ./app.py

ENV VIRTUAL_ENV=/home/packages/.venv
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 777 /install.sh && /install.sh && rm /install.sh
RUN /root/.cargo/bin/uv venv /home/packages/.venv
RUN /root/.cargo/bin/uv pip install --no-cache --system -r requirements.txt

CMD ["python", "app.py"]