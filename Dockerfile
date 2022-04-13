FROM python:3

RUN pip3 install virtualenv

RUN python3 -m venv venvs

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["flask", "run"]