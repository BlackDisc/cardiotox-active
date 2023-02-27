# start by pulling the python image
FROM python:3.8

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update
RUN apt-get install git
RUN git clone https://github.com/Abdulk084/CardioTox
RUN cd CardioTox/PyBioMed && python setup.py install

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

EXPOSE 5000

# configure the container to run in an executed manner
ENTRYPOINT [ "gunicorn", "app:app","-b", "0.0.0.0:5000"]
