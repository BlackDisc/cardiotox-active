# cardiotox-active

## Repository structure
    .
    ├── data                       # Used data
    ├── model                      # Model files (parameters and selected features)
    ├── notebooks                  # Notebooks (building model)
    ├── src                        # Code
    ├── tests                      # Tests
    ├── Dockerfile                 # Dockerfile for web service 
    ├── app.py                     # Main web service file
    ├── requirements.txt           # Requirements for web service
    ├── requirements_notebook.txt  # Requirements for notebook
    └── README.md

## Web service
To deploy service on http://127.0.0.1:5000
```
docker build . -t cardiotox-active:0.1
docker run --rm -p 5000:5000 cardiotox-active:0.1
```
## Usage example
To predict activity for molecule with smile `COc1ccc(-c2cc(-c3ccc(C(=O)N(C)C)cc3)cnc2N)cn1`
```
curl -d '{"smiles":["COc1ccc(-c2cc(-c3ccc(C(=O)N(C)C)cc3)cnc2N)cn1"]}' -H "Content-Type: application/json" http://127.0.0.1:5000/predict
```
Response:`{"predictions":[0]}`

Service accept multiple smiles in one query
```
curl -d '{"smiles":["COc1ccc(-c2cc(-c3ccc(C(=O)N(C)C)cc3)cnc2N)cn1", "O=C(c1ccncc1)N1CCC2(CCN(Cc3cccc(Oc4ccccc4)c3)CC2)CC1"]}' -H "Content-Type: application/json" http://127.0.0.1:5000/predict
```
Response:`{"predictions":[0,1]}`

## Tests
To run functional test
```
python -m pytest
```
To run load tests
```
locust -f tests/load/locustfile.py
```
## Model building
ML model was prepared using jupyter notebook and is avaliable in `notebooks` folder

To install notebook dependencies
```
pip install -r requirements_notebook.txt
```


