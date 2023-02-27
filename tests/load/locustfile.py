import time
from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 10)

    @task
    def prediction(self):
        with self.client.post("/predict", json={"smiles":["COc1ccc(-c2cc(-c3ccc(C(=O)N(C)C)cc3)cnc2N)cn1"]}, catch_response=True, timeout=30) as response:
            if response.content == b"":
                response.failure("No data")
