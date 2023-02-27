import pytest
import json


def test_intro(client):
    res = client.get("/")
    assert res.status_code == 200


@pytest.mark.parametrize(
    ["post_data", "expected_output"],
    [
        pytest.param(
            {"smiles": ["COc1ccc(-c2cc(-c3ccc(C(=O)N(C)C)cc3)cnc2N)cn1"]},
            0,
            id="post_data_case_1",
        ),
        pytest.param(
            {"smiles": ["O=C(c1ccncc1)N1CCC2(CCN(Cc3cccc(Oc4ccccc4)c3)CC2)CC1"]},
            1,
            id="post_data_case_2",
        ),
    ],
)
def test_prediction(post_data, expected_output, client):
    response = client.post(
        "/predict",
        data=json.dumps(post_data),
        headers={"Content-Type": "application/json"},
    )

    assert json.loads(response.data)["predictions"][0] == expected_output
