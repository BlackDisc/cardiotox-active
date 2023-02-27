import pytest
import pandas as pd
import sys
from typing import List

sys.path.append("src/")
from features import extract_features_from_smile


@pytest.mark.parametrize(
    ["smiles"],
    [
        pytest.param(
            ["Fc1ccc(-n2cc(NCCN3CCCCC3)nn2)cc1F"],
            id="feature_extraction_case_1",
        ),
        pytest.param(
            ["COc1cc(N2Cc3ccc(Sc4ccc(F)cc4)nc3C2=O)ccc1OCCN1CCCC1"],
            id="feature_extraction_case_2",
        ),
        pytest.param(
            [
                "CCOC(=O)[C@H]1CC[C@@H](N2CC(NC(=O)CNc3nn(C(N)=O)c4ccc(C(F)(F)F)cc34)C2)CC1"
            ],
            id="feature_extraction_case_3",
        ),
    ],
)
def test_feature_extraction(smiles: List[str]) -> None:
    features = extract_features_from_smile(
        smiles, feature_sets=["DESC"], desc_file_path="model/batch_model_features.txt"
    )
    desc_df = pd.read_csv("data/test_data/test_desc_df.csv", index_col=0)
    label_df = pd.read_csv("data/test_data/test_label_df.csv", index_col=0)

    with open("model/batch_model_features.txt", "r") as fp:
        selected_columns = [line.strip() for line in fp.readlines()]

    features_df = desc_df.join(label_df[["smiles"]])

    for idx, smile in enumerate(smiles):
        for col in selected_columns:
            assert "{:.4f}".format(
                float(features_df.query("smiles == @smile")[col])
            ) == "{:.4f}".format(float(features.iloc[idx, :][col]))
