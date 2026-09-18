import numpy as np
from starlette.testclient import TestClient


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates cosine similarity between two unit-normalized vectors via dot product."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    return float(np.dot(a, b))


def test_multilingual_semantic_similarity(client: TestClient, auth_headers: dict):
    """
    Semantic Acceptance Test:
    Generates embeddings for 4 passages and 5 queries across English, Hinglish, Hindi, and Marathi.
    Dynamically computes cosine similarities and asserts that:
    1. All 4 agricultural queries rank "Sub-Mission on Farm Mechanization" highest.
    2. The "domicile certificate" query ranks "Age, Nationality and Domicile Certificate" highest.
    """
    passages = [
        "Sub-Mission on Farm Mechanization",
        "Age, Nationality and Domicile Certificate",
        "Skill Development Training",
        "Income Certificate",
    ]

    # Generate passage embeddings using batch endpoint with input_type='passage'
    resp_passages = client.post(
        "/v1/embeddings/batch",
        json={"texts": passages, "input_type": "passage"},
        headers=auth_headers,
    )
    assert resp_passages.status_code == 200
    passage_embeddings = resp_passages.json()["embeddings"]
    assert len(passage_embeddings) == len(passages)

    farm_mech_idx = 0
    domicile_idx = 1

    agri_queries = [
        ("tractor subsidy", "English"),
        ("kisan ko tractor ke liye help", "Hinglish"),
        ("किसानों के लिए कृषि मशीन सहायता", "Hindi"),
        ("शेतकऱ्यांसाठी कृषी यंत्र मदत", "Marathi"),
    ]

    for query_text, lang in agri_queries:
        resp_q = client.post(
            "/v1/embeddings",
            json={"text": query_text, "input_type": "query"},
            headers=auth_headers,
        )
        assert resp_q.status_code == 200
        q_emb = resp_q.json()["embedding"]

        # Compute cosine similarity against all passages
        similarities = [
            (idx, passages[idx], cosine_similarity(q_emb, p_emb))
            for idx, p_emb in enumerate(passage_embeddings)
        ]
        similarities.sort(key=lambda x: x[2], reverse=True)

        top_idx, top_passage, top_score = similarities[0]

        # Agriculture queries must rank Sub-Mission on Farm Mechanization highest
        assert top_idx == farm_mech_idx, (
            f"Query '{query_text}' ({lang}) did not rank '{passages[farm_mech_idx]}' highest! "
            f"Rankings: {similarities}"
        )
        # Verify top score is strictly higher than runner-up score
        second_score = similarities[1][2]
        assert top_score > second_score, (
            f"Query '{query_text}' top score ({top_score:.4f}) is not higher than "
            f"runner-up score ({second_score:.4f})"
        )

    # Test the unrelated query
    domicile_query = "domicile certificate"
    resp_domicile = client.post(
        "/v1/embeddings",
        json={"text": domicile_query, "input_type": "query"},
        headers=auth_headers,
    )
    assert resp_domicile.status_code == 200
    dom_emb = resp_domicile.json()["embedding"]

    dom_similarities = [
        (idx, passages[idx], cosine_similarity(dom_emb, p_emb))
        for idx, p_emb in enumerate(passage_embeddings)
    ]
    dom_similarities.sort(key=lambda x: x[2], reverse=True)

    top_idx, top_passage, top_score = dom_similarities[0]
    assert top_idx == domicile_idx, (
        f"Query '{domicile_query}' did not rank '{passages[domicile_idx]}' highest! "
        f"Rankings: {dom_similarities}"
    )
