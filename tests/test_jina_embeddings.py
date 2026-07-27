from unittest.mock import Mock, patch

from src.jina_embeddings import JinaEmbeddings


def test_jina_adapter_uses_passage_for_documents_and_query_for_questions():
    response = Mock()
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]
    }

    with patch("src.jina_embeddings.httpx.post", return_value=response) as post:
        embeddings = JinaEmbeddings(
            api_key="secret",
            base_url="https://api.jina.ai/v1",
            model="jina-embeddings-v5-text-small",
            timeout=20,
            max_retries=1,
        )
        assert embeddings.embed_documents(["a", "b"]) == [[1.0], [2.0]]

    assert post.call_args.kwargs["json"]["input"] == ["a", "b"]
    assert post.call_args.kwargs["json"]["task"] == "retrieval.passage"


def test_jina_adapter_uses_query_task_for_single_question():
    response = Mock()
    response.json.return_value = {"data": [{"index": 0, "embedding": [3.0]}]}

    with patch("src.jina_embeddings.httpx.post", return_value=response) as post:
        embeddings = JinaEmbeddings(
            api_key="secret",
            base_url="https://api.jina.ai/v1",
            model="jina-embeddings-v5-text-small",
            timeout=20,
            max_retries=1,
        )
        assert embeddings.embed_query("question") == [3.0]

    assert post.call_args.kwargs["json"]["task"] == "retrieval.query"
