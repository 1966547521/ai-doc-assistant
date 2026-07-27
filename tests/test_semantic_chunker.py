from src.semantic_chunker import SemanticChunker


class FixedEncoder:
    def encode(self, sentences, normalize_embeddings=True):
        vectors = {
            "苹果是一种水果。": [1.0, 0.0],
            "苹果富含维生素。": [0.9, 0.1],
            "服务器需要重启。": [0.0, 1.0],
        }
        return [vectors[sentence] for sentence in sentences]


def test_semantic_chunker_splits_on_topic_change():
    chunker = SemanticChunker(encoder=FixedEncoder(), similarity_threshold=0.5)

    chunks = chunker.split("苹果是一种水果。苹果富含维生素。服务器需要重启。")

    assert chunks == ["苹果是一种水果。苹果富含维生素。", "服务器需要重启。"]


def test_semantic_chunker_defaults_to_multilingual_model_for_chinese_documents():
    assert SemanticChunker().model_name == "minishlab/potion-multilingual-128M"


def test_semantic_chunker_defaults_to_project_local_model_directory():
    assert SemanticChunker().model_path.name == "potion-multilingual-128M"
    assert SemanticChunker().model_path.parent.name == "models"
