def test_model_imports():
    from ml_project.model import TinyVLA, TinyVLAConfig
    from ml_project.cross_attention_model import FiLMTinyVLA
    from ml_project.data import FrozenVLADataset
    assert TinyVLA and TinyVLAConfig and FiLMTinyVLA and FrozenVLADataset
