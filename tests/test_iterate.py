from pathlib import Path
from multilabel_graphcut_annotation.app import CACHE_DIR, MultiLabelState, iterate, read_label_definitions


DATA_DIR = Path(__file__).parent / 'data'
def test_resume():
    labels = read_label_definitions(DATA_DIR / 'human.json')
    state = MultiLabelState.load(DATA_DIR, 'test')
    iterate(state, labels)
