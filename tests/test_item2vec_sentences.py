from turnaware_recsys.features import Track
from turnaware_recsys.item2vec import RelationItem2Vec


def test_relation_sentences_connect_tracks_to_metadata():
    catalog = {
        "t1": Track("t1", "Blue Sky", "Artist A", "Album A", ("jazz",), 2020)
    }
    sentences = RelationItem2Vec().build_sentences(
        [["t1", "t2"]], catalog, {"t1": ["calm"]}
    )
    assert ["t1", "t2"] in sentences
    assert ["t1", "artist::artist a"] in sentences
    assert ["t1", "tag::jazz"] in sentences
    assert ["t1", "text::calm"] in sentences

