from paperlens.data import parse_feed

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>  A   Test
      Title </title>
    <summary>Some   abstract
      text.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>"""


def test_parse_feed_extracts_fields():
    papers = parse_feed(SAMPLE)
    assert len(papers) == 1
    paper = papers[0]
    assert paper["arxiv_id"] == "2401.00001v1"
    assert paper["title"] == "A Test Title"
    assert paper["abstract"] == "Some abstract text."
    assert paper["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert paper["primary_category"] == "cs.LG"


def test_parse_feed_empty():
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_feed(empty) == []
