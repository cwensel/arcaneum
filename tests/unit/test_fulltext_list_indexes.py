"""FullTextClient.list_indexes returns every index, not MeiliSearch's first page."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from arcaneum.fulltext.client import FullTextClient


def paged_server(uids):
    """Fake meilisearch.Client.get_indexes honoring offset/limit like the server (default 20)."""

    def get_indexes(parameters=None):
        parameters = parameters or {}
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", 20)
        page = [SimpleNamespace(uid=uid) for uid in uids[offset : offset + limit]]
        return {"results": page, "offset": offset, "limit": limit, "total": len(uids)}

    return MagicMock(side_effect=get_indexes)


def client_with(uids):
    client = FullTextClient.__new__(FullTextClient)
    client.client = MagicMock()
    client.client.get_indexes = paged_server(uids)
    return client


def test_list_indexes_returns_every_page():
    uids = [f"Corpus{i:03d}" for i in range(257)]

    listed = [index["uid"] for index in client_with(uids).list_indexes()]

    assert listed == uids


def test_list_indexes_with_no_indexes():
    assert client_with([]).list_indexes() == []


def test_list_indexes_accepts_a_plain_list_response():
    client = FullTextClient.__new__(FullTextClient)
    client.client = MagicMock()
    client.client.get_indexes.return_value = [{"uid": "Notes"}]

    assert client.list_indexes() == [{"uid": "Notes"}]
