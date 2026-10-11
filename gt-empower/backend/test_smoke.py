"""Smoke tests, no simulated embeddings or AI results."""
from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)
def test_health():
    r=client.get('/api/health')
    assert r.status_code==200 and r.json()['api']=='ok'
def test_people():
    r=client.get('/api/people')
    assert r.status_code==200 and isinstance(r.json(),list)
def test_meeting():
    created=client.post('/api/meetings',json={'date':'2026-10-09'})
    assert created.status_code==200
    mid=created.json()['id']
    assert client.get('/api/meetings/'+mid).status_code==200
