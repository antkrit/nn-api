from api.main import API_V1_PATH


def testr_root_redirect(client):
    response = client.get("/")
    assert response.status_code == 200
    assert str(response.url) == "http://testserver" + API_V1_PATH + "/docs"
