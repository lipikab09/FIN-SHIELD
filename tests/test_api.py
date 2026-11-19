from __future__ import annotations

from fastapi.testclient import TestClient

from finshield.serving.app import app


def test_health_and_scoring_endpoints():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["model_loaded"] is True

        payload = {
            "transaction_id": "TXN_TEST_001",
            "timestamp": 3650000.0,
            "amount": 320.0,
            "card_id": "CARD_00015",
            "device_id": "DEV_00005",
            "ip_subnet": "10.0.5.0/24",
            "merchant_id": "MERCH_0003",
            "email_domain": "proton.me",
            "product_code": "W",
            "card_network": "visa",
            "card_type": "credit",
            "addr_region": "REG_04",
            "dist_km": 1150.0,
            "is_international": 1,
        }
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200
        scored = resp.json()
        assert scored["transaction_id"] == "TXN_TEST_001"
        assert 0.0 <= scored["calibrated_fraud_probability"] <= 1.0
        assert scored["decision_action"] in {"APPROVE", "STEPUP_CHALLENGE", "BLOCK"}
        assert len(scored["reason_codes"]) > 0
