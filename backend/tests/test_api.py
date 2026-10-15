"""Integration tests for FastAPI endpoints: Health, Authentication, and History."""

import os
from pathlib import Path
import tempfile
import unittest

# Configure temporary test database before importing app
temp_dir = tempfile.mkdtemp()
os.environ["DATA_DIR"] = temp_dir

# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db

client = TestClient(app)


class TestNutriVisionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_root_endpoint(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("NutriVision AI API", data["service"])

    def test_health_endpoint(self):
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("MultiTaskFoodCNN", data["modelArchitecture"])

    def test_auth_registration_and_login_flow(self):
        test_email = "test.user@example.com"
        test_password = "securePassword123"

        # 1. Register new user
        reg_payload = {
            "email": test_email,
            "password": test_password,
            "name": "Integration Tester",
            "dailyCalorieGoal": 2400,
            "dailyProteinGoal": 150,
        }
        reg_res = client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(reg_res.status_code, 200)
        reg_data = reg_res.json()
        self.assertTrue(reg_data["success"])
        self.assertEqual(reg_data["user"]["email"], test_email)
        self.assertEqual(reg_data["user"]["name"], "Integration Tester")
        self.assertIn("token", reg_data)
        token = reg_data["token"]

        # 2. Reject duplicate registration
        dup_res = client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(dup_res.status_code, 400)
        self.assertIn("already exists", dup_res.json()["detail"])

        # 3. Reject invalid password
        bad_login = client.post("/api/auth/login", json={"email": test_email, "password": "wrongPassword"})
        self.assertEqual(bad_login.status_code, 401)

        # 4. Successful login
        good_login = client.post("/api/auth/login", json={"email": test_email, "password": test_password})
        self.assertEqual(good_login.status_code, 200)
        login_data = good_login.json()
        self.assertTrue(login_data["success"])
        self.assertIn("token", login_data)

        # 5. Get current profile using Bearer token
        headers = {"Authorization": f"Bearer {token}"}
        me_res = client.get("/api/auth/me", headers=headers)
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["email"], test_email)

        # 6. Update profile goals
        patch_res = client.patch(
            "/api/auth/profile",
            headers=headers,
            json={"name": "Updated Tester", "dailyCalorieGoal": 2500},
        )
        self.assertEqual(patch_res.status_code, 200)
        self.assertEqual(patch_res.json()["name"], "Updated Tester")
        self.assertEqual(patch_res.json()["dailyCalorieGoal"], 2500)

    def test_history_crud_flow(self):
        item_payload = {
            "id": "hist_test_scan_01",
            "foodClass": "Grilled Chicken Rice Bowl",
            "calories": 650.0,
            "protein": 38.0,
            "carbohydrates": 72.0,
            "fat": 19.0,
            "fiber": 5.2,
            "confidenceScore": 0.94,
            "confidencePercentage": 94.0,
            "servingSize": "1 bowl",
            "isLowConfidence": False,
        }

        # 1. Save scan to history
        post_res = client.post("/api/history", json=item_payload)
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(post_res.json()["status"], "saved")

        # 2. Retrieve history list
        get_res = client.get("/api/history?limit=10")
        self.assertEqual(get_res.status_code, 200)
        history_data = get_res.json()
        self.assertGreaterEqual(history_data["total"], 1)
        found = any(item["id"] == "hist_test_scan_01" for item in history_data["history"])
        self.assertTrue(found)

        # 3. Delete scan from history
        del_res = client.delete("/api/history/hist_test_scan_01")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "deleted")

        # 4. Verify removal
        get_after = client.get("/api/history?limit=10")
        self.assertFalse(any(item["id"] == "hist_test_scan_01" for item in get_after.json()["history"]))

    def test_predict_endpoint_validation(self):
        # 1. When model weights are not loaded, verify 503 Service Unavailable response
        response = client.post(
            "/api/predict",
            files={"image": ("test.jpg", b"image bytes", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("trained inference model is not loaded", response.json()["detail"])

        # 2. When predictor is mock-ready, verify unsupported file type rejection (400)
        from backend.app.predictor import predictor_instance
        orig_ready = predictor_instance.is_torch_ready
        try:
            predictor_instance.is_torch_ready = True
            bad_format_res = client.post(
                "/api/predict",
                files={"image": ("test.txt", b"not an image", "text/plain")},
            )
            self.assertEqual(bad_format_res.status_code, 400)
            self.assertIn("Unsupported image type", bad_format_res.json()["detail"])
        finally:
            predictor_instance.is_torch_ready = orig_ready


if __name__ == "__main__":
    unittest.main()

