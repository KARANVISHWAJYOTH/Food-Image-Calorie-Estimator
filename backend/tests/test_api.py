"""Integration tests for FastAPI endpoints: Health, Authentication, SQLAlchemy 2.0 History, and Image Uploads."""

import os
from pathlib import Path
import tempfile
import unittest
import io
from PIL import Image

# Configure temporary test database and uploads before importing app
temp_dir = tempfile.mkdtemp()
os.environ["DATA_DIR"] = temp_dir
os.environ["UPLOADS_DIR"] = os.path.join(temp_dir, "uploads")
os.environ["SECRET_KEY"] = "test_jwt_secret_key_30min_test"

# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db, UPLOADS_DIR

client = TestClient(app)


def _create_sample_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


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

        # 1. Register new user via /api/auth/signup
        signup_payload = {
            "email": test_email,
            "username": "tester_user",
            "password": test_password,
            "name": "Integration Tester",
            "dailyCalorieGoal": 2400,
            "dailyProteinGoal": 150,
        }
        reg_res = client.post("/api/auth/signup", json=signup_payload)
        self.assertEqual(reg_res.status_code, 200)
        reg_data = reg_res.json()
        self.assertTrue(reg_data["success"])
        self.assertEqual(reg_data["user"]["email"], test_email)
        self.assertEqual(reg_data["user"]["name"], "Integration Tester")
        self.assertIn("token", reg_data)
        token = reg_data["token"]

        # 2. Reject duplicate registration via /api/auth/register alias
        dup_res = client.post("/api/auth/register", json=signup_payload)
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

    def test_demo_account_seeded(self):
        """Verify seeded demo user works with 1-click Demo credentials."""
        res = client.post(
            "/api/auth/login",
            json={"email": "demo@nutrivision.ai", "password": "demoPassword123"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["email"], "demo@nutrivision.ai")
        self.assertIn("token", data)

    def test_unauthenticated_requests_rejected(self):
        """Ensure /api/predict and /api/history reject unauthenticated requests with 401."""
        pred_res = client.post(
            "/api/predict",
            files={"image": ("test.jpg", b"image bytes", "image/jpeg")},
        )
        self.assertEqual(pred_res.status_code, 401)
        self.assertIn("Authentication token is missing", pred_res.json()["detail"])

        get_res = client.get("/api/history")
        self.assertEqual(get_res.status_code, 401)

        post_res = client.post("/api/history", json={"foodClass": "Pizza", "calories": 300})
        self.assertEqual(post_res.status_code, 401)

        del_res = client.delete("/api/history/item123")
        self.assertEqual(del_res.status_code, 401)

    def test_image_saving_to_volume_and_static_serving(self):
        """Verify that uploaded images in /api/predict are saved with UUID to volume and served via /uploads."""
        login_res = client.post(
            "/api/auth/login",
            json={"email": "demo@nutrivision.ai", "password": "demoPassword123"},
        )
        token = login_res.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        img_bytes = _create_sample_jpeg_bytes()

        from backend.app.predictor import predictor_instance
        orig_ready = predictor_instance.is_torch_ready
        orig_valid = predictor_instance.is_model_valid
        orig_predict = predictor_instance.predict_image

        try:
            predictor_instance.is_torch_ready = True
            predictor_instance.is_model_valid = True
            from backend.app.schemas import PredictionResponse
            predictor_instance.predict_image = lambda contents, filename: PredictionResponse(
                id="pred_mock_uuid",
                foodClass="Sample Margherita Pizza",
                confidenceScore=0.96,
                confidencePercentage=96.0,
                estimatedCalories=580.0,
                protein=24.0,
                carbohydrates=68.0,
                fat=18.0,
                fiber=4.0,
                healthTips="Rich in protein and carbs.",
            )

            res = client.post(
                "/api/predict",
                headers=headers,
                files={"image": ("my_meal.jpg", img_bytes, "image/jpeg")},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("imagePath", data)
            self.assertIn("imageUrl", data)
            self.assertTrue(data["imageUrl"].startswith("/uploads/"))
            self.assertTrue(os.path.exists(data["imagePath"]))

            # Verify image serving via mounted static route
            serve_res = client.get(data["imageUrl"])
            self.assertEqual(serve_res.status_code, 200)
            self.assertEqual(serve_res.content, img_bytes)
        finally:
            predictor_instance.is_torch_ready = orig_ready
            predictor_instance.is_model_valid = orig_valid
            predictor_instance.predict_image = orig_predict

    def test_predict_rejects_unvalidated_checkpoint(self):
        """Verify /api/predict returns 503 when the loaded checkpoint fails validation."""
        login_res = client.post(
            "/api/auth/login",
            json={"email": "demo@nutrivision.ai", "password": "demoPassword123"},
        )
        token = login_res.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        from backend.app.predictor import predictor_instance
        orig_ready = predictor_instance.is_torch_ready
        orig_valid = predictor_instance.is_model_valid
        orig_validation = predictor_instance.model_validation

        try:
            predictor_instance.is_torch_ready = True
            predictor_instance.is_model_valid = False
            predictor_instance.model_validation = {
                "valid": False,
                "errors": ["untrained_batch_norm_statistics"],
                "warnings": [],
            }
            res = client.post(
                "/api/predict",
                headers=headers,
                files={"image": ("meal.jpg", _create_sample_jpeg_bytes(), "image/jpeg")},
            )
            self.assertEqual(res.status_code, 503)
            self.assertIn("untrained_batch_norm_statistics", res.json()["detail"])
        finally:
            predictor_instance.is_torch_ready = orig_ready
            predictor_instance.is_model_valid = orig_valid
            predictor_instance.model_validation = orig_validation

    def test_invalid_upload_rejected_without_orphan_file(self):
        """Malformed image bytes must return 400 and must not create a file in the uploads volume."""
        login_res = client.post(
            "/api/auth/login",
            json={"email": "demo@nutrivision.ai", "password": "demoPassword123"},
        )
        headers = {"Authorization": f"Bearer {login_res.json()['token']}"}

        from backend.app.predictor import predictor_instance
        orig_ready = predictor_instance.is_torch_ready
        orig_valid = predictor_instance.is_model_valid

        before = set(os.listdir(UPLOADS_DIR)) if os.path.isdir(UPLOADS_DIR) else set()
        try:
            predictor_instance.is_torch_ready = True
            predictor_instance.is_model_valid = True
            res = client.post(
                "/api/predict",
                headers=headers,
                files={"image": ("not-an-image.jpg", b"this is definitely not a jpeg", "image/jpeg")},
            )
            self.assertEqual(res.status_code, 400)
            self.assertIn("not a readable image", res.json()["detail"])
        finally:
            predictor_instance.is_torch_ready = orig_ready
            predictor_instance.is_model_valid = orig_valid

        after = set(os.listdir(UPLOADS_DIR)) if os.path.isdir(UPLOADS_DIR) else set()
        self.assertEqual(before, after)

    def test_password_change_flow(self):
        """Changing a password requires the current password and the new one to work afterwards."""
        email = "pw.change@example.com"
        reg = client.post(
            "/api/auth/signup",
            json={"email": email, "password": "originalPass123", "name": "Pw Tester"},
        )
        self.assertEqual(reg.status_code, 200)
        headers = {"Authorization": f"Bearer {reg.json()['token']}"}

        wrong_current = client.post(
            "/api/auth/change-password",
            headers=headers,
            json={"currentPassword": "notMyPassword", "newPassword": "replacementPass456"},
        )
        self.assertEqual(wrong_current.status_code, 400)

        short_new = client.post(
            "/api/auth/change-password",
            headers=headers,
            json={"currentPassword": "originalPass123", "newPassword": "short"},
        )
        self.assertEqual(short_new.status_code, 422)

        changed = client.post(
            "/api/auth/change-password",
            headers=headers,
            json={"currentPassword": "originalPass123", "newPassword": "replacementPass456"},
        )
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()["status"], "password_changed")
        self.assertIn("token", changed.json())

        old_login = client.post("/api/auth/login", json={"email": email, "password": "originalPass123"})
        self.assertEqual(old_login.status_code, 401)
        new_login = client.post("/api/auth/login", json={"email": email, "password": "replacementPass456"})
        self.assertEqual(new_login.status_code, 200)

    def test_signup_validation_rejects_bad_payloads(self):
        """Invalid emails and too-short passwords are rejected before touching the database."""
        bad_email = client.post(
            "/api/auth/signup",
            json={"email": "not-an-email", "password": "validPassword123"},
        )
        self.assertEqual(bad_email.status_code, 422)

        short_password = client.post(
            "/api/auth/signup",
            json={"email": "short.pw@example.com", "password": "short"},
        )
        self.assertEqual(short_password.status_code, 422)

    def test_logout_endpoint(self):
        res = client.post("/api/auth/logout")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "logged_out")

    def test_user_history_isolation_and_owner_only_delete(self):
        """Verify /api/history returns ONLY current user's records, supports limit/offset, and allows owner-only DELETE."""
        # 1. Register User A
        email_a = "user.a@example.com"
        reg_a = client.post(
            "/api/auth/signup",
            json={"email": email_a, "password": "passUserA123", "name": "User A"},
        )
        token_a = reg_a.json()["token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # 2. Register User B
        email_b = "user.b@example.com"
        reg_b = client.post(
            "/api/auth/signup",
            json={"email": email_b, "password": "passUserB123", "name": "User B"},
        )
        token_b = reg_b.json()["token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 3. User A creates 3 prediction history items (server assigns the ids)
        created_ids = []
        for i in range(1, 4):
            item_payload = {
                "foodClass": f"Dish A {i}",
                "calories": 400.0 + i * 50,
                "protein": 20.0 + i,
                "carbohydrates": 50.0 + i,
                "fat": 15.0,
                "fiber": 3.0,
                "confidenceScore": 0.9,
            }
            save_res = client.post("/api/history", json=item_payload, headers=headers_a)
            self.assertEqual(save_res.status_code, 200)
            created_ids.append(save_res.json()["item"]["id"])
        self.assertEqual(len(set(created_ids)), 3)

        # 4. User B checks history -> should be completely empty (User A items isolated)
        b_hist_res = client.get("/api/history", headers=headers_b)
        self.assertEqual(b_hist_res.status_code, 200)
        b_data = b_hist_res.json()
        self.assertEqual(b_data["total"], 0)
        self.assertEqual(len(b_data["history"]), 0)

        # 5. User B tries to DELETE User A's item -> 404 forbidden/not found
        del_attempt = client.delete(f"/api/history/{created_ids[0]}", headers=headers_b)
        self.assertEqual(del_attempt.status_code, 404)

        # 6. User A checks paginated history with limit=2, offset=0
        page1_res = client.get("/api/history?limit=2&offset=0", headers=headers_a)
        self.assertEqual(page1_res.status_code, 200)
        page1 = page1_res.json()
        self.assertEqual(page1["total"], 3)
        self.assertEqual(len(page1["history"]), 2)
        self.assertEqual(page1["limit"], 2)
        self.assertEqual(page1["offset"], 0)

        # 7. User A checks second page with limit=2, offset=2
        page2_res = client.get("/api/history?limit=2&offset=2", headers=headers_a)
        self.assertEqual(page2_res.status_code, 200)
        page2 = page2_res.json()
        self.assertEqual(page2["total"], 3)
        self.assertEqual(len(page2["history"]), 1)
        self.assertEqual(page2["offset"], 2)

        # 8. User A deletes their own item -> succeeds (200)
        del_owner_res = client.delete(f"/api/history/{created_ids[0]}", headers=headers_a)
        self.assertEqual(del_owner_res.status_code, 200)
        self.assertEqual(del_owner_res.json()["status"], "deleted")

        # 9. Verify count decreased to 2
        after_del = client.get("/api/history", headers=headers_a)
        self.assertEqual(after_del.json()["total"], 2)
        self.assertFalse(any(item["id"] == created_ids[0] for item in after_del.json()["history"]))

        # 10. Clear-all removes only User A's remaining records
        clear_res = client.delete("/api/history", headers=headers_a)
        self.assertEqual(clear_res.status_code, 200)
        self.assertEqual(clear_res.json()["deleted"], 2)
        self.assertEqual(client.get("/api/history", headers=headers_a).json()["total"], 0)

    def test_duplicate_client_history_id_is_ignored(self):
        """Two saves that send the same client id must both succeed with distinct server ids."""
        reg = client.post(
            "/api/auth/signup",
            json={"email": "dup.history@example.com", "password": "dupPass123", "name": "Dup"},
        )
        headers = {"Authorization": f"Bearer {reg.json()['token']}"}
        payload = {"id": "hist_client_fixed_id", "foodClass": "Pizza", "calories": 300}

        first = client.post("/api/history", json=payload, headers=headers)
        second = client.post("/api/history", json=payload, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first.json()["item"]["id"], "hist_client_fixed_id")
        self.assertNotEqual(first.json()["item"]["id"], second.json()["item"]["id"])
        self.assertEqual(client.get("/api/history", headers=headers).json()["total"], 2)


if __name__ == "__main__":
    unittest.main()
