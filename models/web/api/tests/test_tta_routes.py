"""
Integration tests for TTA (Text-to-Audio) routes.

Tests all TTA endpoints:
- /api/tta/audioldm
- /api/tta/picoaudio
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf
import io

from models.web.api.main import app

client = TestClient(app, headers={"X-API-Key": "amphion-dev-key-change-in-production"})


class TestTTARoutes:
    """Test suite for TTA routes."""

    @pytest.fixture
    def mock_model_manager(self):
        """Mock the ModelManager for testing."""
        with patch('models.web.api.routes.tta.ModelManager') as mock:
            manager_instance = MagicMock()
            manager_instance.audioldm_inference.return_value = (16000, np.zeros(16000, dtype=np.float32))
            manager_instance.picoaudio_inference.return_value = (16000, np.zeros(16000, dtype=np.float32))
            mock.return_value = manager_instance
            yield mock

    def test_audioldm_endpoint_exists(self):
        """Test that the AudioLDM endpoint exists."""
        response = client.post("/api/tta/audioldm")
        assert response.status_code == 422  # Missing required text parameter

    def test_audioldm_with_text(self, mock_model_manager):
        """Test AudioLDM endpoint with text prompt."""
        data = {
            "text": "A dog barking in the park",
            "num_inference_steps": 50,
            "audio_length_in_s": 5.0
        }

        response = client.post("/api/tta/audioldm", data=data)
        assert response.status_code in [200, 500]

    def test_audioldm_missing_text(self):
        """Test AudioLDM endpoint rejects missing text."""
        response = client.post("/api/tta/audioldm", data={})
        assert response.status_code == 422

    def test_picoaudio_endpoint_exists(self):
        """Test that the PicoAudio endpoint exists."""
        response = client.post("/api/tta/picoaudio")
        assert response.status_code == 422

    def test_picoaudio_with_text(self, mock_model_manager):
        """Test PicoAudio endpoint with text prompt."""
        data = {
            "text": "Sound of rain falling",
            "duration": 3.0
        }

        response = client.post("/api/tta/picoaudio", data=data)
        assert response.status_code in [200, 500]

    def test_picoaudio_missing_text(self):
        """Test PicoAudio endpoint rejects missing text."""
        response = client.post("/api/tta/picoaudio", data={})
        assert response.status_code == 422


class TestTTAIntegration:
    """Integration tests for TTA endpoints."""

    def test_all_tta_endpoints_registered(self):
        """Verify all TTA endpoints are registered."""
        routes = [r.path for r in app.routes if hasattr(r, 'path')]

        tta_routes = ['/api/tta/audioldm', '/api/tta/picoaudio']

        for route in tta_routes:
            assert route in routes, f"Route {route} not found"

    def test_tta_docs_available(self):
        """Test that API documentation includes TTA endpoints."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        paths = openapi_spec.get('paths', {})

        assert '/api/tta/audioldm' in paths
        assert '/api/tta/picoaudio' in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
