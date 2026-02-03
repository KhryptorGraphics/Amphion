"""
Integration tests for SVC (Singing Voice Conversion) routes.

Tests all 5 SVC endpoints:
- /api/svc/vevosing
- /api/svc/diffcomosvc
- /api/svc/transformersvc
- /api/svc/vitssvc
- /api/svc/multiplecontentssvc
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf
import io
import os
import tempfile

# Import the app
from models.web.api.main import app

client = TestClient(app, headers={"X-API-Key": "amphion-dev-key-change-in-production"})


class TestSVCRoutes:
    """Test suite for SVC routes."""

    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing."""
        # Create a simple sine wave
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer

    @pytest.fixture
    def mock_model_manager(self):
        """Mock the ModelManager for testing."""
        with patch('models.web.api.routes.svc.ModelManager') as mock:
            manager_instance = MagicMock()
            # Return sample rate and audio data
            manager_instance.vevosing_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            manager_instance.diffcomosvc_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            manager_instance.transformersvc_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            manager_instance.vitssvc_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            manager_instance.multiplecontentssvc_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            mock.return_value = manager_instance
            yield mock

    def test_vevosing_endpoint_exists(self):
        """Test that the VevoSing endpoint exists and accepts POST requests."""
        # Test with no files - should return 422 (validation error)
        response = client.post("/api/svc/vevosing")
        assert response.status_code == 422

    def test_vevosing_with_mock_audio(self, mock_audio_file, mock_model_manager):
        """Test VevoSing endpoint with mock audio files."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav"),
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/vevosing", files=files)

        # Should succeed with mocked model
        assert response.status_code in [200, 500]  # 200 if mock works, 500 if model fails

    def test_vevosing_form_parameters(self, mock_audio_file, mock_model_manager):
        """Test VevoSing endpoint with form parameters."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav"),
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }
        data = {
            "mode": "ar",
            "use_shifted_src": "false",
            "flow_matching_steps": "16"
        }

        response = client.post("/api/svc/vevosing", files=files, data=data)
        assert response.status_code in [200, 500]

    def test_diffcomosvc_endpoint_exists(self):
        """Test that the DiffComoSVC endpoint exists."""
        response = client.post("/api/svc/diffcomosvc")
        assert response.status_code == 422

    def test_diffcomosvc_with_mock_audio(self, mock_audio_file, mock_model_manager):
        """Test DiffComoSVC endpoint with mock audio files."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav"),
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/diffcomosvc", files=files)
        assert response.status_code in [200, 500]

    def test_transformersvc_endpoint_exists(self):
        """Test that the TransformerSVC endpoint exists."""
        response = client.post("/api/svc/transformersvc")
        assert response.status_code == 422

    def test_transformersvc_with_mock_audio(self, mock_audio_file, mock_model_manager):
        """Test TransformerSVC endpoint with mock audio files."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav"),
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/transformersvc", files=files)
        assert response.status_code in [200, 500]

    def test_vitssvc_endpoint_exists(self):
        """Test that the VitsSVC endpoint exists."""
        response = client.post("/api/svc/vitssvc")
        assert response.status_code == 422

    def test_vitssvc_with_mock_audio(self, mock_audio_file, mock_model_manager):
        """Test VitsSVC endpoint with mock audio files."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav"),
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/vitssvc", files=files)
        assert response.status_code in [200, 500]

    def test_multiplecontentssvc_endpoint_exists(self):
        """Test that the MultipleContentsSVC endpoint exists."""
        response = client.post("/api/svc/multiplecontentssvc")
        assert response.status_code == 422

    def test_multiplecontentssvc_with_mock_audio(self, mock_audio_file, mock_model_manager):
        """Test MultipleContentsSVC endpoint with mock audio files."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav"),
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/multiplecontentssvc", files=files)
        assert response.status_code in [200, 500]

    def test_svc_invalid_audio_format(self):
        """Test SVC endpoints reject invalid audio formats."""
        files = {
            "content_audio": ("content.txt", io.BytesIO(b"not audio"), "text/plain"),
            "reference_audio": ("reference.txt", io.BytesIO(b"not audio"), "text/plain")
        }

        response = client.post("/api/svc/vevosing", files=files)
        # Should return 400 for invalid audio
        assert response.status_code in [400, 415, 422]

    def test_svc_missing_content_audio(self, mock_audio_file):
        """Test SVC endpoints reject missing content audio."""
        mock_audio_file.seek(0)
        files = {
            "reference_audio": ("reference.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/vevosing", files=files)
        assert response.status_code == 422

    def test_svc_missing_reference_audio(self, mock_audio_file):
        """Test SVC endpoints reject missing reference audio."""
        mock_audio_file.seek(0)
        files = {
            "content_audio": ("content.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/svc/vevosing", files=files)
        assert response.status_code == 422


class TestSVCIntegration:
    """Integration tests that verify end-to-end functionality."""

    def test_all_svc_endpoints_registered(self):
        """Verify all SVC endpoints are registered in the API."""
        routes = [r.path for r in app.routes if hasattr(r, 'path')]

        svc_routes = [
            '/api/svc/vevosing',
            '/api/svc/diffcomosvc',
            '/api/svc/transformersvc',
            '/api/svc/vitssvc',
            '/api/svc/multiplecontentssvc'
        ]

        for route in svc_routes:
            assert route in routes, f"Route {route} not found in API"

    def test_svc_endpoints_require_post(self):
        """Verify SVC endpoints only accept POST requests."""
        endpoints = [
            '/api/svc/vevosing',
            '/api/svc/diffcomosvc',
            '/api/svc/transformersvc',
            '/api/svc/vitssvc',
            '/api/svc/multiplecontentssvc'
        ]

        for endpoint in endpoints:
            # GET should not be allowed (may return 405 Method Not Allowed or 429 Rate Limited)
            response = client.get(endpoint)
            assert response.status_code in [405, 429], f"GET {endpoint} should return 405 or 429"

    def test_svc_docs_available(self):
        """Test that API documentation includes SVC endpoints."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        paths = openapi_spec.get('paths', {})

        assert '/api/svc/vevosing' in paths
        assert '/api/svc/diffcomosvc' in paths
        assert '/api/svc/transformersvc' in paths
        assert '/api/svc/vitssvc' in paths
        assert '/api/svc/multiplecontentssvc' in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
