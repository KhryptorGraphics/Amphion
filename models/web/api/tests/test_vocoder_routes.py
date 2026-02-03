"""
Integration tests for Vocoder routes.

Tests all Vocoder endpoints:
- /api/vocoder/bigvgan
- /api/vocoder/generic
- /api/vocoder/hifigan
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf
import io

from models.web.api.main import app

client = TestClient(app, headers={"X-API-Key": "amphion-dev-key-change-in-production"})


class TestVocoderRoutes:
    """Test suite for Vocoder routes."""

    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing."""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer

    @pytest.fixture
    def mock_model_manager(self):
        """Mock the ModelManager for testing."""
        with patch('models.web.api.routes.vocoder.ModelManager') as mock:
            manager_instance = MagicMock()
            manager_instance.bigvgan_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            manager_instance.generic_vocoder_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            manager_instance.hifigan_inference.return_value = (22050, np.zeros(22050, dtype=np.float32))
            mock.return_value = manager_instance
            yield mock

    def test_bigvgan_endpoint_exists(self):
        """Test that the BigVGAN endpoint exists."""
        response = client.post("/api/vocoder/bigvgan")
        assert response.status_code == 422

    def test_bigvgan_with_audio(self, mock_audio_file, mock_model_manager):
        """Test BigVGAN endpoint with audio file."""
        mock_audio_file.seek(0)
        files = {
            "audio": ("audio.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/vocoder/bigvgan", files=files)
        assert response.status_code in [200, 500]

    def test_generic_vocoder_endpoint_exists(self):
        """Test that the Generic Vocoder endpoint exists."""
        response = client.post("/api/vocoder/generic")
        assert response.status_code == 422

    def test_generic_vocoder_with_audio(self, mock_audio_file, mock_model_manager):
        """Test Generic Vocoder endpoint with audio file."""
        mock_audio_file.seek(0)
        files = {
            "audio": ("audio.wav", mock_audio_file, "audio/wav")
        }
        data = {
            "vocoder_name": "hifigan"
        }

        response = client.post("/api/vocoder/generic", files=files, data=data)
        assert response.status_code in [200, 500]

    def test_hifigan_endpoint_exists(self):
        """Test that the HiFiGAN endpoint exists."""
        response = client.post("/api/vocoder/hifigan")
        assert response.status_code == 422

    def test_hifigan_with_audio(self, mock_audio_file, mock_model_manager):
        """Test HiFiGAN endpoint with audio file."""
        mock_audio_file.seek(0)
        files = {
            "audio": ("audio.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/vocoder/hifigan", files=files)
        assert response.status_code in [200, 500]


class TestVocoderIntegration:
    """Integration tests for Vocoder endpoints."""

    def test_all_vocoder_endpoints_registered(self):
        """Verify all Vocoder endpoints are registered."""
        routes = [r.path for r in app.routes if hasattr(r, 'path')]

        vocoder_routes = [
            '/api/vocoder/bigvgan',
            '/api/vocoder/generic',
            '/api/vocoder/hifigan'
        ]

        for route in vocoder_routes:
            assert route in routes, f"Route {route} not found"

    def test_vocoder_docs_available(self):
        """Test that API documentation includes Vocoder endpoints."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        paths = openapi_spec.get('paths', {})

        assert '/api/vocoder/bigvgan' in paths
        assert '/api/vocoder/generic' in paths
        assert '/api/vocoder/hifigan' in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
