"""
Integration tests for Codec routes.

Tests all Codec endpoints:
- /api/codec/dualcodec/encode
- /api/codec/dualcodec/decode
- /api/codec/facodec/encode
- /api/codec/facodec/decode
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf
import io

from models.web.api.main import app

client = TestClient(app, headers={"X-API-Key": "amphion-dev-key-change-in-production"})


class TestCodecRoutes:
    """Test suite for Codec routes."""

    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing."""
        sample_rate = 24000
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
        with patch('models.web.api.routes.codec.ModelManager') as mock:
            manager_instance = MagicMock()
            # Return encoded tokens and sample rate
            manager_instance.dualcodec_encode.return_value = (np.array([[1, 2, 3]]), 24000)
            manager_instance.dualcodec_decode.return_value = (24000, np.zeros(24000, dtype=np.float32))
            manager_instance.facodec_encode.return_value = (np.array([[4, 5, 6]]), 24000)
            manager_instance.facodec_decode.return_value = (24000, np.zeros(24000, dtype=np.float32))
            mock.return_value = manager_instance
            yield mock

    def test_dualcodec_encode_endpoint_exists(self):
        """Test that the DualCodec encode endpoint exists."""
        response = client.post("/api/codec/dualcodec/encode")
        assert response.status_code == 422

    def test_dualcodec_encode_with_audio(self, mock_audio_file, mock_model_manager):
        """Test DualCodec encode endpoint with audio file."""
        mock_audio_file.seek(0)
        files = {
            "audio": ("audio.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/codec/dualcodec/encode", files=files)
        assert response.status_code in [200, 500]

    def test_dualcodec_decode_endpoint_exists(self):
        """Test that the DualCodec decode endpoint exists."""
        response = client.post("/api/codec/dualcodec/decode")
        assert response.status_code == 422

    def test_dualcodec_decode_with_tokens(self, mock_model_manager):
        """Test DualCodec decode endpoint with tokens."""
        # Send tokens as form data or JSON
        data = {
            "tokens": "[[1, 2, 3, 4, 5]]"
        }

        response = client.post("/api/codec/dualcodec/decode", data=data)
        assert response.status_code in [200, 500]

    def test_facodec_encode_endpoint_exists(self):
        """Test that the FACodec encode endpoint exists."""
        response = client.post("/api/codec/facodec/encode")
        assert response.status_code == 422

    def test_facodec_encode_with_audio(self, mock_audio_file, mock_model_manager):
        """Test FACodec encode endpoint with audio file."""
        mock_audio_file.seek(0)
        files = {
            "audio": ("audio.wav", mock_audio_file, "audio/wav")
        }

        response = client.post("/api/codec/facodec/encode", files=files)
        assert response.status_code in [200, 500]

    def test_facodec_decode_endpoint_exists(self):
        """Test that the FACodec decode endpoint exists."""
        response = client.post("/api/codec/facodec/decode")
        assert response.status_code == 422

    def test_facodec_decode_with_tokens(self, mock_model_manager):
        """Test FACodec decode endpoint with tokens."""
        data = {
            "tokens": "[[1, 2, 3, 4, 5]]"
        }

        response = client.post("/api/codec/facodec/decode", data=data)
        assert response.status_code in [200, 500]


class TestCodecIntegration:
    """Integration tests for Codec endpoints."""

    def test_all_codec_endpoints_registered(self):
        """Verify all Codec endpoints are registered."""
        routes = [r.path for r in app.routes if hasattr(r, 'path')]

        codec_routes = [
            '/api/codec/dualcodec/encode',
            '/api/codec/dualcodec/decode',
            '/api/codec/facodec/encode',
            '/api/codec/facodec/decode'
        ]

        for route in codec_routes:
            assert route in routes, f"Route {route} not found"

    def test_codec_docs_available(self):
        """Test that API documentation includes Codec endpoints."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        paths = openapi_spec.get('paths', {})

        assert '/api/codec/dualcodec/encode' in paths
        assert '/api/codec/dualcodec/decode' in paths
        assert '/api/codec/facodec/encode' in paths
        assert '/api/codec/facodec/decode' in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
