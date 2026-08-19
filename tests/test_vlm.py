from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from cycles.core.types import ClassificationResult, EstrousStage
from cycles.stages.vlm import VLMConfig, VLMInterpretationService


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample_cytology.jpg"
    img = Image.new("RGB", (200, 200), color=(180, 180, 190))
    img.save(img_path)
    return img_path


def test_vlm_encode_image(sample_image: Path) -> None:
    service = VLMInterpretationService()
    encoded = service._encode_image(sample_image)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_vlm_fallback_interpretation(sample_image: Path) -> None:
    service = VLMInterpretationService(VLMConfig(endpoint_url="http://invalid.local/v1"))
    result = service.interpret_image(sample_image)
    assert isinstance(result, ClassificationResult)
    assert result.predicted_stage in EstrousStage.canonical_stages()
    assert 0.0 <= result.confidence <= 1.0


def test_vlm_mocked_api_response(sample_image: Path) -> None:
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "predicted_stage": "diestrus",
                            "confidence": 0.94,
                            "probabilities": {
                                "diestrus": 0.94,
                                "proestrus": 0.02,
                                "estrus": 0.01,
                                "metestrus": 0.03,
                            },
                            "cellular_breakdown": {
                                "leukocyte_pct": 92.0,
                                "nucleated_epithelial_pct": 5.0,
                                "cornified_squamous_pct": 3.0,
                            },
                            "visual_rationale": "Smear dominated by dense polymorphonuclear leukocytes.",
                        }
                    )
                }
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    service = VLMInterpretationService(VLMConfig(api_key="test-key"))
    with patch("urllib.request.urlopen", return_value=mock_response):
        result = service.interpret_image(sample_image)

    assert result.predicted_stage == EstrousStage.DIESTRUS
    assert result.confidence == pytest.approx(0.94)
    assert result.probabilities[EstrousStage.DIESTRUS] > 0.90
