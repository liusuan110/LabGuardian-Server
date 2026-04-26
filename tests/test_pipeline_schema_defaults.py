from app.schemas.pipeline import PipelineRequest


def test_pipeline_request_default_imgsz_matches_runtime_default() -> None:
    request = PipelineRequest(station_id="station-1", images_b64=["fake"])

    assert request.imgsz == 960

