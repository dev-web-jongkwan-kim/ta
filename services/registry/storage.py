from __future__ import annotations

import io
import json
import pickle
from typing import Any, Dict

from minio import Minio

from packages.common.config import get_settings


def get_minio_client() -> Minio:
    settings = get_settings()
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
    return client


def upload_model(model: Any, object_name: str) -> str:
    client = get_minio_client()
    data = pickle.dumps(model)
    client.put_object(
        get_settings().minio_bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type="application/octet-stream",
    )
    return f"s3://{get_settings().minio_bucket}/{object_name}"


def upload_json(payload: Dict[str, Any], object_name: str) -> str:
    client = get_minio_client()
    data = json.dumps(payload).encode("utf-8")
    client.put_object(
        get_settings().minio_bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type="application/json",
    )
    return f"s3://{get_settings().minio_bucket}/{object_name}"


def download_model(object_name: str) -> Any:
    client = get_minio_client()
    resp = client.get_object(get_settings().minio_bucket, object_name)
    try:
        return pickle.loads(resp.read())
    finally:
        resp.close()
        resp.release_conn()
