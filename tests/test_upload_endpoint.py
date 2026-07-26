"""
test_upload_endpoint.py

测试文档上传接口（异步改造后的版本）和任务状态查询接口。

核心测试思路：
- 不需要真的启动Celery worker、不需要真的连Redis
- 用monkeypatch替换掉 process_document_task.delay 这个方法，
  验证"接口有没有正确地把任务丢出去"，而不需要真的等任务跑完
- SHARED_UPLOAD_DIR通过环境变量指向pytest提供的临时目录(tmp_path)，
  这样测试不会往真实的容器路径 /app/shared_uploads 写文件
"""

import io
from unittest.mock import MagicMock

import main


def test_upload_rejects_unsupported_format(client):
    """.md等不支持的格式应该被拒绝，返回400，且不应该触发任何后台任务"""
    resp = client.post(
        "/documents/upload",
        files={"file": ("note.md", io.BytesIO(b"some content"), "text/markdown")},
    )

    assert resp.status_code == 400
    assert "只支持" in resp.json()["detail"]


def test_upload_accepts_txt_and_dispatches_celery_task(client, tmp_path, monkeypatch):
    """核心验证：上传一个支持的格式，应该立刻返回task_id，而不是等处理完成"""
    monkeypatch.setenv("SHARED_UPLOAD_DIR", str(tmp_path))

    fake_task_result = MagicMock()
    fake_task_result.id = "fake-task-id-123"
    monkeypatch.setattr(
        main.process_document_task, "delay", MagicMock(return_value=fake_task_result)
    )

    resp = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO("测试内容".encode("utf-8")), "text/plain")},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "fake-task-id-123"
    assert data["filename"] == "test.txt"
    assert "后台处理" in data["message"]
    # 验证.delay确实被调用了一次，也就是任务确实被丢进了队列
    main.process_document_task.delay.assert_called_once()


def test_upload_saves_file_to_shared_dir(client, tmp_path, monkeypatch):
    """验证文件真的被写到了共享目录里——worker容器要能读到这份文件才能真正处理它"""
    monkeypatch.setenv("SHARED_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        main.process_document_task,
        "delay",
        MagicMock(return_value=MagicMock(id="xyz")),
    )

    client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO("内容".encode("utf-8")), "text/plain")},
    )

    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].suffix == ".txt"


def test_status_endpoint_returns_success_state(client, monkeypatch):
    """验证 GET /documents/status/{task_id} 能正确转述Celery任务的完成状态和结果"""
    fake_async_result = MagicMock()
    fake_async_result.state = "SUCCESS"
    fake_async_result.result = {"status": "done", "chunks_count": 5, "doc_id": "abc123"}
    monkeypatch.setattr(
        main.celery_app, "AsyncResult", MagicMock(return_value=fake_async_result)
    )

    resp = client.get("/documents/status/some-task-id")

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "some-task-id"
    assert data["state"] == "SUCCESS"
    assert data["result"]["chunks_count"] == 5


def test_status_endpoint_returns_failure_with_error_message(client, monkeypatch):
    """任务失败时，应该把错误信息带出来，而不是只说"失败了"却不知道为什么"""
    fake_async_result = MagicMock()
    fake_async_result.state = "FAILURE"
    fake_async_result.info = ValueError("解析文档时出错")
    monkeypatch.setattr(
        main.celery_app, "AsyncResult", MagicMock(return_value=fake_async_result)
    )

    resp = client.get("/documents/status/some-task-id")

    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "FAILURE"
    assert "解析文档时出错" in data["error"]
