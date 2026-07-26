"""
test_tasks.py

测试Celery异步任务 process_document_task 的核心逻辑本身
（解析文档 -> 切块 -> embedding -> 入库 -> 清理临时文件）。

关键技巧：用 .run() 而不是 .delay() 来调用这个任务。
.delay()会真的通过Redis把任务发出去，需要Celery worker和Redis都在运行；
.run()则是直接同步执行任务函数本身的代码逻辑，跳过整个消息队列层，
这是官方推荐的、测试Celery任务业务逻辑的标准做法。

注意：tasks.py里 `from rag_core import embed_model, collection` 这两个名字，
拿到的是conftest.py里造的假对象（因为rag_core在sys.modules里已经被替换过了），
所以这里配置 tasks.embed_model / tasks.collection 的返回值，
跟main.py里、conftest.py里的mock_embed_model/mock_collection fixture，
其实都是同一个对象，配置一处，处处生效。
"""

import pytest
import tasks as tasks_module
from tasks import process_document_task


def test_process_document_task_success(tmp_path):
    """成功路径：文档被正确解析、切块、embedding、存入向量库，并返回处理结果摘要"""
    file_path = tmp_path / "test.txt"
    content = "这是一段测试文档内容，用来验证异步任务能不能正确处理文档。"
    file_path.write_text(content, encoding="utf-8")

    tasks_module.embed_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

    result = process_document_task.run(str(file_path), "test.txt")

    assert result["status"] == "done"
    assert result["filename"] == "test.txt"
    assert result["chunks_count"] >= 1
    assert result["chars_total"] == len(content)
    # 验证真的调用了collection.add去写入向量库
    tasks_module.collection.add.assert_called_once()


def test_process_document_task_cleans_up_temp_file_after_success(tmp_path):
    """处理成功之后，临时上传的文件应该被删除，不能一直堆在共享目录里占空间"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("测试内容", encoding="utf-8")

    tasks_module.embed_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]

    process_document_task.run(str(file_path), "test.txt")

    assert not file_path.exists()


def test_process_document_task_cleans_up_temp_file_even_on_failure(tmp_path):
    """即使处理过程中途报错（比如写入向量库失败），临时文件也应该被清理掉，
    不能因为出错就留下垃圾文件——这是finally块要保证的行为"""
    file_path = tmp_path / "broken.txt"
    file_path.write_text("内容", encoding="utf-8")

    tasks_module.embed_model.encode.return_value.tolist.return_value = [[0.1]]
    tasks_module.collection.add.side_effect = RuntimeError("模拟Chroma写入失败")

    with pytest.raises(RuntimeError):
        process_document_task.run(str(file_path), "broken.txt")

    assert not file_path.exists()


def test_process_document_task_ids_are_unique_per_chunk(tmp_path):
    """验证存入Chroma的每个chunk都有唯一ID（doc_id前缀 + chunk序号），
    这个ID格式是main.py里/chat接口做溯源(sources)时依赖的基础"""
    file_path = tmp_path / "test.txt"
    # 构造一段足够长的文本，确保能切出至少2个chunk
    file_path.write_text("测试内容" * 100, encoding="utf-8")

    tasks_module.embed_model.encode.return_value.tolist.return_value = [
        [0.1] for _ in range(10)
    ]

    process_document_task.run(str(file_path), "test.txt")

    call_kwargs = tasks_module.collection.add.call_args.kwargs
    ids = call_kwargs["ids"]
    # 所有生成的id都应该是独一无二的，不能有重复
    assert len(ids) == len(set(ids))
