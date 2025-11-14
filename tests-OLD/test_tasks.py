# tests/test_tasks.py
import pytest
import asyncio
from eve.tasks import ACTIVE_TASKS, start_task, cancel_task

@pytest.mark.asyncio
async def test_task_cancellation():
    
    # Start a task
    task_id = "my-test-task"
    task = await start_task(task_id)

    # Confirm it's in ACTIVE_TASKS
    assert task_id in ACTIVE_TASKS
    assert ACTIVE_TASKS[task_id] is task

    await asyncio.sleep(1)

    # Cancel the task
    cancel_task(task_id)

    # Await to confirm it raises CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task
    
    # Verify it's removed from ACTIVE_TASKS
    assert task_id not in ACTIVE_TASKS


asyncio.run(test_task_cancellation())