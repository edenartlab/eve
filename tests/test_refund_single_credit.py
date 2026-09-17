"""A task must be refunded at most once, even under a concurrent double-cancel.

The regression this locks down: refund_manna claimed the refund with an upsert
on {task, type:"refund"} and its own comment asserted that was race-proof
"because of the UNIQUE index on {task, type} (see
Transaction.ensure_unique_refund_index)". That function never existed, and the
only {task,type} index in production is NON-unique (added for a slow query in
scripts/add_cost_indexes.py). Two concurrent upserts that both match nothing
therefore both insert, and both credit.

This is not hypothetical: two refund callers race by construction on every
cancel of a running task — Tool.handle_cancel in the API container and
_task_handler's finally block in the Modal worker.

The claim now happens on the task document via a conditional
find_one_and_update on _id, so it relies only on the primary key index.
"""

from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from eve.task import Task


class _FakeTaskCollection:
    """Emulates the {"refunded": {"$ne": True}} conditional claim."""

    def __init__(self):
        self.refunded = False

    def find_one_and_update(self, filt, update):
        if self.refunded:
            return None  # someone else already claimed it
        self.refunded = True
        return {"_id": filt["_id"], "refunded": False}

    def update_one(self, filt, update):
        if update.get("$set", {}).get("refunded") is False:
            self.refunded = False  # claim released after a failed credit


class _FakeTxnCollection:
    def __init__(self, spend_txn=None):
        self.spend_txn = spend_txn
        self.upserts = []

    def find_one(self, filt):
        return self.spend_txn if filt.get("type") == "spend" else None

    def update_one(self, filt, update, upsert=False):
        self.upserts.append(update)
        return MagicMock(upserted_id=ObjectId())


def _task(cost=100.0, n_samples=1):
    return Task(
        user=ObjectId(),
        tool="flux_dev",
        output_type="image",
        args={"n_samples": n_samples},
        cost=cost,
        status="cancelled",
        result=[],
    )


@pytest.fixture
def wired():
    """Patch the collaborators refund_manna reaches out to."""
    task_coll = _FakeTaskCollection()
    txn_coll = _FakeTxnCollection()
    manna = MagicMock()
    manna.id = ObjectId()

    creation_coll = MagicMock()
    creation_coll.count_documents.return_value = 0  # nothing delivered

    with patch.object(Task, "get_collection", return_value=task_coll), patch(
        "eve.task.Transaction"
    ) as txn, patch("eve.task.Manna") as manna_cls, patch(
        "eve.task.Creation"
    ) as creation:
        txn.get_collection.return_value = txn_coll
        manna_cls.load.return_value = manna
        creation.get_collection.return_value = creation_coll
        yield task_coll, txn_coll, manna


def test_refund_credits_exactly_once(wired):
    _, _, manna = wired
    task = _task()

    task.refund_manna()
    task.refund_manna()  # the racing caller

    assert manna.refund.call_count == 1
    assert manna.refund.call_args.args[0] == pytest.approx(100.0)


def test_second_caller_writes_no_second_ledger_row(wired):
    _, txn_coll, _ = wired
    task = _task()

    task.refund_manna()
    task.refund_manna()

    assert len(txn_coll.upserts) == 1


def test_claim_is_released_when_the_credit_fails(wired):
    """A retriable failure must not silently eat manna the user is owed."""
    task_coll, _, manna = wired
    manna.refund.side_effect = RuntimeError("mongo unavailable")
    task = _task()

    with pytest.raises(RuntimeError):
        task.refund_manna()

    assert task_coll.refunded is False  # reclaimable

    manna.refund.side_effect = None
    task.refund_manna()
    assert manna.refund.call_count == 2  # the retry got through


def test_partial_delivery_refunds_only_the_undelivered_fraction(wired):
    _, _, manna = wired
    task = _task(cost=100.0, n_samples=4)
    task.result = [{"output": []}, {"output": []}]  # 2 of 4 delivered

    task.refund_manna()

    assert manna.refund.call_args.args[0] == pytest.approx(50.0)


def test_fully_delivered_task_refunds_nothing(wired):
    _, _, manna = wired
    task = _task(cost=100.0, n_samples=2)
    task.result = [{"output": []}, {"output": []}]

    task.refund_manna()

    manna.refund.assert_not_called()
