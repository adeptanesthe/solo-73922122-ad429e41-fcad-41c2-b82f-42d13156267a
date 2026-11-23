"""线程安全的 JSON 文件持久化与试算任务管理。

状态文件结构：
{
  "revision": 整数,                 # 每次保存原树/草稿/确认后自增
  "original_tree": 树 JSON | null,
  "candidate_tree": 树 JSON | null,
  "draft": 工况草稿 | null,
  "confirmation": {                 # 已确认的接管方案，null 表示尚无/已被换版停用
      "revision": 确认时的草稿版本号,
      "zone": [lo, hi],
      "draft": {…工况…},
      "candidate_tree": {…确认时快照…},
      "trial_id": 试算任务 id,
      "confirmed_at": ISO 时间
  } | null,
  "deactivation": {                 # 换版核对停用接管的记录（含原因），null 表示从未停用
      "deactivated_at": ISO 时间,
      "reason": 中文停用原因,
      "zone": 原获准温区,
      "approved_draft": 原获准工况,
      "candidate_trial_id": 原试算 id,
      "disagreement_intervals": 分歧温度区间列表,
      "sample": 示例分歧读数与两树类别
  } | null
}

试算任务保存在进程内存中（守护线程异步执行），重启后需重新试算；
草稿与确认结果落盘，重启不丢。

换版（保存新原树）与确认共用同一把锁并采用“期望 revision”条件写入：
期间若另一次换原树或确认已经生效，条件写入抛出 ConflictError，
由上层按最新状态重查或要求工艺员重试。
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone


class ConflictError(Exception):
    """条件写入失败：调用方持有的 revision 已过期，需按最新状态重查/重试。"""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _default_state():
    return {
        "revision": 0,
        "original_tree": None,
        "candidate_tree": None,
        "draft": None,
        "confirmation": None,
        # 接管被换版核对停用后的停用记录（含原因），null 表示从未停用。
        "deactivation": None,
    }


class Store:
    MAX_JOBS = 20

    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        self._jobs_lock = threading.RLock()
        self._jobs = {}  # job_id -> 任务 dict
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._lock:
            if not os.path.exists(path):
                self._write_locked(_default_state())

    # ---------- 底层读写 ----------
    def _read_locked(self):
        with open(self.path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        # 兼容旧版本状态文件：补出换版停用记录字段。
        if "deactivation" not in state:
            state["deactivation"] = None
        return state

    def _write_locked(self, state):
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def snapshot_state(self):
        """读取当前完整状态（加锁），供换版核对前取快照。"""
        with self._lock:
            return self._read_locked()

    def get_state(self):
        with self._lock:
            state = self._read_locked()
            revision = state["revision"]
        state["jobs"] = self.public_jobs(revision)
        return state

    def get_revision(self):
        with self._lock:
            return self._read_locked()["revision"]

    # ---------- 保存 ----------
    def save_original_tree(self, tree, expected_revision=None):
        """保存原树。

        expected_revision 给定时为条件写入：仅当当前 revision 未变化才提交，
        否则抛 ConflictError；revision 自增与文件替换在同一把锁内完成。
        候选树与工况草稿（含最新草稿）原样保留，不受换版影响。
        """
        with self._lock:
            state = self._read_locked()
            if expected_revision is not None and state["revision"] != expected_revision:
                raise ConflictError(
                    f"期望 revision={expected_revision}，当前已为 {state['revision']}"
                )
            state["original_tree"] = tree
            state["revision"] += 1
            self._write_locked(state)
            return state["revision"]

    def replace_original_tree(self, tree, expected_revision, confirmation_change=None,
                              deactivation=None):
        """换版专用条件写入：新原树与（核对后的）接管状态在同一原子事务落盘。

        confirmation_change:
          None        保留现有 confirmation（核对通过或本无接管方案）；
          "deactivate" 停用接管（confirmation 置空，写入 deactivation）。
        草稿与候选树始终原样保留。
        """
        with self._lock:
            state = self._read_locked()
            if state["revision"] != expected_revision:
                raise ConflictError(
                    f"期望 revision={expected_revision}，当前已为 {state['revision']}"
                )
            state["original_tree"] = tree
            if confirmation_change == "deactivate":
                state["confirmation"] = None
                state["deactivation"] = deactivation
            state["revision"] += 1
            self._write_locked(state)
            return state["revision"]

    def save_draft(self, candidate_tree, draft):
        """保存候选树与工况草稿（可在试算进行中随时调用）。"""
        with self._lock:
            state = self._read_locked()
            state["candidate_tree"] = candidate_tree
            state["draft"] = draft
            state["revision"] += 1
            self._write_locked(state)
            return state["revision"]

    def confirm(self, job):
        """依据一个成功且未过期的试算任务写入确认（条件写入，防并发换版）。

        若 revision 已被并发的换原树/草稿保存推进，则抛 ConflictError，
        由上层返回 409，要求工艺员按最新状态重新试算。
        新确认生效时清掉历史停用记录。
        """
        with self._lock:
            state = self._read_locked()
            if state["revision"] != job["revision"]:
                raise ConflictError(
                    f"期望 revision={job['revision']}，当前已为 {state['revision']}"
                )
            state["confirmation"] = {
                "revision": job["revision"],
                "zone": list(job["result"]["zone"]),
                "draft": json.loads(json.dumps(job["snapshot"]["draft"])),
                "candidate_tree": json.loads(
                    json.dumps(job["snapshot"]["candidate_tree"])
                ),
                "trial_id": job["id"],
                "confirmed_at": _now_iso(),
            }
            state["deactivation"] = None
            state["revision"] += 1
            self._write_locked(state)
            return state["revision"], state["confirmation"]

    # ---------- 试算任务（内存态）----------
    def create_job(self, snapshot, revision):
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "status": "pending",  # pending | running | solved | no_solution | error
            "revision": revision,
            "snapshot": snapshot,
            "result": None,
            "error": None,
            "created_at": _now_iso(),
            "finished_at": None,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job
            # 仅保留最近 MAX_JOBS 个任务，避免无界增长。
            if len(self._jobs) > self.MAX_JOBS:
                oldest = sorted(self._jobs, key=lambda jid: self._jobs[jid]["created_at"])
                for old_id in oldest[: len(self._jobs) - self.MAX_JOBS]:
                    self._jobs.pop(old_id, None)
        return job

    def update_job(self, job_id, **changes):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.update(changes)
            return job

    def get_job(self, job_id):
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def public_jobs(self, current_revision=None):
        """对外任务视图；current_revision 给定时标注 stale（草稿已改过）。"""
        if current_revision is None:
            current_revision = self.get_revision()
        with self._jobs_lock:
            return [{
                "id": job["id"],
                "status": job["status"],
                "revision": job["revision"],
                "stale": job["revision"] != current_revision,
                "result": job["result"],
                "error": job["error"],
                "created_at": job["created_at"],
                "finished_at": job["finished_at"],
            } for job in self._jobs.values()]
