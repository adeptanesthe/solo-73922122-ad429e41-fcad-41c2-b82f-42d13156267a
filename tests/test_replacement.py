"""换版核对自测：保存新原树时与获准候选树的一致性核对、停用、并发与持久化。"""

import json
import os
import tempfile
import time
import unittest

from app.solver import check_takeover_compatibility
from app.store import ConflictError, Store
from app.tree import classify
from app.web import create_app


def _leaf(category):
    return {"leaf": category}


def _node(feature, threshold, left, right):
    return {"feature": feature, "threshold": threshold, "left": left, "right": right}


# 与内置示例一同构：原树高温高后压 REJECT，候选树同位置 REWORK。
ORIGINAL = _node(
    "temperature", 200,
    _leaf("PASS"),
    _node("back", 400, _leaf("PASS"), _leaf("REJECT")),
)
CANDIDATE = _node(
    "temperature", 200,
    _leaf("PASS"),
    _node("back", 400, _leaf("PASS"), _leaf("REWORK")),
)
DRAFT = {
    "temperature": [150, 300],
    "front": [100, 200],
    "back": [150, 500],
    "diff": [0, 300],
}


class ReplacementWebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "state.json")
        app = create_app(data_file=self.db)
        app.testing = True
        self.client = app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _confirm_example(self):
        r = self.client.post("/api/original-tree", json={"tree": ORIGINAL})
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/draft", json={
            "candidate_tree": CANDIDATE, "draft": DRAFT,
        })
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/trials")
        job_id = r.get_json()["job_id"]
        for _ in range(50):
            job = self.client.get(f"/api/trials/{job_id}").get_json()["job"]
            if job["status"] not in ("pending", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "solved")
        self.assertEqual(job["result"]["zone"], [150, 200])
        r = self.client.post("/api/confirmations", json={"trial_id": job_id})
        self.assertEqual(r.status_code, 200)
        return job_id

    def _classify(self, temperature, front, back):
        return self.client.post("/api/classify", json={
            "temperature": temperature, "front": front, "back": back,
        }).get_json()

    # ---------- 核对通过：接管继续 ----------
    def test_compatible_new_tree_keeps_takeover(self):
        self._confirm_example()
        # 新原树仅在获准温区/工况之外改变判定（温度 100 处 REWORK），
        # 获准域 [150,200] 内与候选树完全一致。
        new_tree = _node(
            "temperature", 149,
            _leaf("REWORK"),
            _node("temperature", 200,
                  _leaf("PASS"),
                  _node("back", 400, _leaf("PASS"), _leaf("REJECT"))),
        )
        r = self.client.post("/api/original-tree", json={"tree": new_tree})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["takeover"], "active")
        # 确认方案仍在、没有停用记录。
        state = self.client.get("/api/state").get_json()
        self.assertIsNotNone(state["confirmation"])
        self.assertIsNone(state["deactivation"])
        # 获准合规读数仍走候选树。
        data = self._classify(180, 150, 300)
        self.assertEqual(data["tree_used"], "candidate")

    # ---------- 核心验收：新原树判退的合规读数不再被候选树放行 ----------
    def test_disagreeing_new_tree_deactivates_takeover(self):
        self._confirm_example()
        # 新原树：温区 150..200 内原本 PASS 的合规读数改判 REJECT，
        # 而候选树在获准域内判 PASS —— 新原树判退的读数绝不能再被放行。
        new_tree = _node(
            "temperature", 200,
            _leaf("REJECT"),
            _node("back", 400, _leaf("PASS"), _leaf("REJECT")),
        )
        r = self.client.post("/api/original-tree", json={"tree": new_tree})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["takeover"], "deactivated")
        self.assertIn("分歧", data["message"])

        state = self.client.get("/api/state").get_json()
        self.assertIsNone(state["confirmation"])
        deactivation = state["deactivation"]
        self.assertIsNotNone(deactivation)
        self.assertIn("150", deactivation["reason"])
        # 示例分歧读数必须真实落在获准域内且两树分类确实不同。
        sample = deactivation["sample"]
        self.assertIsNotNone(sample)
        sr = sample["readings"]
        self.assertTrue(150 <= sr["temperature"] <= 200)
        self.assertTrue(100 <= sr["front"] <= 200)
        self.assertTrue(150 <= sr["back"] <= 500)
        self.assertTrue(0 <= sr["back"] - sr["front"] <= 300)
        self.assertEqual(sample["original_category"],
                         classify(new_tree, sr))
        self.assertEqual(sample["candidate_category"],
                         classify(CANDIDATE, sr))
        self.assertNotEqual(sample["original_category"],
                            sample["candidate_category"])

        # 停用后：原先被候选树判 PASS 的合规读数现在由新原树判 REJECT，
        # 且响应带上停用原因。
        data = self._classify(180, 150, 300)
        self.assertEqual(data["tree_used"], "original")
        self.assertFalse(data["eligible"])
        self.assertEqual(data["category"], "REJECT")
        self.assertIsNotNone(data["deactivation"])
        self.assertIn("停用", data["reason"])

    # ---------- 停用持久化：重启后仍由新原树分拣 ----------
    def test_deactivation_persists_across_restart(self):
        self._confirm_example()
        new_tree = _node(
            "temperature", 200,
            _leaf("REJECT"),
            _node("back", 400, _leaf("PASS"), _leaf("REJECT")),
        )
        r = self.client.post("/api/original-tree", json={"tree": new_tree})
        self.assertEqual(r.get_json()["takeover"], "deactivated")

        app2 = create_app(data_file=self.db)
        client2 = app2.test_client()
        state = client2.get("/api/state").get_json()
        self.assertIsNone(state["confirmation"])
        self.assertIsNotNone(state["deactivation"])
        data = client2.post("/api/classify", json={
            "temperature": 180, "front": 150, "back": 300,
        }).get_json()
        self.assertEqual(data["tree_used"], "original")
        self.assertEqual(data["category"], "REJECT")

    # ---------- 换版保留最新草稿与候选树 ----------
    def test_replace_keeps_latest_draft(self):
        self._confirm_example()
        state0 = self.client.get("/api/state").get_json()
        # 先保存一份更新过的草稿（换版前的最新草稿）。
        newer_draft = dict(DRAFT, temperature=[120, 280])
        r = self.client.post("/api/draft", json={
            "candidate_tree": CANDIDATE, "draft": newer_draft,
        })
        self.assertEqual(r.status_code, 200)

        new_tree = _leaf("REJECT")  # 单叶树，与候选树在获准域内有分歧
        r = self.client.post("/api/original-tree", json={"tree": new_tree})
        self.assertEqual(r.status_code, 200)

        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["draft"], newer_draft)
        self.assertEqual(state["candidate_tree"], state0["candidate_tree"])
        self.assertEqual(state["original_tree"], new_tree)

    # ---------- 无接管时换版不产生停用记录 ----------
    def test_replace_without_confirmation(self):
        r = self.client.post("/api/original-tree", json={"tree": ORIGINAL})
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/original-tree", json={"tree": _leaf("X")})
        data = r.get_json()
        self.assertEqual(data["takeover"], "none")
        state = self.client.get("/api/state").get_json()
        self.assertIsNone(state["confirmation"])
        self.assertIsNone(state["deactivation"])

    # ---------- 校验/核对失败：取消换版，原状态不变 ----------
    def test_invalid_tree_cancels_replacement(self):
        self._confirm_example()
        before = self.client.get("/api/state").get_json()
        bad = {"feature": "nope", "threshold": 1,
               "left": _leaf("A"), "right": _leaf("B")}
        r = self.client.post("/api/original-tree", json={"tree": bad})
        self.assertEqual(r.status_code, 400)
        after = self.client.get("/api/state").get_json()
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["original_tree"], before["original_tree"])
        self.assertIsNotNone(after["confirmation"])

    # ---------- 并发：换版期间另一操作先生效，自动按最新状态重查 ----------
    def test_concurrent_change_retried_with_latest_state(self):
        self._confirm_example()
        store = self.client.application.config["STORE"]
        original_snapshot = store.snapshot_state

        calls = {"n": 0}

        def flaky_snapshot():
            calls["n"] += 1
            snap = original_snapshot()
            if calls["n"] == 1:
                # 模拟第一次取快照后、条件写入前，另一保存先推进了 revision。
                store.save_draft(
                    json.loads(json.dumps(CANDIDATE)),
                    json.loads(json.dumps(DRAFT)),
                )
            return snap

        store.snapshot_state = flaky_snapshot
        try:
            r = self.client.post("/api/original-tree",
                                 json={"tree": _leaf("REJECT")})
        finally:
            store.snapshot_state = original_snapshot
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(calls["n"], 2)  # 至少重查了一次
        self.assertEqual(r.get_json()["takeover"], "deactivated")

    def test_persistent_conflict_returns_409_and_keeps_state(self):
        self._confirm_example()
        store = self.client.application.config["STORE"]
        before = self.client.get("/api/state").get_json()
        # 每次条件写入前都被别的保存抢先：revision 永远对不上。
        original_replace = store.replace_original_tree
        original_save = store.save_original_tree

        def always_conflict(*args, **kwargs):
            store.save_draft(
                json.loads(json.dumps(CANDIDATE)),
                json.loads(json.dumps(DRAFT)),
            )
            return original_replace(*args, **kwargs)

        store.replace_original_tree = always_conflict
        store.save_original_tree = lambda *a, **k: (_ for _ in ()).throw(
            ConflictError("forced"))
        try:
            r = self.client.post("/api/original-tree", json={"tree": _leaf("REJECT")})
        finally:
            store.replace_original_tree = original_replace
            del store.save_original_tree
        self.assertEqual(r.status_code, 409)
        self.assertIn("重试", r.get_json()["error"])
        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["original_tree"], before["original_tree"])
        self.assertIsNotNone(state["confirmation"])

    # ---------- 新原树先保存成功，旧版确认必须被拒绝 ----------
    def test_confirmation_after_replacement_is_rejected(self):
        job_id = self._confirm_example()
        # 再发起一个新试算（与当前 revision 绑定），随后换版推进 revision。
        r = self.client.post("/api/trials")
        second_job = r.get_json()["job_id"]
        for _ in range(50):
            job = self.client.get(f"/api/trials/{second_job}").get_json()["job"]
            if job["status"] not in ("pending", "running"):
                break
            time.sleep(0.05)

        new_tree = _leaf("REJECT")
        r = self.client.post("/api/original-tree", json={"tree": new_tree})
        self.assertEqual(r.status_code, 200)

        # 换版前发起、仍在等待写入的确认（无论新旧任务）一律拒绝。
        r = self.client.post("/api/confirmations", json={"trial_id": second_job})
        self.assertEqual(r.status_code, 409)
        r = self.client.post("/api/confirmations", json={"trial_id": job_id})
        self.assertEqual(r.status_code, 409)
        state = self.client.get("/api/state").get_json()
        self.assertIsNone(state["confirmation"])

    # ---------- 停用后重新试算确认可再次接管，并清掉停用记录 ----------
    def test_reconfirm_after_deactivation_clears_record(self):
        self._confirm_example()
        new_tree = _node(
            "temperature", 200,
            _leaf("REJECT"),
            _node("back", 400, _leaf("PASS"), _leaf("REJECT")),
        )
        r = self.client.post("/api/original-tree", json={"tree": new_tree})
        self.assertEqual(r.get_json()["takeover"], "deactivated")

        # 用与新原树完全相同的候选树重新走流程：全域一致，可再次确认。
        r = self.client.post("/api/draft", json={
            "candidate_tree": new_tree, "draft": DRAFT,
        })
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/trials")
        job_id = r.get_json()["job_id"]
        for _ in range(50):
            job = self.client.get(f"/api/trials/{job_id}").get_json()["job"]
            if job["status"] not in ("pending", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "solved")
        r = self.client.post("/api/confirmations", json={"trial_id": job_id})
        self.assertEqual(r.status_code, 200)
        state = self.client.get("/api/state").get_json()
        self.assertIsNotNone(state["confirmation"])
        self.assertIsNone(state["deactivation"])
        data = self._classify(180, 150, 300)
        self.assertEqual(data["tree_used"], "candidate")
        self.assertEqual(data["category"], "REJECT")


class CheckFunctionTests(unittest.TestCase):
    """核对函数本身：与暴力枚举对拍 + 不枚举压力组合的结构保证。"""

    def test_check_matches_brute_force_on_small_domain(self):
        import random
        from app.tree import validate_tree_full
        rng = random.Random(20261123)
        from tests.test_solver import random_tree

        for _ in range(300):
            new_orig = random_tree(rng)
            cand = random_tree(rng)
            validate_tree_full(new_orig)
            validate_tree_full(cand)
            # 获准温区/工况就是一份普通草稿域。
            draft = {
                "temperature": [rng.randint(0, 3), rng.randint(4, 9)],
                "front": [rng.randint(0, 4), rng.randint(5, 9)],
                "back": [rng.randint(0, 4), rng.randint(5, 9)],
                "diff": [rng.randint(-4, 2), rng.randint(3, 9)],
            }
            report = check_takeover_compatibility(
                new_orig, cand, draft["temperature"], draft
            )
            # 暴力法：枚举获准域内全部合规读数，看是否存在分类分歧。
            brute_disagree_temps = set()
            for t in range(draft["temperature"][0], draft["temperature"][1] + 1):
                for f in range(draft["front"][0], draft["front"][1] + 1):
                    for b in range(draft["back"][0], draft["back"][1] + 1):
                        if not (draft["diff"][0] <= b - f <= draft["diff"][1]):
                            continue
                        reading = {"temperature": t, "front": f, "back": b}
                        if classify(new_orig, reading) != classify(cand, reading):
                            brute_disagree_temps.add(t)
            interval_temps = set()
            for lo, hi in report["disagreement_intervals"]:
                interval_temps.update(range(lo, hi + 1))
            self.assertEqual(interval_temps, brute_disagree_temps)
            self.assertEqual(report["compatible"], not brute_disagree_temps)
            if brute_disagree_temps:
                sample = report["sample"]
                self.assertIsNotNone(sample)
                sr = sample["readings"]
                self.assertNotEqual(
                    classify(new_orig, sr), classify(cand, sr)
                )

    def test_check_uses_approved_zone_not_full_draft(self):
        # 分歧只发生在温度 201..300；获准温区仅 [150,200] → 必须核对通过。
        report = check_takeover_compatibility(ORIGINAL, CANDIDATE, [150, 200], DRAFT)
        self.assertTrue(report["compatible"])
        self.assertEqual(report["disagreement_intervals"], [])
        # 同一对树、把温区放宽到 [150,300] 就能发现分歧。
        report2 = check_takeover_compatibility(ORIGINAL, CANDIDATE, [150, 300], DRAFT)
        self.assertFalse(report2["compatible"])
        self.assertTrue(report2["disagreement_intervals"])

    def test_check_respects_pressure_and_diff_constraints(self):
        # 分歧盒子在 back>400；把获准后压收紧到 [150,400] 后分歧不可达。
        tight = dict(DRAFT, back=[150, 400])
        report = check_takeover_compatibility(ORIGINAL, CANDIDATE, [150, 300], tight)
        self.assertTrue(report["compatible"])


class StoreConditionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "state.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_conditional_writes_raise_conflict_and_rollback(self):
        rev = self.store.save_original_tree(ORIGINAL)
        # 另一保存先推进 revision。
        self.store.save_draft(CANDIDATE, DRAFT)
        with self.assertRaises(ConflictError):
            self.store.save_original_tree(_leaf("X"), expected_revision=rev)
        with self.assertRaises(ConflictError):
            self.store.replace_original_tree(
                _leaf("X"), rev, confirmation_change="deactivate",
                deactivation={"reason": "x"},
            )
        # 冲突时原树未被改动。
        self.assertEqual(self.store.snapshot_state()["original_tree"], ORIGINAL)

    def test_deactivate_is_atomic_with_tree_replace(self):
        rev = self.store.save_original_tree(ORIGINAL)
        self.store.save_draft(CANDIDATE, DRAFT)
        # 用一个假 job 直接走 confirm 的条件写入。
        job = {
            "id": "job1",
            "revision": self.store.get_revision(),
            "result": {"zone": [150, 200]},
            "snapshot": {"draft": DRAFT, "candidate_tree": CANDIDATE},
        }
        self.store.confirm(job)
        self.assertIsNotNone(self.store.snapshot_state()["confirmation"])
        rev = self.store.get_revision()
        new_rev = self.store.replace_original_tree(
            _leaf("PASS"), rev, confirmation_change="deactivate",
            deactivation={"reason": "停用测试", "zone": [150, 200]},
        )
        state = self.store.snapshot_state()
        self.assertEqual(new_rev, rev + 1)
        self.assertIsNone(state["confirmation"])
        self.assertEqual(state["deactivation"]["reason"], "停用测试")
        self.assertEqual(state["original_tree"], _leaf("PASS"))
        # 草稿未丢。
        self.assertEqual(state["draft"], DRAFT)


if __name__ == "__main__":
    unittest.main()
