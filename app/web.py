"""Flask 应用：原树录入、候选树与工况草稿、试算、确认、读数分流查询。"""

import json
import os
import threading

from flask import Flask, jsonify, render_template, request

from .examples import EXAMPLES
from .solver import check_takeover_compatibility, run_trial
from .store import ConflictError, Store
from .tree import classify, validate_tree_full
from .validation import validate_draft, validate_reading

# 换版条件写入遭遇并发修改时的有界重试次数（重查最新状态后重新核对）。
MAX_REPLACE_RETRIES = 5


def create_app(data_file=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    data_file = data_file or os.environ.get("SEAL_DATA_FILE", "data/state.json")
    store = Store(data_file)
    app.config["STORE"] = store

    def error(message, status=400):
        return jsonify({"ok": False, "error": message}), status

    # ---------------- 页面 ----------------
    @app.get("/")
    def index():
        return render_template("index.html")

    # ---------------- 元数据 ----------------
    @app.get("/api/state")
    def get_state():
        state = store.get_state()
        return jsonify({"ok": True, **state})

    @app.get("/api/examples")
    def get_examples():
        return jsonify({"ok": True, "examples": EXAMPLES})

    # ---------------- 原树（换版核对） ----------------
    @app.post("/api/original-tree")
    def save_original_tree():
        payload = request.get_json(silent=True)
        if payload is None or "tree" not in payload:
            return error("请求体必须是包含 tree 字段的 JSON")
        tree = payload["tree"]
        # 校验失败直接拒绝，状态文件不发生任何变化（取消这次换版）。
        try:
            validate_tree_full(tree)
        except ValueError as exc:
            return error(f"原树校验失败：{exc}")

        # 乐观并发循环：核对期间若另一次换原树或确认已经生效，
        # 条件写入会失败，按最新状态重查并重做核对；重试用尽后请工艺员重试。
        last_conflict = None
        for _ in range(MAX_REPLACE_RETRIES):
            snapshot = store.snapshot_state()
            expected_revision = snapshot["revision"]
            confirmation = snapshot["confirmation"]

            # 无生效接管方案：直接换原树，草稿与候选树原样保留。
            if confirmation is None:
                try:
                    revision = store.save_original_tree(
                        tree, expected_revision=expected_revision
                    )
                except ConflictError as exc:
                    last_conflict = exc
                    continue
                return jsonify({
                    "ok": True,
                    "revision": revision,
                    "takeover": "none",
                    "message": "当前无生效接管方案，新原树已保存，全部读数由新原树分拣",
                })

            # 已有生效接管：按获准温区、前后压和压差核对新原树与获准候选树。
            # 核对全程只做叶子盒子区间运算，不逐组枚举读数，也不持有写锁，
            # 期间网页仍可按生效方案查询分拣、保存草稿。
            try:
                report = check_takeover_compatibility(
                    tree,
                    confirmation["candidate_tree"],
                    confirmation["zone"],
                    confirmation["draft"],
                )
            except Exception as exc:  # 核对出错：取消这次换版，旧状态不变
                return error(f"新原树与获准候选树核对失败，已取消本次换版：{exc}", 500)

            if report["compatible"]:
                # 两树对每个获准合规读数分类都相同：新原树落盘，接管继续。
                try:
                    revision = store.replace_original_tree(
                        tree, expected_revision
                    )
                except ConflictError as exc:
                    last_conflict = exc
                    continue
                return jsonify({
                    "ok": True,
                    "revision": revision,
                    "takeover": "active",
                    "message": (
                        "核对通过：新原树与获准候选树对获准温区内所有合规读数分类一致，"
                        "新原树已保存，接管继续有效"
                    ),
                })

            # 发现分歧：同时保存新原树并停用接管，停用原因随状态持久化，
            # 重启后由新原树分拣。
            deactivation = _build_deactivation(confirmation, report)
            try:
                revision = store.replace_original_tree(
                    tree,
                    expected_revision,
                    confirmation_change="deactivate",
                    deactivation=deactivation,
                )
            except ConflictError as exc:
                last_conflict = exc
                continue
            return jsonify({
                "ok": True,
                "revision": revision,
                "takeover": "deactivated",
                "deactivation": deactivation,
                "message": deactivation["reason"],
            })

        # 重试期间状态被并发修改了太多次：本次换版取消，请工艺员按最新状态重试。
        return error(
            "换版期间状态已被其他换原树或确认接管操作多次修改，"
            f"本次换版未生效，请刷新后重试（{last_conflict}）",
            409,
        )

    # ---------------- 候选树 + 工况草稿 ----------------
    @app.post("/api/draft")
    def save_draft():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "candidate_tree" not in payload or "draft" not in payload:
            return error("请求体必须包含 candidate_tree 与 draft 字段")
        try:
            validate_tree_full(payload["candidate_tree"])
            draft = validate_draft(payload["draft"])
        except ValueError as exc:
            return error(f"草稿校验失败：{exc}")
        revision = store.save_draft(payload["candidate_tree"], draft)
        return jsonify({"ok": True, "revision": revision})

    # ---------------- 试算（后台线程，不阻塞草稿保存）----------------
    @app.post("/api/trials")
    def create_trial():
        state = store.get_state()
        if state["original_tree"] is None:
            return error("请先录入并保存原树", 409)
        if state["candidate_tree"] is None or state["draft"] is None:
            return error("请先保存候选树与工况草稿", 409)

        revision = state["revision"]
        # 快照：试算只针对发起时的版本，后续改草稿不影响本任务，
        # 任务完成后会以 revision 是否仍为当前版本判断是否过期。
        snapshot = {
            "original_tree": json.loads(json.dumps(state["original_tree"])),
            "candidate_tree": json.loads(json.dumps(state["candidate_tree"])),
            "draft": json.loads(json.dumps(state["draft"])),
        }
        job = store.create_job(snapshot, revision)

        def worker():
            store.update_job(job["id"], status="running")
            try:
                result = run_trial(
                    snapshot["original_tree"],
                    snapshot["candidate_tree"],
                    snapshot["draft"],
                )
                store.update_job(
                    job["id"],
                    status=result["status"],
                    result=result,
                    finished_at=_now(),
                )
            except Exception as exc:  # 试算失败：记录原因，沿用当前分流
                store.update_job(
                    job["id"], status="error", error=str(exc), finished_at=_now()
                )

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "job_id": job["id"], "revision": revision}), 201

    @app.get("/api/trials/<job_id>")
    def get_trial(job_id):
        job = store.get_job(job_id)
        if job is None:
            return error("试算任务不存在", 404)
        current_revision = store.get_revision()
        return jsonify({
            "ok": True,
            "job": {
                "id": job["id"],
                "status": job["status"],
                "revision": job["revision"],
                "stale": job["revision"] != current_revision,
                "current_revision": current_revision,
                "result": job["result"],
                "error": job["error"],
                "created_at": job["created_at"],
                "finished_at": job["finished_at"],
            },
        })

    # ---------------- 确认 ----------------
    @app.post("/api/confirmations")
    def confirm():
        payload = request.get_json(silent=True) or {}
        job_id = payload.get("trial_id")
        if not job_id:
            return error("请提供试算任务 id（trial_id）")
        job = store.get_job(job_id)
        if job is None:
            return error("试算任务不存在，无法确认", 404)

        if job["status"] in ("pending", "running"):
            return error("该试算尚未完成，不能确认", 409)
        if job["status"] == "error":
            return error(f"试算失败，不能确认（原因：{job['error']}），沿用当前分流", 409)
        if job["status"] == "no_solution":
            reason = job["result"]["reason"] if job["result"] else "试算无解"
            return error(f"试算无解，不能确认（原因：{reason}），沿用当前分流", 409)

        current_revision = store.get_revision()
        if job["revision"] != current_revision:
            return error(
                "该试算对应的草稿已被修改（即使改回原值也必须重新试算），"
                "请用当前草稿重新试算后再确认",
                409,
            )
        if job["status"] != "solved" or not job.get("result") or not job["result"].get("zone"):
            return error("试算未产出有效温区，不能确认", 409)

        try:
            revision, confirmation = store.confirm(job)
        except ConflictError:
            # 典型竞态：换原树已先一步生效，旧版确认必须拒绝，
            # 请工艺员按最新状态重新试算后再确认。
            return error(
                "确认期间状态已被换原树或其他保存操作修改，"
                "本次确认未生效，请按最新状态重新试算后再确认",
                409,
            )
        return jsonify({"ok": True, "revision": revision, "confirmation": confirmation})

    # ---------------- 读数分流查询 ----------------
    @app.post("/api/classify")
    def classify_reading():
        state = store.get_state()
        if state["original_tree"] is None:
            return error("尚未录入原树，无法分流", 409)
        payload = request.get_json(silent=True)
        try:
            temperature, front, back = validate_reading(payload)
        except ValueError as exc:
            return error(f"读数校验失败：{exc}")

        readings = {"temperature": temperature, "front": front, "back": back}
        confirmation = state["confirmation"]

        tree_used = "original"
        deactivation = state.get("deactivation")
        if deactivation is not None:
            reason = (
                "接管已因换版核对发现分歧而停用，全部读数使用新原树"
                f"（停用原因：{deactivation['reason']}）"
            )
        else:
            reason = "没有已确认的接管方案，全部读数使用原树"
        eligible = False
        if confirmation is not None:
            zone = confirmation["zone"]
            draft = confirmation["draft"]
            in_zone = zone[0] <= temperature <= zone[1]
            in_front = draft["front"][0] <= front <= draft["front"][1]
            in_back = draft["back"][0] <= back <= draft["back"][1]
            diff = back - front
            in_diff = draft["diff"][0] <= diff <= draft["diff"][1]
            # 获准工况以确认时的工况区间为准（温度由温区限定）。
            eligible = in_zone and in_front and in_back and in_diff
            if eligible:
                tree_used = "candidate"
                reason = (
                    f"读数位于已确认温区 [{zone[0]},{zone[1]}] 且符合获准工况，"
                    "使用候选树"
                )
            else:
                reasons = []
                if not in_zone:
                    reasons.append(f"温度 {temperature} 不在温区 [{zone[0]},{zone[1]}]")
                if not in_front:
                    reasons.append("前压不在获准区间")
                if not in_back:
                    reasons.append("后压不在获准区间")
                if not in_diff:
                    reasons.append(f"后压-前压={diff} 不在获准压差区间")
                reason = "；".join(reasons) + "，使用原树"

        tree = (
            confirmation["candidate_tree"]
            if tree_used == "candidate"
            else state["original_tree"]
        )
        category = classify(tree, readings)
        return jsonify({
            "ok": True,
            "readings": readings,
            "tree_used": tree_used,
            "eligible": eligible,
            "category": category,
            "reason": reason,
            "confirmation_zone": confirmation["zone"] if confirmation else None,
            "deactivation": deactivation,
        })

    return app


def _build_deactivation(confirmation, report):
    """根据核对报告构造持久化的停用记录（含中文停用原因与示例分歧读数）。"""
    from datetime import datetime, timezone

    zone = confirmation["zone"]
    draft = confirmation["draft"]
    intervals = report["disagreement_intervals"]
    interval_text = "、".join(f"[{lo}, {hi}]" for lo, hi in intervals)
    sample = report["sample"]
    sample_text = ""
    if sample is not None:
        r = sample["readings"]
        sample_text = (
            f"；示例合规读数 (温度={r['temperature']}, 前压={r['front']}, "
            f"后压={r['back']})：新原树判 {sample['original_category']}，"
            f"获准候选树判 {sample['candidate_category']}"
        )
    reason = (
        f"换版核对发现分歧：在获准温区 [{zone[0]}, {zone[1]}] 内，"
        f"温度区间 {interval_text} 存在使新原树与获准候选树分类不一致的合规读数"
        f"（获准前压 [{draft['front'][0]}, {draft['front'][1]}]、"
        f"后压 [{draft['back'][0]}, {draft['back'][1]}]、"
        f"压差 [{draft['diff'][0]}, {draft['diff'][1]}]）{sample_text}；"
        "新原树已保存，接管同时停用，此后全部读数由新原树分拣"
    )
    return {
        "deactivated_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "zone": list(zone),
        "approved_draft": json.loads(json.dumps(draft)),
        "candidate_trial_id": confirmation.get("trial_id"),
        "disagreement_intervals": [list(iv) for iv in intervals],
        "sample": sample,
    }


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
