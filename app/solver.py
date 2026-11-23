"""试算求解器。

输入两棵树与一份工况草稿（四个闭区间），求最长连续整数温区：
  1. 温区内每个温度都存在“合规压力组合”；
  2. 两棵树对“所有合规读数”判为同类；
  3. 同长取起点较低者。

关键约束：求解不能枚举读数组合。本模块的做法是
  - 把两棵树展开为至多 128 个轴对齐叶子盒子（区间，而非逐点枚举）；
  - 用闭式区间判据检查矩形压力域上后压-前压差是否可行；
  - 对两树“类别不同的叶子盒子对”做矩形相交（O(La*Lb) 次区间运算），
    得到分歧温度区间并集；
  - 在工况温度区间上求补并做区间合并，扫描出最长连续整数段。
温度轴至多 65536 个整数点，仅在最终扫描时按整数段计数，压力组合从不枚举。
"""

from .tree import FEATURES, leaf_boxes


def pressure_diff_feasible(front_lo, front_hi, back_lo, back_hi, d_lo, d_hi):
    """闭区间矩形域上是否存在 (f, b) 使 d_lo <= b - f <= d_hi。

    对独立闭区间，b-f 的取值范围本身就是闭区间 [b_lo-f_hi, b_hi-f_lo]，
    因此只需判断两个整数区间是否相交——无需枚举任何 (f, b) 组合。
    """
    diff_min = back_lo - front_hi
    diff_max = back_hi - front_lo
    return max(diff_min, d_lo) <= min(diff_max, d_hi)


def _boxes_intersect(a, b):
    """两个三维轴对齐盒子（各特征为闭区间）是否相交。

    先拒绝空盒子（某维 lo>hi），否则空区间会被区间谓词误判为相交。
    """
    for box in (a, b):
        if any(lo > hi for lo, hi in box.values()):
            return False
    return all(a[f][0] <= b[f][1] and b[f][0] <= a[f][1] for f in FEATURES)


def _intersect_boxes(a, b):
    """返回两个盒子的交集盒子（调用方已保证它们相交）。"""
    return {f: [max(a[f][0], b[f][0]), min(a[f][1], b[f][1])] for f in FEATURES}


def _merge_intervals(intervals):
    """合并互相重叠或邻接的整数闭区间，输入允许无序。"""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + 1:  # 整数轴上相邻即连续
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _difference_intervals(universe_lo, universe_hi, covered):
    """求 [universe_lo, universe_hi] 相对已合并区间并集 covered 的补集。"""
    result = []
    cursor = universe_lo
    for lo, hi in covered:
        if hi < cursor:
            continue
        if lo > cursor:
            result.append([cursor, lo - 1])
        cursor = max(cursor, hi + 1)
        if cursor > universe_hi:
            break
    if cursor <= universe_hi:
        result.append([cursor, universe_hi])
    return result


def run_trial(original_tree, candidate_tree, draft):
    """执行一次试算。

    返回 dict：
      status: "solved" | "no_solution" | "error"
      zone: [lo, hi] 或 None
      length: 整数
      reason: 中文说明
      pressure_feasible: 工况压力组合是否可行
      bad_interval_count / leaf_pair_count: 统计信息，便于页面展示
    """
    t_lo, t_hi = draft["temperature"]
    f_lo, f_hi = draft["front"]
    b_lo, b_hi = draft["back"]
    d_lo, d_hi = draft["diff"]

    base = {
        "status": "error",
        "zone": None,
        "length": 0,
        "reason": "",
        "pressure_feasible": False,
        "bad_interval_count": 0,
        "leaf_pair_count": 0,
    }

    # 第一步：工况压力组合本身是否可行（闭式判据，不枚举）。
    feasible = pressure_diff_feasible(f_lo, f_hi, b_lo, b_hi, d_lo, d_hi)
    base["pressure_feasible"] = feasible
    if not feasible:
        base["status"] = "no_solution"
        base["reason"] = (
            "工况无解：在给定前压、后压闭区间内，后压-前压无法落入"
            f"[{d_lo}, {d_hi}]，不存在任何合规压力组合"
        )
        return base

    # 第二步：展开两棵树的叶子盒子。
    orig_boxes = leaf_boxes(original_tree)
    cand_boxes = leaf_boxes(candidate_tree)

    # 第三步：找出“类别不同且在合规压力域内可达”的叶子对，
    # 收集它们在温度轴上的分歧区间。合规压力域是一个三维盒子：
    # 温度取工况温区，压力取各自工况区间并附加压差约束的可达范围。
    pressure_box = {"temperature": [t_lo, t_hi], "front": [f_lo, f_hi], "back": [b_lo, b_hi]}

    bad_t_intervals = []
    pair_count = 0
    for o_box, o_cat in orig_boxes:
        for c_box, c_cat in cand_boxes:
            if o_cat == c_cat:
                continue
            pair_count += 1
            inter = _intersect_boxes(o_box, c_box)
            # 分歧盒子必须非空且与工况域相交。
            if not _boxes_intersect(inter, pressure_box):
                continue
            # 关键：先把分歧盒子裁剪到工况域，再在裁剪后的压力范围上
            # 判压差可行性，否则域外的压力取值会造成“伪可行”。
            clipped = _intersect_boxes(inter, pressure_box)
            if not pressure_diff_feasible(
                clipped["front"][0], clipped["front"][1],
                clipped["back"][0], clipped["back"][1],
                d_lo, d_hi,
            ):
                continue
            bad_t_intervals.append([clipped["temperature"][0], clipped["temperature"][1]])

    base["leaf_pair_count"] = pair_count
    merged_bad = _merge_intervals(bad_t_intervals)
    base["bad_interval_count"] = len(merged_bad)

    # 第四步：工况温区上的补集即为“全程两树一致”的连续段候选。
    good_intervals = _difference_intervals(t_lo, t_hi, merged_bad)
    if not good_intervals:
        base["status"] = "no_solution"
        base["reason"] = (
            "工况压力组合可行，但工况温区 "
            f"[{t_lo}, {t_hi}] 内每个温度都存在使两树分类不一致的合规读数，"
            "没有可接管的连续温区"
        )
        return base

    # 第五步：取最长段；同长取起点最低。区间已按起点升序，min 自然满足平局规则。
    best = min(good_intervals, key=lambda iv: (-(iv[1] - iv[0] + 1), iv[0]))
    base["status"] = "solved"
    base["zone"] = [best[0], best[1]]
    base["length"] = best[1] - best[0] + 1
    base["reason"] = (
        f"试算成功：最长连续整数温区为 [{best[0]}, {best[1]}]，共 "
        f"{base['length']} 个温度；该温区内每个温度都有合规压力组合，"
        "且两树对所有合规读数判为同类"
    )
    return base


def _witness_point(box, d_lo, d_hi):
    """在分歧盒子与压差约束的交集中构造一条具体读数（闭式，不枚举）。

    给定 front 闭区间 [f_lo,f_hi]、back 闭区间 [b_lo,b_hi]，
    b-f 的可达范围是 [b_lo-f_hi, b_hi-f_lo]；调用方已判定它与
    [d_lo,d_hi] 相交。按以下顺序取点，可一次落在交集内：
      1. 取交集中的最小压差 d = max(d_lo, b_lo-f_hi)；
      2. 后压取 max(b_lo, f_lo+d)，保证 back >= b_lo 且 back-front >= d_lo；
      3. 前压取 back-d，由交集非空可证它落在 [f_lo,f_hi]，
         且 back-front = d 同时满足 d <= d_hi 与各压力区间约束。
    """
    f_lo, f_hi = box["front"]
    b_lo, b_hi = box["back"]
    diff = max(d_lo, b_lo - f_hi)
    back = max(b_lo, f_lo + diff)
    front = back - diff
    # 防御性自检：构造点必须落在盒子与压差范围内。
    assert f_lo <= front <= f_hi and b_lo <= back <= b_hi and d_lo <= back - front <= d_hi
    # 取该盒子温度区间内的任一整数（区间非空，取下限即可）。
    return box["temperature"][0], front, back


def check_takeover_compatibility(new_original_tree, candidate_tree, zone, approved_draft):
    """换原树时核对：新原树与获准候选树对“所有获准合规读数”分类是否一致。

    获准合规读数域（一个三维盒子加一条压差约束）：
      温度 ∈ zone [z_lo, z_hi]（获准温区），
      前压 ∈ approved_draft.front，后压 ∈ approved_draft.back，
      后压-前压 ∈ approved_draft.diff。
    本函数与 run_trial 一样只做叶子盒子级别的区间运算（O(La*Lb)），
    不逐组枚举任何读数；分歧点的具体读数由 _witness_point 闭式构造。

    返回 dict：
      compatible: bool
      disagreement_intervals: 获准温区内两树分类不同的温度整数区间（已合并）
      sample: 首处分歧的示例读数与两树类别（compatible 为 True 时为 None）
    """
    z_lo, z_hi = zone
    f_lo, f_hi = approved_draft["front"]
    b_lo, b_hi = approved_draft["back"]
    d_lo, d_hi = approved_draft["diff"]

    approved_box = {
        "temperature": [z_lo, z_hi],
        "front": [f_lo, f_hi],
        "back": [b_lo, b_hi],
    }

    orig_boxes = leaf_boxes(new_original_tree)
    cand_boxes = leaf_boxes(candidate_tree)

    bad_t_intervals = []
    sample = None
    for o_box, o_cat in orig_boxes:
        for c_box, c_cat in cand_boxes:
            if o_cat == c_cat:
                continue
            inter = _intersect_boxes(o_box, c_box)
            # 分歧盒子必须非空且与获准工况域相交。
            if not _boxes_intersect(inter, approved_box):
                continue
            # 先裁剪到获准工况域，再在裁剪后的压力范围上判压差可行性，
            # 与试算器的“伪可行”防护保持一致。
            clipped = _intersect_boxes(inter, approved_box)
            if not pressure_diff_feasible(
                clipped["front"][0], clipped["front"][1],
                clipped["back"][0], clipped["back"][1],
                d_lo, d_hi,
            ):
                continue
            bad_t_intervals.append([clipped["temperature"][0], clipped["temperature"][1]])
            if sample is None:
                temperature, front, back = _witness_point(clipped, d_lo, d_hi)
                sample = {
                    "readings": {
                        "temperature": temperature,
                        "front": front,
                        "back": back,
                    },
                    "original_category": o_cat,
                    "candidate_category": c_cat,
                }

    merged = _merge_intervals(bad_t_intervals)
    return {
        "compatible": not merged,
        "disagreement_intervals": merged,
        "sample": sample,
    }
