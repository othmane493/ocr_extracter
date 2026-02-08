def y_center(block):
    return block["y"] + block["height"] / 2


def compute_dynamic_y_tolerance(blocks, ratio=0.6, min_tol=8, max_tol=35):
    heights = [b["height"] for b in blocks if b["height"] > 0]
    if not heights:
        return 15
    avg = sum(heights) / len(heights)
    tol = int(avg * ratio)
    return max(min_tol, min(tol, max_tol))
