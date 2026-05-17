"""Timestamp helpers for CVI batch plots."""
from __future__ import annotations


def calendar_date(timestamp: str) -> str:
    ts = timestamp.strip()
    return ts.split()[0] if ts else ""


def insert_null_between_calendar_days(
    t: list[str],
    *ys: list[float],
) -> tuple[list[str | None], ...]:
    if len(t) <= 1:
        return (list(t),) + tuple(list(y) for y in ys)
    out_t: list[str | None] = []
    outs: list[list[float | None]] = [[] for _ in ys]
    prev_day: str | None = None
    for i in range(len(t)):
        day = calendar_date(t[i])
        if prev_day is not None and day != prev_day:
            out_t.append(None)
            for o in outs:
                o.append(None)
        out_t.append(t[i])
        for j, o in enumerate(outs):
            o.append(float(ys[j][i]))
        prev_day = day
    return (out_t, *outs)
