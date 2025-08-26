from __future__ import annotations
import os
import re
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, Tuple
from dateutil import tz, parser as dtparser
from calendar import monthrange


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


TZ_NAME: str = os.getenv("TIMEZONE", "UTC")
DAYFIRST: bool = bool(_env_int("DAYFIRST", 1))
FY_START_MONTH: int = _env_int("FISCAL_YEAR_START_MONTH", 4)
FY_START_DAY: int = _env_int("FISCAL_YEAR_START_DAY", 1)


def _tznow() -> datetime:
    try:
        zone = tz.gettz(TZ_NAME) or tz.UTC
    except Exception:
        zone = tz.UTC
    return datetime.now(zone)


def _to_date_safe(s) -> Optional[date]:
    if not s:
        return None
    if isinstance(s, date):
        return s
    try:
        return dtparser.parse(str(s), dayfirst=DAYFIRST).date()
    except Exception:
        for f in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%b %d %Y", "%d %b %Y", "%B %d %Y", "%d %B %Y"):
            try:
                return datetime.strptime(str(s), f).date()
            except Exception:
                pass
    return None


def _end_of_month(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def _add_months(d: date, n: int) -> date:
    y, m = d.year, d.month
    m2 = m + n
    y2 = y + (m2 - 1) // 12
    m2 = ((m2 - 1) % 12) + 1
    dd = min(d.day, monthrange(y2, m2)[1])
    return date(y2, m2, dd)


def _fiscal_year_for(d: date) -> int:
    fy_start = date(d.year, FY_START_MONTH, FY_START_DAY)
    return d.year if d >= fy_start else d.year - 1


def _fy_boundaries(fy_end_year: int) -> Tuple[date, date]:
    start = date(fy_end_year - 1, FY_START_MONTH, FY_START_DAY)
    end = _add_months(start, 12) - timedelta(days=1)
    return start, end


def _this_quarter_fiscal(d: date) -> Tuple[date, date]:
    fy_end = _fiscal_year_for(d)
    fy_start, _ = _fy_boundaries(fy_end)
    delta_m = (d.year - fy_start.year)*12 + (d.month - fy_start.month)
    q = max(1, min(4, (delta_m // 3) + 1))
    start = _add_months(fy_start, (q-1)*3)
    end = _add_months(start, 3) - timedelta(days=1)
    return start, end


def _prev_quarter_fiscal(d: date) -> Tuple[date, date]:
    fy_end = _fiscal_year_for(d)
    fy_start, _ = _fy_boundaries(fy_end)
    delta_m = (d.year - fy_start.year)*12 + (d.month - fy_start.month)
    cq = (delta_m // 3) + 1
    if cq <= 1:
        return _add_months(fy_start, 9), _add_months(fy_start, 12)-timedelta(days=1)
    start = _add_months(fy_start, (cq-2)*3)
    end = _add_months(start, 3) - timedelta(days=1)
    return start, end


@dataclass
class TimeHint:
    operator: str
    at: Optional[str] = None
    _from: Optional[str] = None
    to: Optional[str] = None
    explain: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {"operator": self.operator}
        if self.at:
            d["at"] = self.at
        if self._from:
            d["from"] = self._from
        if self.to:
            d["to"] = self.to
        if self.explain:
            d["explain"] = self.explain
        return d


YEAR = r"(19\d{2}|20\d{2})"
DATERANGE_SEP = r"\s*(?:to|till|until|through|–|-|—)\s*"


def parse_time_hint(query: str) -> Dict[str, Any]:
    q = (query or "").strip()
    ql = q.lower()
    today = _tznow().date()

    if re.search(r"\b(as of|as at|currently|today|now)\b", ql):
        return TimeHint("AS_OF", at=today.isoformat(), explain="Point-in-time").as_dict()
    if re.search(r"\byesterday\b", ql):
        return TimeHint("AS_OF", at=(today-timedelta(days=1)).isoformat(), explain="Point-in-time").as_dict()

    if re.search(r"\b(ytd)\b", ql):
        return TimeHint("BETWEEN", _from=f"{today.year}-01-01", to=today.isoformat(), explain="YTD").as_dict()
    if re.search(r"\b(mtd)\b", ql):
        return TimeHint("BETWEEN", _from=f"{today.year}-{today.month:02d}-01", to=today.isoformat(), explain="MTD").as_dict()
    if re.search(r"\b(qtd)\b", ql):
        qs, qe = _this_quarter_fiscal(today)
        return TimeHint("BETWEEN", _from=qs.isoformat(), to=min(qe, today).isoformat(), explain="QTD (fiscal)").as_dict()
    if re.search(r"\b(fytd|fy ytd)\b", ql):
        fy = _fiscal_year_for(today)
        fs, fe = _fy_boundaries(fy)
        return TimeHint("BETWEEN", _from=fs.isoformat(), to=min(fe, today).isoformat(), explain="FYTD").as_dict()

    m = re.search(
        r"\b(last|past)\s+(\d{1,3})\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)\b", ql)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        if "day" in unit:
            start = today - timedelta(days=n)
        elif "week" in unit:
            start = today - timedelta(days=7*n)
        elif "month" in unit:
            start = _add_months(today, -n)
        elif "quarter" in unit:
            start = _add_months(today, -3*n)
        else:
            start = date(today.year - n, today.month, today.day)
        return TimeHint("BETWEEN", _from=start.isoformat(), to=today.isoformat(), explain=f"last {n} {unit}").as_dict()

    if re.search(r"\b(previous|last)\s+month\b", ql):
        first = date(today.year, today.month, 1)
        end = first - timedelta(days=1)
        start = date(end.year, end.month, 1)
        return TimeHint("BETWEEN", _from=start.isoformat(), to=end.isoformat(), explain="Previous month").as_dict()
    if re.search(r"\b(this)\s+month\b", ql):
        start = date(today.year, today.month, 1)
        end = _end_of_month(today.year, today.month)
        return TimeHint("BETWEEN", _from=start.isoformat(), to=end.isoformat(), explain="This month").as_dict()

    if re.search(r"\b(previous|last)\s+quarter\b", ql):
        s, e = _prev_quarter_fiscal(today)
        return TimeHint("BETWEEN", _from=s.isoformat(), to=e.isoformat(), explain="Previous fiscal quarter").as_dict()
    if re.search(r"\b(this)\s+quarter\b", ql):
        s, e = _this_quarter_fiscal(today)
        return TimeHint("BETWEEN", _from=s.isoformat(), to=e.isoformat(), explain="This fiscal quarter").as_dict()

    if re.search(r"\b(previous|last)\s+year\b", ql):
        y = today.year - 1
        return TimeHint("BETWEEN", _from=f"{y}-01-01", to=f"{y}-12-31", explain="Previous year").as_dict()
    if re.search(r"\b(this)\s+year\b", ql):
        y = today.year
        return TimeHint("BETWEEN", _from=f"{y}-01-01", to=f"{y}-12-31", explain="This year").as_dict()

    m = re.search(r"\bq([1-4])\s*(?:fy)?\s*(\d{2,4})\b", ql)
    if m:
        qn = int(m.group(1))
        yraw = m.group(2)
        if "fy" in ql:
            fy_end = int(yraw) if len(yraw) == 4 else 2000+int(yraw)
            s, e = _add_months(_fy_boundaries(fy_end)[
                               0], (qn-1)*3), _add_months(_fy_boundaries(fy_end)[0], (qn)*3)-timedelta(days=1)
        else:
            qstarts = [(1, 1), (4, 1), (7, 1), (10, 1)]
            sm, sd = qstarts[qn-1]
            y = int(yraw)
            s = date(y, sm, sd)
            e = _add_months(s, 3)-timedelta(days=1)
        return TimeHint("BETWEEN", _from=s.isoformat(), to=e.isoformat(), explain="Quarter").as_dict()

    m = re.search(r"\bfy\s*(\d{2,4})(?:\s*[-/]\s*(\d{2,4}))?\b", ql)
    if m:
        a, b = m.group(1), m.group(2)
        fy_end = (int(b) if b else int(a))
        fy_end = fy_end if len(
            str(fy_end)) == 4 else 2000 + int(str(fy_end)[-2:])
        fs, fe = _fy_boundaries(fy_end)
        return TimeHint("BETWEEN", _from=fs.isoformat(), to=fe.isoformat(), explain="Fiscal year").as_dict()

    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", ql) or re.search(
        r"\bfrom\s+(.+?)\s+(?:to|till|until|through)\s+(.+)$", ql)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        da, db = _to_date_safe(a), _to_date_safe(b)
        if da and db:
            if da > db:
                da, db = db, da
            return TimeHint("BETWEEN", _from=da.isoformat(), to=db.isoformat(), explain="Explicit range").as_dict()

    m = re.search(rf"\b({YEAR}){DATERANGE_SEP}({YEAR})\b", ql)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 > y2:
            y1, y2 = y2, y1
        return TimeHint("BETWEEN", _from=f"{y1}-01-01", to=f"{y2}-12-31", explain="Year range").as_dict()

    m = re.search(r"\b(before|prior to)\s+(.+)$", ql)
    if m:
        d = _to_date_safe(m.group(2).strip())
        if d:
            return TimeHint("BEFORE", to=d.isoformat(), explain="Directional: before").as_dict()
    m = re.search(r"\b(after|since|on or after)\s+(.+)$", ql)
    if m:
        d = _to_date_safe(m.group(2).strip())
        if d:
            return TimeHint("AFTER", _from=d.isoformat(), explain="Directional: after").as_dict()

    m = re.search(r"\b(in|during)\s+(19\d{2}|20\d{2})\b", ql)
    if m:
        y = int(m.group(2))
        return TimeHint("BETWEEN", _from=f"{y}-01-01", to=f"{y}-12-31", explain="Year scope").as_dict()

    m = re.search(r"\bon\s+(.+)$", ql)
    if m:
        d = _to_date_safe(m.group(1).strip())
        if d:
            return TimeHint("AS_OF", at=d.isoformat(), explain="Specific date").as_dict()

    return TimeHint("OPEN", explain="No explicit temporal intent").as_dict()


def _iou_days(a_start: date, a_end: date, b_start: date, b_end: date) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    inter = (right - left).days + 1
    if inter < 0:
        inter = 0
    union = (max(a_end, b_end) - min(a_start, b_start)).days + 1
    return max(0.0, inter/union)


def time_compat(valid_from: Optional[str], valid_to: Optional[str], hint: Dict[str, Any]) -> float:
    op = (hint or {}).get("operator", "OPEN")
    if op == "OPEN":
        return 1.0
    vf = _to_date_safe(valid_from)
    vt = _to_date_safe(valid_to)
    if vf and vt and vf > vt:
        vf, vt = vt, vf

    def smooth_decay(days: int, k: float = 0.018) -> float:
        return 1.0 / (1.0 + math.exp(k*days))

    if op == "AS_OF":
        at = _to_date_safe(hint.get("at"))
        if not at:
            return 0.6
        if vf is None and vt is None:
            return 0.65
        if vf and vt and vf <= at <= vt:
            return 1.0
        distances = []
        if vf:
            distances.append(abs((at-vf).days))
        if vt:
            distances.append(abs((at-vt).days))
        dmin = min(distances) if distances else 3650
        return max(0.0, smooth_decay(dmin))

    if op == "BETWEEN":
        f = _to_date_safe(hint.get("from"))
        t = _to_date_safe(hint.get("to"))
        if not (f and t):
            return 0.55
        if vf is None and vt is None:
            return 0.7
        a_start = vf or f
        a_end = vt or t
        return _iou_days(a_start, a_end, f, t)

    if op == "BEFORE":
        p = _to_date_safe(hint.get("to"))
        if not p:
            return 0.6
        if vf and vt and vt <= p:
            return 1.0
        if vf and vf > p:
            return 0.0
        base = 0.75
        if vt:
            d = (vt-p).days
            if d > 0:
                base *= smooth_decay(d)
        return max(0.0, min(1.0, base))

    if op == "AFTER":
        p = _to_date_safe(hint.get("from"))
        if not p:
            return 0.6
        if vf and vt and vf >= p:
            return 1.0
        if vt and vt < p:
            return 0.0
        base = 0.75
        if vf:
            d = (p-vf).days
            if d > 0:
                base *= smooth_decay(d)
        return max(0.0, min(1.0, base))

    return 1.0


def hard_pass(valid_from: Optional[str], valid_to: Optional[str], hint: Dict[str, Any]) -> bool:
    op = (hint or {}).get("operator", "OPEN")
    if op == "OPEN":
        return True
    vf = _to_date_safe(valid_from)
    vt = _to_date_safe(valid_to)
    if vf and vt and vf > vt:
        vf, vt = vt, vf

    if op == "AS_OF":
        at = _to_date_safe(hint.get("at"))
        if not at:
            return True
        return ((vf is None) or (vf <= at)) and ((vt is None) or (at <= vt))

    if op == "BETWEEN":
        f = _to_date_safe(hint.get("from"))
        t = _to_date_safe(hint.get("to"))
        if not (f and t):
            return True
        a = vf or f
        b = vt or t
        return not (b < f or a > t)

    if op == "BEFORE":
        p = _to_date_safe(hint.get("to"))
        if not p:
            return True
        if vf and vf > p:
            return False
        return True

    if op == "AFTER":
        p = _to_date_safe(hint.get("from"))
        if not p:
            return True
        if vt and vt < p:
            return False
        return True

    return True
