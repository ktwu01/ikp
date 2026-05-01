"""Live parallel-progress display for the IKP CLI.

Reserves one line per model in caller-supplied (preset/tier) order. Each
line starts as a spinner with elapsed time and is rewritten in place with
the final result the moment that model finishes. No re-ordering, no
head-of-line blocking. Falls back to plain sequential output when stdout
is not a TTY.
"""

import math
import re
import shutil
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

DIM = "\033[90m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_width(s: str) -> int:
    """Number of terminal columns occupied by s, ignoring ANSI escapes."""
    s = ANSI_RE.sub("", s)
    w = 0
    for ch in s:
        if ch == "\n":
            continue
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _wrapped_rows(s: str, term_w: int) -> int:
    """How many visual rows the string occupies when the terminal wraps at term_w."""
    cols = _visible_width(s)
    if cols == 0:
        return 1
    return max(1, math.ceil(cols / term_w))


def run_with_progress(models: list, work_fn, *, format_done,
                      default_phase: str = "querying") -> list:
    """Run work_fn for each model concurrently with live in-place progress.

    work_fn(model, set_status) -> result. set_status(text) updates that
    model's phase label.

    format_done(model, result, error, elapsed) -> str. Called exactly once
    when a model finishes; the returned string is the full line content
    (after the leading two-space indent, no trailing newline) that
    permanently replaces the spinner row for that model.

    Returns a list aligned with `models`: {model, result, error, elapsed}.
    """
    n = len(models)
    states = [{"phase": default_phase, "status": "pending", "elapsed": 0.0,
               "result": None, "error": None, "final_line": None}
              for _ in range(n)]
    starts = [None] * n
    lock = threading.Lock()
    done_event = threading.Event()

    name_w = max(len(m["name"]) for m in models)
    tier_w = max(len(f"[{m['tier']}]") for m in models)
    is_tty = sys.stdout.isatty()

    def render_row(i: int, frame: str) -> str:
        s = states[i]
        if s["final_line"] is not None:
            return "  " + s["final_line"]
        m = models[i]
        tag = f"[{m['tier']}]"
        marker = f"{DIM}{frame}{RESET}"
        msg = f"{DIM}{s['phase']}... {s['elapsed']:.1f}s{RESET}"
        return f"  {marker} {m['name']:<{name_w}} {DIM}{tag:<{tier_w}}{RESET} {msg}"

    def term_width() -> int:
        return max(20, shutil.get_terminal_size((100, 24)).columns)

    # Total visual rows occupied by the block at the previous draw.
    drawn_rows = {"n": 0}

    def initial_draw():
        frame = SPINNER[0]
        w = term_width()
        total = 0
        for i in range(n):
            line = render_row(i, frame)
            sys.stdout.write(line + "\n")
            total += _wrapped_rows(line, w)
        drawn_rows["n"] = total
        sys.stdout.flush()

    def repaint():
        frame = SPINNER[int(time.time() * 10) % len(SPINNER)]
        w = term_width()
        if drawn_rows["n"] > 0:
            sys.stdout.write(f"\033[{drawn_rows['n']}A")
        sys.stdout.write("\r\033[J")  # clear from cursor to end of screen
        total = 0
        for i in range(n):
            line = render_row(i, frame)
            sys.stdout.write(line + "\n")
            total += _wrapped_rows(line, w)
        drawn_rows["n"] = total
        sys.stdout.flush()

    def redraw_loop():
        while not done_event.is_set():
            time.sleep(0.1)
            with lock:
                now = time.time()
                for i, s in enumerate(states):
                    if s["status"] == "pending" and starts[i] is not None:
                        s["elapsed"] = now - starts[i]
                repaint()

    def set_status_for(i: int, phase: str):
        with lock:
            states[i]["phase"] = phase

    def finalize(i: int):
        s = states[i]
        if s["final_line"] is None:
            s["final_line"] = format_done(models[i], s["result"], s["error"], s["elapsed"])

    def runner(i: int, m: dict):
        with lock:
            starts[i] = time.time()
        try:
            result = work_fn(m, lambda t, ii=i: set_status_for(ii, t))
            with lock:
                states[i]["status"] = "done"
                states[i]["elapsed"] = time.time() - starts[i]
                states[i]["result"] = result
                finalize(i)
        except Exception as e:
            with lock:
                states[i]["status"] = "error"
                states[i]["elapsed"] = time.time() - starts[i]
                states[i]["error"] = str(e)
                finalize(i)

    if is_tty:
        with lock:
            initial_draw()
        drawer = threading.Thread(target=redraw_loop, daemon=True)
        drawer.start()

    with ThreadPoolExecutor(max_workers=max(1, n)) as ex:
        futures = [ex.submit(runner, i, m) for i, m in enumerate(models)]
        for f in futures:
            f.result()

    done_event.set()

    with lock:
        if is_tty:
            repaint()
        else:
            for i in range(n):
                sys.stdout.write("  " + states[i]["final_line"] + "\n")
            sys.stdout.flush()

    return [
        {"model": m, "result": s["result"], "error": s["error"], "elapsed": s["elapsed"]}
        for m, s in zip(models, states)
    ]
