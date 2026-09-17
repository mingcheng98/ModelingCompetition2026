# -*- coding: utf-8 -*-
"""本地机器狗模拟器。

本文件不启动 HTTP 服务，也不监听任何网络端口。它在当前 Python 进程内
实现附件 1/2 中约定的 ``/enter``、``/measure``、``/clear``、``/exit``
四个动作，并把本地后端注入现有 Q3/Q4 策略的 ``SimClient``。

用法（在仓库根目录或本目录均可）：

    python Q34_机器狗/Simulator.py
    python Q34_机器狗/Q3_RobotDog.py
    python Q34_机器狗/Q4_RobotDog.py

第一条命令准备本地 Q3/Q4 会话，随后直接运行任一机器狗程序即可测试。
全程不监听网络端口，也不要求输入队伍编号。程序会自动生成一个内部编号，
仅用于保持协议字段完整和复现本次随机案例。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# 允许从仓库根目录直接执行 ``python Q34_机器狗/Simulator.py``。
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Q3_RobotDog import SimClient, StrategyQ3, scan_grid  # noqa: E402
from Q4_RobotDog import StrategyQ4, scan_grid_q4  # noqa: E402


ARENA_RADIUS = 1800.0
SPEED = 5.0
MEASURE_TIME = 5.0
CHANNEL_SWITCH_TIME = 1.0
CLEAR_SEARCH_TIME = 3.0
CLEAR_LASER_TIME = 2.0
NEAR_RADIUS = 5.0
CLEAR_RADIUS = 20.0
MAX_VIRTUAL_DURATION = 360000.0
MAX_REAL_DURATION = 1200.0
STATE_ROOT = Path(tempfile.gettempdir()) / "modeling_competition2026_simulator"


def session_path(question: int) -> Path:
    """返回当前 Q3/Q4 本地会话文件路径（位于系统临时目录）。"""
    if question not in (3, 4):
        raise ValueError("question 必须为 3 或 4")
    return STATE_ROOT / f"session_q{question}.json"


@dataclass
class InterferenceSource:
    """一局测试中的隐藏干扰源。"""

    channel: int
    x: float
    y: float
    receive_radius: float
    directional: bool = False
    direction_deg: float = 0.0
    cleared: bool = False

    @property
    def position(self) -> Tuple[float, float]:
        return self.x, self.y


def _angle_diff(a: float, b: float) -> float:
    """返回两个角度的最小夹角，范围 [0, 180]。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _stable_seed(robot_id: str, question: int) -> int:
    digest = hashlib.sha256(f"{question}:{robot_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _has_forbidden_chars(value: str) -> bool:
    return any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in value)


def _generate_sources(robot_id: str, question: int) -> List[InterferenceSource]:
    """按队伍编号稳定生成一局案例，且保证扫描网格至少能听到每个源。

    正式模拟器的案例是随机的；本地版本用队伍编号作种子，便于复现实验，
    但不把源真值返回给机器狗策略。
    """
    rng = random.Random(_stable_seed(robot_id, question))
    count = rng.randint(10, 16)
    channels = rng.sample(range(1, 21), count)
    grid = scan_grid() if question == 3 else scan_grid_q4()
    sources: List[InterferenceSource] = []

    for channel in channels:
        # 有效接收半径严格落在题设区间内。拒绝无法被相应覆盖网格
        # 接收的点，避免本地测试因随机案例产生不必要的“漏源”。
        for _ in range(1000):
            radius = rng.uniform(1050.0, 1450.0)
            polar_r = math.sqrt(rng.random()) * 1700.0
            polar_a = rng.uniform(0.0, 2.0 * math.pi)
            x, y = polar_r * math.cos(polar_a), polar_r * math.sin(polar_a)
            candidates = [p for p in grid if math.hypot(p[0] - x, p[1] - y) <= radius]
            if candidates:
                break
        else:  # 极端情况下仍给出目标区域内合法点
            x, y, radius, candidates = 0.0, 0.0, 1250.0, [grid[0]]

        directional = question == 4 and rng.random() < 0.45
        direction_deg = 0.0
        if directional:
            # 选择一个可被覆盖网格看见的方向；其余位置仍严格遵守 ±90°。
            visible_point = min(candidates, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            direction_deg = math.degrees(math.atan2(visible_point[1] - y, visible_point[0] - x)) % 360.0
        sources.append(InterferenceSource(channel, x, y, radius, directional, direction_deg))
    return sources


class LocalSimulator:
    """进程内协议后端，接口形状与 HTTP 业务响应一致。"""

    def __init__(self, robot_id: str, question: int, sources: Optional[Iterable[InterferenceSource]] = None):
        if (not robot_id or _has_forbidden_chars(robot_id)
                or not 1 <= len(robot_id.encode("utf-8")) <= 64):
            raise ValueError("robot_id 长度必须为 1 至 64 字节且不能包含控制字符")
        self.robot_id = robot_id
        self.question = question
        self.seed = _stable_seed(robot_id, question)
        self.sources = list(sources) if sources is not None else _generate_sources(robot_id, question)
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_time = 0.0
        self.entered = False
        self.finished = False
        self.last_error: Optional[str] = None
        self.requests: Dict[str, Tuple[str, str, dict]] = {}
        self.actions: List[dict] = []

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _response(self, **fields) -> dict:
        return {"accepted": True, "real_timestamp_ms": self._now_ms(),
                "virtual_time_s": round(self.virtual_time, 6), **fields}

    def _reject(self, reason: str) -> dict:
        # 附件协议规定 accepted=false 响应只含三个公共字段；具体原因
        # 保存在 ``last_error`` 供本地调试，不泄漏到机器狗接口。
        self.last_error = reason
        return {"accepted": False, "real_timestamp_ms": self._now_ms(),
                "virtual_time_s": 0}

    def _remember(self, request_id: str, path: str, payload: dict, response: dict) -> dict:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.requests[request_id] = (path, encoded, copy.deepcopy(response))
        return response

    def _cached(self, request_id: str, path: str, payload: dict) -> Optional[dict]:
        prior = self.requests.get(request_id)
        if prior is None:
            return None
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if prior[0] != path or prior[1] != encoded:
            return self._reject("request_id 已用于不同动作")
        return copy.deepcopy(prior[2])

    def _validate_base(self, payload: dict, required: set) -> Optional[str]:
        if not isinstance(payload, dict):
            return "请求体必须是 JSON 对象"
        allowed = {"arena_id", "robot_id", "request_id"} | required
        if set(payload) - allowed:
            return "请求包含未声明字段"
        if payload.get("arena_id") != "default":
            return "arena_id 不匹配"
        if payload.get("robot_id") != self.robot_id:
            return "robot_id 不匹配"
        request_id = payload.get("request_id")
        if (not isinstance(request_id, str) or not request_id
                or _has_forbidden_chars(request_id)
                or len(request_id.encode("utf-8")) > 128):
            return "request_id 不合法"
        return None

    @staticmethod
    def _position(payload: dict) -> Optional[Tuple[float, float]]:
        p = payload.get("position")
        if not isinstance(p, dict) or set(p) != {"x", "y"}:
            return None
        x, y = p.get("x"), p.get("y")
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)) or max(abs(x), abs(y)) > 2_000_000:
            return None
        return x, y

    @staticmethod
    def _channel(payload: dict) -> Optional[int]:
        value = payload.get("channel")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not number.is_integer() or not 1 <= int(number) <= 20:
            return None
        return int(number)

    def post(self, path: str, payload: dict) -> dict:
        """处理一个动作，返回与附件协议相同的业务字段。"""
        if not isinstance(payload, dict):
            return self._reject("请求体必须是 JSON 对象")
        request_id = payload.get("request_id")
        if isinstance(request_id, str):
            cached = self._cached(request_id, path, payload)
            if cached is not None:
                return cached

        if path == "/enter":
            response = self._enter(payload)
        elif path == "/measure":
            response = self._measure(payload)
        elif path == "/clear":
            response = self._clear(payload)
        elif path == "/exit":
            response = self._exit(payload)
        else:
            response = self._reject("路径必须是 /enter、/measure、/clear 或 /exit")

        # 只有 accepted=true 的业务动作占用 request_id，符合附件中的幂等约定。
        if response.get("accepted") and isinstance(request_id, str):
            self._remember(request_id, path, payload, response)
        self.actions.append({"path": path, "request": copy.deepcopy(payload), "response": copy.deepcopy(response)})
        return copy.deepcopy(response)

    def _enter(self, payload: dict) -> dict:
        error = self._validate_base(payload, set())
        if error:
            return self._reject(error)
        if self.entered or self.finished:
            return self._reject("测试已经进入或结束")
        self.entered = True
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_time = 0.0
        return self._response(max_virtual_duration_s=MAX_VIRTUAL_DURATION,
                              max_real_duration_s=MAX_REAL_DURATION,
                              remaining_real_duration_s=MAX_REAL_DURATION)

    def _active_error(self, payload: dict, required: set) -> Optional[str]:
        error = self._validate_base(payload, required)
        if error:
            return error
        if not self.entered:
            return "尚未成功调用 /enter"
        if self.finished:
            return "测试已经结束"
        return None

    def _advance(self, position: Tuple[float, float], extra: float) -> bool:
        distance = math.hypot(position[0] - self.position[0], position[1] - self.position[1])
        delta = distance / SPEED + extra
        if self.virtual_time + delta > MAX_VIRTUAL_DURATION + 1e-9:
            self.finished = True
            return False
        self.virtual_time += delta
        self.position = position
        return True

    def _source_signal(self, source: InterferenceSource, position: Tuple[float, float]) -> bool:
        distance = math.hypot(position[0] - source.x, position[1] - source.y)
        if distance > source.receive_radius:
            return False
        if not source.directional:
            return True
        detector_angle = math.degrees(math.atan2(position[1] - source.y, position[0] - source.x)) % 360.0
        return _angle_diff(detector_angle, source.direction_deg) <= 90.0 + 1e-12

    def _error_at(self, source: InterferenceSource, position: Tuple[float, float]) -> float:
        # 同一地点的误差固定，不同地点呈现统计变化；始终位于 [-1, 1]。
        key = f"{self.seed}:{source.channel}:{position[0]:.6f}:{position[1]:.6f}"
        value = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big")
        return value / float(2**64 - 1) * 2.0 - 1.0

    def _measure(self, payload: dict) -> dict:
        error = self._active_error(payload, {"position", "channel"})
        position = self._position(payload)
        channel = self._channel(payload)
        if error or position is None or channel is None:
            return self._reject(error or "position 或 channel 不合法")
        switch = CHANNEL_SWITCH_TIME if channel != self.channel else 0.0
        if not self._advance(position, switch + MEASURE_TIME):
            return self._reject("虚拟时间已达到上限")
        self.channel = channel
        source = next((s for s in self.sources if s.channel == channel and not s.cleared), None)
        if source is None or not self._source_signal(source, position):
            return self._response(measure_result="no_signal")
        distance = math.hypot(position[0] - source.x, position[1] - source.y)
        if distance <= NEAR_RADIUS + 1e-12:
            return self._response(measure_result="near")
        bearing = math.degrees(math.atan2(source.y - position[1], source.x - position[0])) % 360.0
        bearing = (bearing + self._error_at(source, position)) % 360.0
        return self._response(measure_result="direction", svd_deg=round(bearing, 2))

    def _clear(self, payload: dict) -> dict:
        error = self._active_error(payload, {"position", "channel"})
        position = self._position(payload)
        channel = self._channel(payload)
        if error or position is None or channel is None:
            return self._reject(error or "position 或 channel 不合法")
        source = next((s for s in self.sources if s.channel == channel and not s.cleared), None)
        in_range = source is not None and math.hypot(position[0] - source.x, position[1] - source.y) <= CLEAR_RADIUS + 1e-12
        total_extra = CLEAR_SEARCH_TIME + (CLEAR_LASER_TIME if in_range else 0.0)
        if not self._advance(position, total_extra):
            return self._reject("虚拟时间已达到上限")
        if not in_range:
            return self._response(clear_result="no_target_in_range")
        source.cleared = True
        return self._response(clear_result="success")

    def _exit(self, payload: dict) -> dict:
        error = self._active_error(payload, set())
        if error:
            return self._reject(error)
        self.finished = True
        return self._response(exit_reason="user_exit")

    def truth_summary(self) -> dict:
        """供本地结果日志使用的真值摘要，不会发给策略。"""
        return {
            "source_count": len(self.sources),
            "directional_count": sum(s.directional for s in self.sources),
            "channels": sorted(s.channel for s in self.sources),
            "sources": [
                {"channel": s.channel, "x": round(s.x, 3), "y": round(s.y, 3),
                 "receive_radius": round(s.receive_radius, 3),
                 "directional": s.directional, "direction_deg": round(s.direction_deg, 3),
                 "cleared": s.cleared}
                for s in sorted(self.sources, key=lambda item: item.channel)
            ],
        }

    def state_dict(self) -> dict:
        """将会话状态序列化，供跨进程的本地 Q3/Q4 程序继续执行。"""
        return {
            "version": 1,
            "robot_id": self.robot_id,
            "question": self.question,
            "seed": self.seed,
            "position": list(self.position),
            "channel": self.channel,
            "virtual_time": self.virtual_time,
            "entered": self.entered,
            "finished": self.finished,
            "last_error": self.last_error,
            "sources": [s.__dict__.copy() for s in self.sources],
            "requests": {key: [value[0], value[1], value[2]]
                         for key, value in self.requests.items()},
            "actions": self.actions,
        }

    @classmethod
    def from_state(cls, state: dict) -> "LocalSimulator":
        sources = [InterferenceSource(**item) for item in state.get("sources", [])]
        obj = cls(state["robot_id"], int(state["question"]), sources=sources)
        obj.seed = int(state.get("seed", obj.seed))
        obj.position = tuple(state.get("position", (0.0, 0.0)))
        obj.channel = int(state.get("channel", 1))
        obj.virtual_time = float(state.get("virtual_time", 0.0))
        obj.entered = bool(state.get("entered", False))
        obj.finished = bool(state.get("finished", False))
        obj.last_error = state.get("last_error")
        obj.requests = {key: (value[0], value[1], value[2])
                        for key, value in state.get("requests", {}).items()}
        obj.actions = state.get("actions", [])
        return obj


class FileLocalSimulator(LocalSimulator):
    """把 ``LocalSimulator`` 状态保存到临时 JSON，实现跨进程无端口通信。"""

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path).resolve()
        if not self.state_path.exists():
            raise FileNotFoundError(f"本地模拟会话不存在: {self.state_path}")
        self._load()

    def _load(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        loaded = LocalSimulator.from_state(state)
        self.__dict__.update(loaded.__dict__)
        self.state_path = Path(self.state_path).resolve()
        self.status = state.get("status", "ready")
        self.log_dir = state.get("log_dir", str(STATE_ROOT / "runs"))

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        state = self.state_dict()
        state.update({"status": "finished" if self.finished else "running",
                      "log_dir": self.log_dir})
        temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.state_path)

    def post(self, path: str, payload: dict) -> dict:
        response = super().post(path, payload)
        # 单次策略运行期间状态保留在内存；进入和退出时落盘即可，
        # 避免每个动作都重写不断增长的完整轨迹。
        if path in {"/enter", "/exit"} or not response.get("accepted"):
            self._save()
        return response

    def truth_summary(self) -> dict:
        return super().truth_summary()


def prepare_sessions(robot_id: str = "LOCAL-TEAM", question: str = "both") -> dict:
    """生成 Q3/Q4 本地会话文件，供随后单击运行机器狗程序。"""
    if question not in {"3", "4", "both"}:
        raise ValueError("question 必须为 3、4 或 both")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = STATE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    questions = [3, 4] if question == "both" else [int(question)]
    prepared = {}
    for q in questions:
        backend = LocalSimulator(robot_id, q)
        state = backend.state_dict()
        state.update({"status": "ready", "log_dir": str(run_dir)})
        path = session_path(q)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
        prepared[f"q{q}"] = {"session_path": str(path), "log_dir": str(run_dir),
                             "source_count": len(backend.sources)}
    return prepared


def _run_one(robot_id: str, question: int, run_dir: Path) -> dict:
    backend = LocalSimulator(robot_id, question)
    strategy_cls = StrategyQ3 if question == 3 else StrategyQ4
    client = SimClient(robot_id=robot_id, backend=backend)
    started = time.perf_counter()
    strategy = strategy_cls(client)
    summary = strategy.run()
    elapsed = time.perf_counter() - started
    result = dict(summary)
    result.update({"question": question, "robot_id": robot_id,
                   "program_run_time_s": round(elapsed, 6),
                   "simulator": "in_process", "network_port": None,
                   "actions": strategy.trace["actions"],
                   "protocol_actions": backend.actions,
                   "truth": backend.truth_summary()})
    safe_robot_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", robot_id).strip("._") or "team"
    out_path = run_dir / f"Q{question}_{safe_robot_id}_{int(time.time() * 1000)}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["log_path"] = str(out_path)
    return result


def run_simulation(robot_id: str, question: str = "both", log_dir: Optional[Path] = None) -> List[dict]:
    """运行 Q3、Q4 或指定问题，返回汇总结果列表。"""
    if not robot_id or not robot_id.strip():
        raise ValueError("必须提供队伍编号")
    if _has_forbidden_chars(robot_id):
        raise ValueError("队伍编号不能包含控制字符")
    if question not in {"3", "4", "both"}:
        raise ValueError("question 必须为 3、4 或 both")
    if log_dir is None:
        log_dir = Path(tempfile.gettempdir()) / "modeling_competition2026_simulator" / time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    questions = [3, 4] if question == "both" else [int(question)]
    return [_run_one(robot_id.strip(), q, log_dir) for q in questions]


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化无网络端口的 Q3/Q4 本地模拟会话")
    parser.add_argument("robot_id", nargs="?", default=None,
                        help="可选队伍编号；省略时自动生成，无需输入")
    parser.add_argument("--question", choices=("3", "4", "both"), default="both",
                        help="准备问题3、问题4或两者(默认: both)")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="兼容旧版一体化运行时的日志目录")
    parser.add_argument("--run", action="store_true",
                        help="兼容旧版：准备后立即运行策略；默认只准备会话")
    args = parser.parse_args()
    robot_id = args.robot_id or f"LOCAL-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}"
    try:
        if args.run:
            results = run_simulation(robot_id, args.question, args.log_dir)
            for result in results:
                total = result["truth"]["source_count"]
                print(f"Q{result['question']}  队伍 {result['robot_id']}: "
                      f"清除 {result['cleared']}/{total} 个，虚拟用时 {result['total_time']:.2f} s")
                print(f"日志(仓库外): {result['log_path']}")
            return 0
        prepared = prepare_sessions(robot_id, args.question)
    except Exception as exc:
        print(f"模拟失败: {exc}", file=sys.stderr)
        return 1
    print(f"本地模拟会话已准备，内部编号: {robot_id}")
    for name, info in prepared.items():
        print(f"{name.upper()} 可直接运行；日志目录(仓库外): {info['log_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
