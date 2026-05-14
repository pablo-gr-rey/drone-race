import dataclasses
import struct
import types
from typing import Any, Optional

import numpy as np
import zmq
from renderer import EnvironmentRenderer
from utils import EVENT_TYPE, MSG_TYPE, ControllerConfig, GateEnvironmentConfig, PIDConfig


class BytePacker:
    "Format: int32, f32, and an array of float is encoded as size (int) + raw array"

    def __init__(self):
        self.buf = bytearray()

    def pushInt(self, x: int) -> None:
        self.buf += struct.pack("<i", np.int32(x))  # little-endian int32

    def pushFloat(self, x: float) -> None:
        self.buf += struct.pack("<f", np.float32(x))  # little-endian float32

    def pushArray(self, arr: np.ndarray) -> None:
        "Push size then float array"
        a = np.asarray(arr, dtype=np.float32, order="C")
        # store length first (int32), then raw bytes
        self.pushInt(a.size)
        self.buf += a.tobytes(order="C")

    def pushObj(self, val: Any) -> bool:
        "Return False if the object (or part of it in case of tuple) could not be pushed"
        if isinstance(val, int):
            self.pushInt(val)
        elif isinstance(val, bool):
            self.pushInt(int(val))
        elif isinstance(val, float):
            self.pushFloat(val)
        elif isinstance(val, np.ndarray):
            self.pushArray(val)
        elif isinstance(val, tuple):
            for v in val:
                if not self.pushObj(v):
                    return False
        elif val is None or isinstance(val, types.FunctionType):
            pass
        else:
            return False

        return True

    def toBytes(self) -> bytes:
        return bytes(self.buf)


class ByteUnpacker:
    def __init__(self, data: bytes):
        self.buf = memoryview(data)
        self.offset = 0

    def read(self, n: int) -> memoryview:
        if self.offset + n > len(self.buf):
            raise ValueError(f"Buffer underflow: need {n} bytes, have {len(self.buf) - self.offset}")
        chunk = self.buf[self.offset : self.offset + n]
        self.offset += n
        return chunk

    def readInt(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def readFloat(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def readArray(self, copy: bool = True) -> np.ndarray:
        n = self.readInt()
        if n < 0:
            raise ValueError(f"Negative array length: {n}")
        byte_count = n * 4
        raw = self.read(byte_count)

        arr = np.frombuffer(raw, dtype=np.float32, count=n)
        return arr.copy() if copy else arr

    def assert_finished(self):
        if self.offset != len(self.buf):
            raise ValueError(f"{len(self.buf) - self.offset} trailing bytes left")


def encodeConfig(config: Any, msg_type: Optional[int] = None, warn=True, log=False, p: Optional[BytePacker] = None) -> BytePacker:
    if p is None:
        p = BytePacker()

    if msg_type is not None:
        p.pushInt(msg_type)
    if log:
        print(f"Encoding {type(config)}...")

    for field in dataclasses.fields(config):
        val = getattr(config, field.name)
        # avoid numeric issues: it's important to send the correct type! (ie. trackWidth=2 instead of 2.0 is wrongly sent as int and reinterpreted as messy float)
        if field.type is float:
            val = float(val)
        elif field.type is int:
            val = int(val)

        if log:
            print(f"Packing field {field.name} of type {type(val)}, value {val if not isinstance(val, np.ndarray) else val}")

        if field.name == "init_state":
            # legacy field: skip, explicit init_pos/init_vel are used
            continue

        if field.name == "init_pos":
            assert isinstance(val, np.ndarray)
            p.pushArray(val)
            continue

        if field.name == "init_vel":
            assert isinstance(val, np.ndarray)
            p.pushArray(val)
            continue

        if isinstance(val, ControllerConfig):
            encodeConfig(val, None, warn, log, p)
        elif not p.pushObj(val) and warn:
            print(f"Cannot pack field {field.name} of type {type(val)} value {val} ")

    return p


def unpackState(unpack: ByteUnpacker) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    "Return (step, pos, vel, currentS, nLaps, currentGates) from bytes"

    step = unpack.readInt()
    pos = unpack.readArray()
    vel = unpack.readArray()
    currentS = unpack.readArray()
    nLaps = unpack.readArray()
    currentGates = unpack.readArray()

    unpack.assert_finished()

    return step, pos, vel, currentS, nLaps, currentGates


class ZMQRecv:
    def __init__(self, addr: str = "tcp://localhost:5555"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PAIR)
        self.sock.connect(addr)

        self.posLog: list[np.ndarray] = []
        self.velLog: list[np.ndarray] = []

        self.sLog: list[np.ndarray] = []
        self.nLapsLog: list[np.ndarray] = []
        self.currentGatesLog: list[np.ndarray] = []

    def runSim(
        self,
        config: GateEnvironmentConfig,
        contConfigs: list[ControllerConfig],
        render: bool = True,
        contNames: Optional[list[str]] = None,
    ) -> tuple[EVENT_TYPE, int]:
        if render:
            # only display the racelines which are actually used
            used = [False] * config.nRaceLines
            for cfg in contConfigs:
                if isinstance(cfg, PIDConfig) and cfg.racelineIndex >= 0:
                    used[cfg.racelineIndex] = True

            if contNames is None:
                contNames = [cfg.getDefaultName() for cfg in contConfigs]

            renderer = EnvironmentRenderer(
                config,
                contNames,
                interval=0,
                frameSkipWaiting=5,
                frameSkipPlayback=2,
                defaultZoomAgent=-1,
                display_raceline=used,
            )
        else:
            renderer = None

        # send header
        print("Sending header...")

        header = encodeConfig(config, MSG_TYPE.MSG_HEADER).toBytes()
        self.sock.send(header)

        for cont in contConfigs:
            self.sock.send(encodeConfig(cont, MSG_TYPE.MSG_HEADER).toBytes())

        print("header sent OK, waiting for first state...")

        result = None

        while True:
            unpack = ByteUnpacker(self.sock.recv())
            msg_type = unpack.readInt()

            if msg_type == MSG_TYPE.MSG_HEADER:
                print("Received unexpected header message from C++")

            elif msg_type == MSG_TYPE.MSG_STATE:
                step, pos, vel, newS, newLaps, newGates = unpackState(unpack)

                # print(f"received step {step}")

                if step != len(self.posLog):
                    print(f"expected step number {len(self.posLog)} but received step {step}")

                if step == 0:
                    # we confirm that the first state is equal to the initial state we sent (to detect early potential transmission bugs)
                    if (
                        not np.all(np.isclose(pos, config.init_pos))
                        or not np.all(np.isclose(vel, config.init_vel))
                        or not np.all(np.isclose(newS, config.initS))
                        or not np.all(np.isclose(newLaps, config.initnLaps))
                        or not np.all(np.isclose(newGates, config.initGates))
                    ):
                        print(pos, vel, newS, newLaps, newGates)
                        print(config.init_pos, config.init_vel, config.initS, config.initnLaps, config.initGates)
                        raise ValueError("First state sent back by C++ backend did not match expected first state")

                self.posLog.append(pos)
                self.velLog.append(vel)
                self.sLog.append(newS)
                self.nLapsLog.append(newLaps)
                self.currentGatesLog.append(newGates)

                if renderer is not None:
                    renderer.onNewState(pos, vel, newS, newLaps, newGates)

            elif msg_type == MSG_TYPE.MSG_EVENT:
                evt_type, agent_id = unpack.readInt(), unpack.readInt()
                unpack.assert_finished()

                ename = EVENT_TYPE(evt_type)
                print(f"Event: {ename.name} " + f"agent={agent_id + 1}" if agent_id != -1 else "")

                if renderer is not None:
                    if ename == EVENT_TYPE.EVT_COLLISION:
                        renderer.collision = True
                    elif ename == EVENT_TYPE.EVT_OUTSIDE:
                        renderer.outside = agent_id
                    elif ename == EVENT_TYPE.EVT_WINNER:
                        renderer.winner = agent_id
                # nothing to do if truncation

                result = (ename, agent_id)

            elif msg_type == MSG_TYPE.MSG_DONE:
                unpack.assert_finished()
                print(f"Simulation done. Total steps: {len(self.posLog)}")
                if renderer is not None:
                    renderer.finish(tooglePlay=False, jumpToLast=False)

                break

            else:
                print("unknown msg type", msg_type)

        if result is None:
            raise ValueError("C++ finished without sending event")

        return result

    def close(self):
        self.sock.close()
        self.ctx.term()
