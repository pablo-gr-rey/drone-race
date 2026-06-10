import dataclasses
import struct
import types
from typing import Any, Optional
import typing

import numpy as np
import zmq
from renderer import EnvironmentRenderer
from utils import EVENT_TYPE, MSG_TYPE, ControllerConfig, FullStateInfo, GateEnvironmentConfig, PIDConfig, MPPIConfig, VerifConfig


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

    def pushObj(self, val: Any, warn: bool = True, log: bool = False) -> bool:
        "Return False if the object (or part of it in case of tuple) could not be pushed"
        if isinstance(val, int):
            self.pushInt(val)
        elif isinstance(val, bool):
            self.pushInt(int(val))
        elif isinstance(val, float):
            self.pushFloat(val)
        elif isinstance(val, np.ndarray):
            self.pushArray(val)
        elif isinstance(val, tuple) or isinstance(val, list):
            for v in val:
                if not self.pushObj(v):
                    return False
        elif dataclasses.is_dataclass(val) and not isinstance(val, type):
            for field in dataclasses.fields(val):
                n_val = getattr(val, field.name)
                # avoid numeric issues: it's important to send the correct type! (ie. trackWidth=2 instead of 2.0 is wrongly sent as int and reinterpreted as messy float)
                if field.type is float:
                    n_val = float(n_val)
                elif field.type is int:
                    n_val = int(n_val)

                if log:
                    print(
                        f"Packing field {field.name} of type {type(n_val)}, n_value {n_val if not isinstance(n_val, np.ndarray) else n_val}"
                    )

                if not self.pushObj(n_val):
                    return False
        elif val is None or isinstance(val, types.FunctionType):
            pass
        else:
            print(f"Failed to pack object of type {type(val)} value {val} ")
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

    def readAny[T](self, t: type[T]) -> T:
        if t is float:
            return self.readFloat()  # type: ignore
        elif issubclass(t, int):  # also handles bool and enums
            return t(self.readInt())
        elif t is np.ndarray:
            return self.readArray()  # type: ignore
        elif dataclasses.is_dataclass(t):
            return self.readConfig(t)

        raise ValueError(f"Unsupported reading type {t}")

    def is_finished(self):
        return self.offset == len(self.buf)

    def readConfig[ConfigClass](self, cl: type[ConfigClass]) -> ConfigClass:
        attrs: dict[str, Any] = {}
        if not dataclasses.is_dataclass(cl):
            raise ValueError("Can only unpack dataclass type in readConfig")

        for field in dataclasses.fields(cl):
            # print(f"Reading field {field.name}...", end="", flush=True)
            if typing.get_origin(field.type) is list:
                if "len" not in field.metadata:
                    raise ValueError(f"Missing len information in metadata for field {field.name} of type list")
                if field.metadata["len"] not in attrs:
                    raise ValueError(
                        f"Len metadata for field {field.name} is declared as  {field.metadata['len']} but it was not found in already defined class attributes"
                    )

                length = int(attrs[field.metadata["len"]])
                elem_type = typing.get_args(field.type)[0]

                attrs[field.name] = [self.readAny(elem_type) for i in range(length)]

            else:
                attrs[field.name] = self.readAny(field.type)  # type: ignore

            # print(f" value {attrs[field.name]}")

        return cl(**attrs)

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

    p.pushObj(config, warn, log)

    return p


def unpackState(
    unpack: ByteUnpacker,
) -> FullStateInfo:
    stateInfo = unpack.readConfig(FullStateInfo)
    unpack.assert_finished()

    return stateInfo


class ZMQRecv:
    def __init__(self, addr: str = "tcp://localhost:5555"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PAIR)
        self.sock.connect(addr)

        self.stateLog: list[FullStateInfo] = []

    def runSim(
        self,
        config: GateEnvironmentConfig,
        verifConfig: VerifConfig,
        contConfigs: list[ControllerConfig],
        render: bool = True,
        contNames: Optional[list[str]] = None,
        oppNames: Optional[list[str] | list[list[str]]] = None,
    ) -> tuple[EVENT_TYPE, int]:
        if render:
            # only display the racelines which are actually used
            used = [False] * config.nRaceLines
            nMppi = 0
            for cfg in contConfigs:
                if isinstance(cfg, PIDConfig):
                    used[cfg.racelineIndex] = True
                elif isinstance(cfg, MPPIConfig):
                    nMppi += 1
                    for opp in cfg.opponentPidConfigs:
                        used[opp.racelineIndex] = True

            mppiConfig = None

            for cont in contConfigs:
                if isinstance(cont, MPPIConfig):
                    mppiConfig = cont

            if mppiConfig is None:
                print("Warning: did not find any MPPIConfig, proceeding with default")
                mppiConfig = MPPIConfig()

            if contNames is None:
                contNames = [cfg.getDefaultName() for cfg in contConfigs]

            completedOppNames: list[list[str]] = []

            if oppNames is None:
                for cont in contConfigs:
                    if isinstance(cont, MPPIConfig):
                        completedOppNames.append([f"Model {i}" for i in range(cont.nModels)])
            elif oppNames and isinstance(oppNames[0], str):
                completedOppNames = [oppNames for i in range(nMppi)]  # type: ignore
            else:
                completedOppNames = oppNames  # type: ignore

            renderer = EnvironmentRenderer(
                config,
                contConfigs,
                contNames,
                verifConfig,
                completedOppNames,
                interval=0,
                frameSkipWaiting=2,
                frameSkipPlayback=2,
                defaultZoomAgent=-1,
                display_raceline=used,
                renderTrails=False,
            )
        else:
            renderer = None

        # send header
        print("Sending header...")

        header = encodeConfig(config, MSG_TYPE.MSG_HEADER).toBytes()
        self.sock.send(header)

        self.sock.send(encodeConfig(verifConfig, MSG_TYPE.MSG_HEADER, log=True).toBytes())

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
                state = unpackState(unpack)
                # step, pos, vel, newS, newLaps, newGates, belief = unpackState(unpack)

                # print(f"received step {state.step}")

                if state.step != len(self.stateLog):
                    print(f"expected step number {len(self.stateLog)} but received step {state.step}")

                if state.step == 0:
                    # we confirm that the first state is equal to the initial state we sent (to detect early potential transmission bugs)
                    if (
                        not np.all(np.isclose(state.pos, config.init_pos))
                        or not np.all(np.isclose(state.speed, config.init_vel))
                        # or not np.all(np.isclose(state.currentS, config.initS))     # for non-PID agents, engine sets S to -1
                        or not np.all(np.isclose(state.nLaps, config.initnLaps))
                        or not np.all(np.isclose(state.currentGates, config.initGates))
                    ):
                        print(state.pos, state.speed, state.currentS, state.nLaps, state.currentGates)
                        print(config.init_pos, config.init_vel, config.initS, config.initnLaps, config.initGates)
                        raise ValueError("First state sent back by C++ backend did not match expected first state")

                self.stateLog.append(state)

                if renderer is not None:
                    # find out if there are waiting states (ie. if they are computed faster than rendered)
                    events = self.sock.getsockopt(zmq.EVENTS)
                    hasPending = events & zmq.POLLIN  # type: ignore
                    renderer.onNewState(state, pendingState=bool(hasPending))

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
                print(f"Simulation done. Total steps: {len(self.stateLog)}")
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
