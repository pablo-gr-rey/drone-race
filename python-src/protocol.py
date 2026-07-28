import dataclasses
import struct
import types
from typing import Any, Optional
import typing

import numpy as np
import zmq
from renderer import EnvironmentRenderer
from utils import EVENT_TYPE, MSG_TYPE, BaseEnvironmentConfig, BaseSimState, BaseControllerConfig, FullStateInfo


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
        if isinstance(val, int):  # also handles bool and enums
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
                if not field.metadata.get("send", True):
                    continue

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
            if "recv" in field.metadata and not field.metadata["recv"]:
                continue

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

            elif isinstance(field.type, typing.TypeVar) or typing.get_origin(field.type) is typing.Union:
                if "class" not in field.metadata:
                    raise ValueError(f"Missing source class information in metadata for field {field.name} of type {field.type}")
                if not hasattr(cl, field.metadata["class"]):
                    raise ValueError(
                        f"Class source metadata for field {field.name} is declared as {field.metadata['class']} but it was not found in attributes of config class {cl}"
                    )

                subcl = getattr(cl, field.metadata["class"])
                attrs[field.name] = self.readConfig(subcl)

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

    if not p.pushObj(config, warn, log):
        print(f"Warning: failed to write config of type {type(config)}")

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
        envConfig: BaseEnvironmentConfig,
        contConfig: BaseControllerConfig,
        initState: BaseSimState,
        render: bool = True,
        oppNames: Optional[list[list[str]]] = None,
        **kwargs: Any,
    ) -> EVENT_TYPE:
        if render:
            renderer = EnvironmentRenderer(
                envConfig,
                contConfig,
                oppNames,
                interval=0,
                frameSkipWaiting=5,
                frameSkipPlayback=2,
                defaultZoomAgent=-1,
                **kwargs,
            )
        else:
            renderer = None

        # update controller type in FullStateInfo
        FullStateInfo.updateTypes(envConfig, contConfig)

        # send header
        print("Sending header...")

        self.sock.send(encodeConfig(envConfig, MSG_TYPE.MSG_HEADER).toBytes())

        self.sock.send(encodeConfig(contConfig, MSG_TYPE.MSG_HEADER).toBytes())

        self.sock.send(encodeConfig(initState, MSG_TYPE.MSG_HEADER).toBytes())
        print("header sent OK, waiting for first state...")

        result: Optional[EVENT_TYPE] = None

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

                self.stateLog.append(state)

                if renderer is not None:
                    # find out if there are waiting states (ie. if they are computed faster than rendered)
                    events = self.sock.getsockopt(zmq.EVENTS)
                    hasPending = events & zmq.POLLIN  # type: ignore
                    renderer.onNewState(state, pendingState=bool(hasPending))

            elif msg_type == MSG_TYPE.MSG_EVENT:
                evt_type = unpack.readInt()
                unpack.assert_finished()

                ename = EVENT_TYPE(evt_type)
                print(f"Event: {ename.name}")

                if renderer is not None:
                    if ename == EVENT_TYPE.EVT_OUTSIDE:
                        renderer.outside = True
                    elif ename == EVENT_TYPE.EVT_WINNER:
                        renderer.winner = True
                    elif ename == EVENT_TYPE.EVT_OPP_WINNER:
                        renderer.opp_winner = True
                # nothing to do if truncation

                result = ename

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
