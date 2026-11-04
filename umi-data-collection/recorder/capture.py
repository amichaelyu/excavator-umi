"""Decode the capture helper's framed stream without losing V4L2 metadata."""
import struct

HEADER = struct.Struct('<4sIIIQQII')
MONOTONIC = 0x2000
TIMESTAMP_MASK = 0xE000
ERROR = 0x40


class TimedFrames:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk):
        self.buffer.extend(chunk)
        result = []
        while len(self.buffer) >= HEADER.size:
            magic, size, sequence, flags, timestamp, dequeued, width, height = HEADER.unpack_from(self.buffer)
            if magic != b'UMI1' or not 0 < size <= 8*1024*1024 or not (0 < width <= 8192 and 0 < height <= 8192):
                raise ValueError('Invalid timestamped camera packet')
            end = HEADER.size + size
            if len(self.buffer) < end:
                break
            frame = bytes(self.buffer[HEADER.size:end])
            del self.buffer[:end]
            result.append((frame, {
                'sequence': sequence, 'flags': flags, 'camera_timestamp_ns': timestamp,
                'dequeue_monotonic_ns': dequeued, 'width': width, 'height': height,
                'clock': 'CLOCK_MONOTONIC' if flags & TIMESTAMP_MASK == MONOTONIC else 'unknown',
                'timestamp_source': 'start_of_exposure' if flags & 0x70000 == 0x10000 else 'end_of_frame',
                'buffer_error': bool(flags & ERROR),
            }))
        return result
