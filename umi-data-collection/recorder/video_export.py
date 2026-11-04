"""Create a viewing MP4 after capture, preserving variable camera frame timing."""
import csv
from fractions import Fraction
import io


def check_video_support():
    try:
        import av
        av.Codec('libx264', 'w')
    except (ImportError, ValueError) as error:
        raise ValueError('Video tools are missing. Run the updated Pi installer before recording.') from error


def export_video(directory, progress, closing):
    import av
    from PIL import Image
    partial = directory/'video.partial.mp4'
    try:
        with (directory/'frames.csv').open() as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        if not count:
            raise ValueError('No camera frames were captured')
        with av.open(str(partial), 'w', format='mp4', options={'movflags': '+faststart'}) as output, \
                (directory/'video.mjpg').open('rb') as images, (directory/'frames.csv').open() as index:
            encoder = None
            start = previous = None
            for number, row in enumerate(csv.DictReader(index), 1):
                if closing.is_set():
                    raise InterruptedError('Dashboard stopped during video conversion')
                timestamp = int(row['camera_timestamp_ns'])
                if encoder is None:
                    start = timestamp
                    encoder = output.add_stream('libx264', rate=20)
                    encoder.width, encoder.height = int(row['width']), int(row['height'])
                    encoder.pix_fmt = 'yuv420p'
                    encoder.time_base = encoder.codec_context.time_base = Fraction(1, 1_000_000)
                    encoder.codec_context.thread_count = 2
                    encoder.options = {'preset': 'ultrafast', 'crf': '20', 'tune': 'zerolatency'}
                if previous is not None and timestamp <= previous:
                    raise ValueError('Camera timestamps are not increasing')
                previous = timestamp
                images.seek(int(row['offset']))
                with Image.open(io.BytesIO(images.read(int(row['size'])))) as image:
                    frame = av.VideoFrame.from_image(image.convert('RGB'))
                frame.pts = round((timestamp-start)/1000)
                frame.time_base = Fraction(1, 1_000_000)
                for packet in encoder.encode(frame):
                    output.mux(packet)
                progress(number/count)
            for packet in encoder.encode():
                output.mux(packet)
        partial.replace(directory/'video.mp4')
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
