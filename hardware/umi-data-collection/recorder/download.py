"""Stream a recording archive without duplicating a long take on the SD card."""
import io
import tarfile


class RecordingDownload:
    def __init__(self, paths):
        self.members = []
        for path in paths:
            # Metadata can change while the MP4 is encoding. Freeze its contents.
            contents = path.read_bytes() if path.name == 'metadata.json' else None
            info = tarfile.TarInfo(path.name)
            info.size = len(contents) if contents is not None else path.stat().st_size
            info.mode = 0o600
            self.members.append((info, path, contents))
        size = sum(len(info.tobuf(format=tarfile.PAX_FORMAT)) + (info.size+511)//512*512
                   for info, _, _ in self.members) + 1024
        self.size = (size+tarfile.RECORDSIZE-1)//tarfile.RECORDSIZE*tarfile.RECORDSIZE

    def write(self, destination):
        with tarfile.open(fileobj=destination, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for info, path, contents in self.members:
                with io.BytesIO(contents) if contents is not None else path.open('rb') as source:
                    archive.addfile(info, source)
