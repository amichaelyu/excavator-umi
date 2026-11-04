/* UVC MJPEG capture with the driver's timestamp, sequence and flags intact.
 * stdout: little-endian <4sIIIQQII> header followed by one JPEG.
 * No timestamp is inferred from frame number or pipe arrival time.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stopping = 0;
static void stop(int sig) { (void)sig; stopping = 1; }
static int control(int fd, unsigned long request, void *arg) {
    int result;
    do { result = ioctl(fd, request, arg); } while (result < 0 && errno == EINTR && !stopping);
    return result;
}
static int write_all(const void *data, size_t length) {
    const unsigned char *cursor = data;
    while (length && !stopping) {
        ssize_t n = write(STDOUT_FILENO, cursor, length);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        cursor += n; length -= (size_t)n;
    }
    return length ? -1 : 0;
}
static void u32(unsigned char *out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out[i] = (unsigned char)(value >> (8*i));
}
static void u64(unsigned char *out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out[i] = (unsigned char)(value >> (8*i));
}

int main(int argc, char **argv) {
    int fd = -1, streaming = 0, result = 1;
    void *maps[4] = {0}; size_t lengths[4] = {0};
    struct sigaction action = {0}; action.sa_handler = stop;
    sigaction(SIGINT, &action, NULL); sigaction(SIGTERM, &action, NULL);
    signal(SIGPIPE, SIG_IGN);
    if (argc != 2) { fprintf(stderr, "Usage: capture /dev/video0\n"); return 2; }
    fd = open(argv[1], O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) { perror("open camera"); goto done; }
    struct v4l2_capability cap = {0};
    if (control(fd, VIDIOC_QUERYCAP, &cap) < 0) { perror("camera capabilities"); goto done; }
    uint32_t capabilities = cap.capabilities & V4L2_CAP_DEVICE_CAPS ? cap.device_caps : cap.capabilities;
    if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "Camera must support single-plane streaming capture\n"); goto done;
    }
    struct v4l2_format format = {0}; format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = 1280; format.fmt.pix.height = 720;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; format.fmt.pix.field = V4L2_FIELD_ANY;
    if (control(fd, VIDIOC_S_FMT, &format) < 0) { perror("camera format"); goto done; }
    if (format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
        fprintf(stderr, "Camera did not accept MJPEG\n"); goto done;
    }
    struct v4l2_streamparm parm = {0}; parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1; parm.parm.capture.timeperframe.denominator = 60;
    if (control(fd, VIDIOC_S_PARM, &parm) < 0) { perror("camera frame rate"); goto done; }
    fprintf(stderr, "Camera: %.32s; bus %.32s; %ux%u; %u/%u seconds per frame\n",
        cap.card, cap.bus_info, format.fmt.pix.width, format.fmt.pix.height,
        parm.parm.capture.timeperframe.numerator, parm.parm.capture.timeperframe.denominator);
    struct v4l2_requestbuffers request = {0};
    request.count = 4; request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; request.memory = V4L2_MEMORY_MMAP;
    if (control(fd, VIDIOC_REQBUFS, &request) < 0) { perror("camera buffers"); goto done; }
    if (request.count < 2 || request.count > 4) { fprintf(stderr, "Unexpected camera buffer count\n"); goto done; }
    for (uint32_t i = 0; i < request.count; ++i) {
        struct v4l2_buffer buffer = {0};
        buffer.index = i; buffer.type = request.type; buffer.memory = request.memory;
        if (control(fd, VIDIOC_QUERYBUF, &buffer) < 0) { perror("query buffer"); goto done; }
        lengths[i] = buffer.length;
        maps[i] = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buffer.m.offset);
        if (maps[i] == MAP_FAILED) { maps[i] = NULL; perror("map camera buffer"); goto done; }
        if (control(fd, VIDIOC_QBUF, &buffer) < 0) { perror("queue buffer"); goto done; }
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (control(fd, VIDIOC_STREAMON, &type) < 0) { perror("start camera"); goto done; }
    streaming = 1;
    while (!stopping) {
        struct pollfd pollfd = {.fd = fd, .events = POLLIN};
        int ready = poll(&pollfd, 1, 2000);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0 || (pollfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            fprintf(stderr, "Camera stopped supplying frames\n"); goto done;
        }
        struct v4l2_buffer buffer = {0}; buffer.type = type; buffer.memory = V4L2_MEMORY_MMAP;
        if (control(fd, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) continue;
            perror("dequeue camera buffer"); goto done;
        }
        struct timespec now;
        if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) { perror("monotonic clock"); goto done; }
        if (buffer.index >= request.count || buffer.bytesused > lengths[buffer.index] || buffer.bytesused > 8*1024*1024) {
            fprintf(stderr, "Invalid camera buffer length\n"); goto done;
        }
        unsigned char header[40] = {'U', 'M', 'I', '1'};
        u32(header+4, buffer.bytesused); u32(header+8, buffer.sequence); u32(header+12, buffer.flags);
        u64(header+16, (uint64_t)buffer.timestamp.tv_sec*1000000000ULL + (uint64_t)buffer.timestamp.tv_usec*1000ULL);
        u64(header+24, (uint64_t)now.tv_sec*1000000000ULL + (uint64_t)now.tv_nsec);
        u32(header+32, format.fmt.pix.width); u32(header+36, format.fmt.pix.height);
        if (write_all(header, sizeof(header)) || write_all(maps[buffer.index], buffer.bytesused)) goto done;
        if (control(fd, VIDIOC_QBUF, &buffer) < 0) { perror("requeue camera buffer"); goto done; }
    }
    result = 0;
done:
    if (streaming) { enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd, VIDIOC_STREAMOFF, &type); }
    for (int i = 0; i < 4; ++i) if (maps[i]) munmap(maps[i], lengths[i]);
    if (fd >= 0) close(fd);
    return result;
}
