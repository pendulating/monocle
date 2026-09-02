// test_ffmpeg_decode.cpp - prove FFmpeg decode works in the vendored OpenCV.
// Build: see setup_env.sh stage verify.
// Run: test_ffmpeg_decode <video-file>
// The program opens the file with the default backend and reads 10 frames.

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <video-file>" << std::endl;
        return 2;
    }

    // Default backend order. The vendored OpenCV holds FFmpeg only.
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "FAIL: open failed: " << argv[1] << std::endl;
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "opened: " << argv[1] << std::endl;
    std::cout << "size: " << width << "x" << height
              << " fps: " << fps << " frames: " << frames << std::endl;

    cv::Mat frame;
    int count = 0;
    while (count < 10 && cap.read(frame) && !frame.empty()) {
        count++;
    }
    std::cout << "decoded frames: " << count << "/10" << std::endl;

    if (count == 0) {
        std::cerr << "FAIL: no frames decoded" << std::endl;
        return 1;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
