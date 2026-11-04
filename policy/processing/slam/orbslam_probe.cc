// ORB-SLAM3 offline probe for monocular-inertial (and camera-only) runs on EuRoC-style exports.
// Logs the online tracking state for every frame and, at shutdown, the final (BA-optimized)
// trajectory of every map in the atlas. Baked into the umi-excavator-processing image (policy/processing/docker/Dockerfile).
// Pre-include third-party headers so the access override below only affects ORB-SLAM3 classes.
#include <sstream>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#define private public
#define protected public
#include <System.h>
#include <Atlas.h>
#include <Map.h>
#include <MapPoint.h>
#include <Tracking.h>
#undef private
#undef protected
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: orbslam_probe vocabulary settings dataset timestamps outdir imu|mono\n";
        return 2;
    }
    const std::string dataset = argv[3], outdir = argv[5], mode = argv[6];
    const bool inertial = mode == "imu";
    std::vector<long long> times;
    { std::ifstream f(argv[4]); std::string line;
      while (std::getline(f, line)) if (!line.empty() && line[0] != '#') times.push_back(std::stoll(line)); }
    std::vector<double> it; std::vector<cv::Point3f> ia, ig;
    if (inertial) {
        std::ifstream f(dataset + "/mav0/imu0/data.csv"); std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line); std::string tok; std::vector<double> v;
            while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
            if (v.size() != 7) continue;
            it.push_back(v[0] * 1e-9); ig.emplace_back(v[1], v[2], v[3]); ia.emplace_back(v[4], v[5], v[6]);
        }
        if (it.empty() || it.front() >= times.front() * 1e-9 || it.back() <= times.back() * 1e-9) {
            std::cerr << "IMU must bracket camera timestamps\n"; return 2;
        }
    }
    std::ofstream log(outdir + "/tracking.csv");
    log << "frame,timestamp_ns,state,map_id,imu_initialized,inertial_ba2,map_points,tx,ty,tz,qx,qy,qz,qw\n";
    log << std::setprecision(12);
    cv::setNumThreads(1);
    ORB_SLAM3::System slam(argv[1], argv[2], inertial ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::MONOCULAR, false);
    size_t j = 0;
    while (inertial && j < it.size() && it[j] <= times[0] * 1e-9) ++j;
    for (size_t i = 0; i < times.size(); ++i) {
        const double t = times[i] * 1e-9;
        cv::Mat image = cv::imread(dataset + "/mav0/cam0/data/" + std::to_string(times[i]) + ".png", cv::IMREAD_GRAYSCALE);
        if (image.empty()) { std::cerr << "Missing image " << times[i] << "\n"; slam.Shutdown(); return 2; }
        std::vector<ORB_SLAM3::IMU::Point> meas;
        if (inertial && i > 0) {
            while (j < it.size() && it[j] <= t) {
                meas.emplace_back(ia[j].x, ia[j].y, ia[j].z, ig[j].x, ig[j].y, ig[j].z, it[j]); ++j;
            }
        }
        const auto began = std::chrono::steady_clock::now();
        Sophus::SE3f Tcw = slam.TrackMonocular(image, t, meas);
        const int state = slam.GetTrackingState();
        ORB_SLAM3::Map* map = slam.mpAtlas->GetCurrentMap();
        size_t points = 0;
        for (auto* p : slam.GetTrackedMapPoints()) if (p && !p->isBad()) ++points;
        log << i << ',' << times[i] << ',' << state << ',' << (map ? (long long)map->GetId() : -1) << ','
            << (map && map->isImuInitialized()) << ',' << (map && map->GetIniertialBA2()) << ',' << points;
        if ((state == 2 || state == 3) && Tcw.matrix().allFinite()) {
            auto Twc = Tcw.inverse(); auto q = Twc.unit_quaternion(); auto p = Twc.translation();
            log << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w();
        } else log << ",,,,,,,";
        log << '\n'; log.flush();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        if (i + 1 < times.size()) {
            // Replay at 2x real time at most so local mapping keeps up with tracking.
            const double wait = 2.0 * (times[i + 1] - times[i]) * 1e-9 - elapsed;
            if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    slam.Shutdown();
    std::ofstream maps(outdir + "/maps.csv");
    maps << "map_id,keyframes,map_points,imu_initialized,inertial_ba2\n";
    for (ORB_SLAM3::Map* m : slam.mpAtlas->GetAllMaps()) {
        const auto kfs = m->GetAllKeyFrames();
        maps << m->GetId() << ',' << kfs.size() << ',' << m->GetAllMapPoints().size() << ','
             << m->isImuInitialized() << ',' << m->GetIniertialBA2() << '\n';
        if (kfs.size() >= 2) slam.SaveTrajectoryEuRoC(outdir + "/map_" + std::to_string(m->GetId()) + ".txt", m);
    }
    return 0;
}
