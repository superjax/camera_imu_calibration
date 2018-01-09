#include <vector>
#include <deque>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core/eigen.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_unstable/slam/ProjectionFactorPPPC.h>
#include <gtsam_unstable/slam/ProjectionFactorPPP.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Cal3DS2.h>

using namespace cv;
using namespace std;
using namespace Eigen;
namespace gt = gtsam;


using gt::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gt::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gt::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gt::symbol_shorthand::L; // Landmarks (x, y, z)
using gt::symbol_shorthand::K; // Calibration (fx, fy, s, px, py)
using gt::symbol_shorthand::T; // Camera-IMU Transform(r, p, y, x, y, z)

class Calibrator
{
public:
    Calibrator();
    ~Calibrator();
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void imu_callback(const sensor_msgs::ImuConstPtr& msg);

private:

    void get_corners(const InputArray img, vector<Point2f>& charucoCorners, vector<int>& charucoIds, OutputArray tracked, OutputArray &R, OutputArray &tvec);
    void initialize_graph(Matrix3d& Rot, Vector3d& trans, Matrix3d& camK, Matrix<double, 5, 1> &camD);
    void add_measurement_to_graph(const vector<Point2f> &corners, const vector<int> &ids, const Vector3d pos);

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    image_transport::ImageTransport it_;

    image_transport::Publisher output_pub_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber imu_sub_;
    ros::Publisher odometry_pub_;

    Mat camMatrix_;
    Mat distCoeffs_;

    // Create a bunch of aruco objects
    Ptr<aruco::DetectorParameters> detectorParams_;
    Ptr<aruco::Dictionary> dictionary_;

    // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard_;
    Ptr<aruco::Board> board_;

    bool show_tracked_ = false;
    uint64_t last_imu_time_nsec = 0;

    // GTSAM Factor Graph objects
    gt::NonlinearFactorGraph fg_;
    gt::Values initial_values_;
    int num_poses_ = 0;

    gt::ISAM2* isam_;
    gt::PreintegratedCombinedMeasurements *imu_preintegrated_;

    // Constant Noise Models
    gt::noiseModel::Diagonal::shared_ptr z_cov_;
    gt::noiseModel::Diagonal::shared_ptr l_cov_;
    gt::noiseModel::Diagonal::shared_ptr imu_bias_init_cov_;

    gt::NavState current_estimated_state_;
    gt::imuBias::ConstantBias current_estimated_imu_bias_;
    gt::imuBias::ConstantBias init_imu_bias_;
};

