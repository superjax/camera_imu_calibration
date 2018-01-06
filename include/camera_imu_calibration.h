#include <vector>
#include <deque>
#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <Eigen/Core>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_unstable/slam/ProjectionFactorPPPC.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/inference/Symbol.h>

using namespace cv;
using namespace std;
using namespace Eigen;
namespace gt = gtsam;


using gt::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gt::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gt::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gt::symbol_shorthand::L; // Landmarks (x, y, z)
using gt::symbol_shorthand::K; // Calibration (fx, fy, s, px, py)

class Calibrator
{
public:
    Calibrator();
    ~Calibrator();
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void imu_callback(const sensor_msgs::ImuConstPtr& msg);

private:

    typedef struct
    {
        Vector3d acc;
        Vector3d gyro;
        uint64_t usec;
    } imu_measurement_t;

    typedef struct
    {
        vector<Point2f> corner_positions;
        vector<int> ids;
        uint64_t usec;
    } camera_measurement_t;

    deque<imu_measurement_t> imu_queue_;
    camera_measurement_t camera_meas_;
    bool new_camera_;

    void get_corners(const InputArray img, vector<Point2f>& charucoCorners, vector<int>& charucoIds, OutputArray tracked);
    void process_measurement_queues();
    void initialize_graph();
    void add_measurement_to_graph(const Vector3d &dtheta, const Vector3d &dvel, const vector<Point2f> &corners, const vector<int> &ids);

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    image_transport::ImageTransport it_;

    image_transport::Publisher output_pub_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber imu_sub_;

    Mat camMatrix_;
    Mat distCoeffs_;

    // Create a bunch of aruco objects
    Ptr<aruco::DetectorParameters> detectorParams_;
    Ptr<aruco::Dictionary> dictionary_;

    // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard_;
    Ptr<aruco::Board> board_;

    bool show_tracked_ = false;

    // GTSAM Factor Graph objects
    gt::Values initial_values_;
    int num_poses_ = 0;

    gt::NonlinearFactorGraph graph_;
    gt::PreintegratedCombinedMeasurements *imu_preintegrated_;

    gt::Pose3* camera_transform_;
};
