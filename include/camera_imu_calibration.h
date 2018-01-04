#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <iostream>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

using namespace cv;
using namespace std;

class Calibrator
{
public:
    Calibrator();
    ~Calibrator();
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void imu_callback(const sensor_msgs::ImuConstPtr& msg);

private:
    void get_corners(const InputArray img, vector<Point2f>& charucoCorners, vector<int>& charucoIds, OutputArray tracked);

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
};
