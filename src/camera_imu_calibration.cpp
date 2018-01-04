#include "camera_imu_calibration.h"

using namespace std;
using namespace cv;

Calibrator::Calibrator() :
    nh_private_("~"),
    it_(nh_)
{
    image_sub_ = it_.subscribe("/cv_camera/image_raw", 1, &Calibrator::image_callback, this);
    output_pub_ = it_.advertise("/image_converter/tracked", 1);

    nh_private_.param<bool>("show_tracked", show_tracked_, true);
    if (show_tracked_)
    {
        cv::namedWindow("tracked");
    }

    // Load intrinsics
    string intrinsics_path;
    nh_private_.param<string>("intrinsics_path", intrinsics_path, ros::package::getPath("camera_imu_calibration")+"/param/elp.yaml");
    FileStorage fs(intrinsics_path, FileStorage::READ);
    if (!fs.isOpened()) {
        ROS_FATAL("unable to find camera intrinsics at %s", intrinsics_path.c_str());
    } else {
        fs["camMatrix"] >> camMatrix_;
        fs["distCoeffs"] >> distCoeffs_;
    }

    // Create a bunch of aruco objects
    detectorParams_ = aruco::DetectorParameters::create();
    dictionary_ = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(aruco::DICT_6X6_50));

    // create charuco board object
    charucoboard_ = aruco::CharucoBoard::create(5, 7, 0.0351f, 0.0222f, dictionary_);
    board_ = charucoboard_.staticCast<aruco::Board>();
}

Calibrator::~Calibrator()
{
    if (show_tracked_)
    {
        cv::destroyWindow("tracked");
    }
}

void Calibrator::image_callback(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_FATAL("cv_bridge exception: %s", e.what());
        return;
    }

    // Track Corners
    vector<Point2f> charucoCorners;
    vector<int> charucoIds;
    Mat tracked;
    get_corners(cv_ptr->image, charucoCorners, charucoIds, tracked);

    // Output modified video stream
    cv_ptr->image = tracked;
    output_pub_.publish(cv_ptr->toImageMsg());
}

void Calibrator::get_corners(const InputArray image, vector<Point2f>& charucoCorners, vector<int>& charucoIds,
                             OutputArray tracked=noArray())
{
    // length of axis when drawing pose
    float axisLength = 0.5f * (5 * 0.0351);

    bool draw_rejected = false;

    // Containers to hold marker stuff
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedMarkers;

    // detect markers
    aruco::detectMarkers(image, dictionary_, markerCorners, markerIds, detectorParams_,
                         rejectedMarkers);

    // refind strategy to detect more markers
    aruco::refineDetectedMarkers(image, board_, markerCorners, markerIds, rejectedMarkers,
                                 camMatrix_, distCoeffs_);

    // interpolate charuco corners
    int interpolatedCorners = 0;
    if(markerIds.size() > 0)
        interpolatedCorners =
                aruco::interpolateCornersCharuco(markerCorners, markerIds, image, charucoboard_,
                                                 charucoCorners, charucoIds, camMatrix_, distCoeffs_);

    // draw results if an output array was supplied
    Mat imageCopy;
    image.copyTo(imageCopy);
    if(markerIds.size() > 0)
    {
        aruco::drawDetectedMarkers(imageCopy, markerCorners);
    }

    // Draw the rejected markers
    if (draw_rejected)
        aruco::drawDetectedMarkers(imageCopy, rejectedMarkers, noArray(), Scalar(100, 0, 255));

    if(interpolatedCorners > 0)
    {
        Scalar color;
        color = Scalar(255, 0, 150);
        aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, color);
    }

    // estimate and draw charuco board pose (requires camera intrinsics)
    if(camMatrix_.total() != 0)
    {
        Vec3d rvec, tvec;
        if ( aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard_,
                                             camMatrix_, distCoeffs_, rvec, tvec) )
        {
            aruco::drawAxis(imageCopy, camMatrix_, distCoeffs_, rvec, tvec, axisLength);
        }
    }
    imageCopy.copyTo(tracked);

    // Update GUI Window
    if (show_tracked_)
    {
        cv::imshow("tracked", tracked);
        cv::waitKey(3);
    }
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "camera_imu_calibrator");
    Calibrator thing;
    ros::spin();
    return 0;
}

