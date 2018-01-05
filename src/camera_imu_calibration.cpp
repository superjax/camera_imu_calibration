#include "camera_imu_calibration.h"

using namespace std;
using namespace cv;

Calibrator::Calibrator() :
    nh_private_("~"),
    it_(nh_)
{
    imu_sub_ = nh_.subscribe("imu", 100, &Calibrator::imu_callback, this);
    image_sub_ = it_.subscribe("/cv_camera/image_raw", 1, &Calibrator::image_callback, this);
    output_pub_ = it_.advertise("/image_converter/tracked", 1);

    nh_private_.param<bool>("show_tracked", show_tracked_, true);
    if (show_tracked_)
    {
        cv::namedWindow("tracked");
    }

    // Load camera intrinsics
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

    // Add Measurement to Queue
    camera_meas_.corner_positions = charucoCorners;
    camera_meas_.ids = charucoIds;
    camera_meas_.usec = msg->header.stamp.toNSec();
    new_camera_ = true;

    // Output modified video stream
    cv_ptr->image = tracked;
    output_pub_.publish(cv_ptr->toImageMsg());
}

void Calibrator::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
    imu_measurement_t meas;
    meas.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
    meas.gyro << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    meas.usec = msg->header.stamp.toNSec();
    imu_queue_.push_back(meas);
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

void Calibrator::process_measurement_queues()
{
    // If we have a new camera measurement, integrate all the IMU data between these frames,
    // but check to make sure that we have all the data between these frames
    // by making sure that we have at least one IMU measurement which comes after the image
    // TODO: propagate right up to the frame, and save for later, rather than integrating to the closest previous IMU measurement
    if (new_camera_)
    {
        if (imu_queue_.end()->usec > camera_meas_.usec)
        {
            Vector3d dtheta;
            Vector3d dvel;
            while (imu_queue_.begin()->usec < camera_meas_.usec)
            {
                imu_measurement_t meas = imu_queue_.front();
                imu_queue_.pop_front();
                uint64_t dt = (imu_queue_.begin())->usec - meas.usec;
                dtheta += dt * meas.gyro;
                dvel += dt * meas.acc;
            }

            // We now have a new measurement to add to the graph
            add_measurement_to_graph(dtheta, dvel, camera_meas_.corner_positions, camera_meas_.ids);
            new_camera_ = false;
        }
    }
}

void Calibrator::initialize_graph(const gtsam::Pose3 initial_pose)
{
    gt::Vector3 prior_velocity(0.0, 0.0, 0.0);
    gt::imuBias::ConstantBias prior_imu_bias; // assume zero

    // Add Origin to initial guess
    initial_values_.insert(X(num_poses_), initial_pose);
    initial_values_.insert(V(num_poses_), prior_velocity);
    initial_values_.insert(B(num_poses_), prior_imu_bias);

    // Create noise estimates for initial pose, velocity and bias
    gt::noiseModel::Diagonal::shared_ptr pose_noise_model = gt::noiseModel::Diagonal::Sigmas((gt::Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished()); // rad,rad,rad, m, m, m
    gt::noiseModel::Diagonal::shared_ptr velocity_noise_model = gt::noiseModel::Isotropic::Sigma(3,0.1); // m/s
    gt::noiseModel::Diagonal::shared_ptr bias_noise_model = gt::noiseModel::Isotropic::Sigma(6,1e-3);

    // Add all prior factors to the graph
    graph_.add(gt::PriorFactor<gt::Pose3>(X(num_poses_), initial_pose, pose_noise_model));
    graph_.add(gt::PriorFactor<gt::Vector3>(V(num_poses_), prior_velocity,velocity_noise_model));
    graph_.add(gt::PriorFactor<gt::imuBias::ConstantBias>(B(num_poses_), prior_imu_bias,bias_noise_model));

    // Add chessboard corner XYZ positions to initial_values and graph
    gt::noiseModel::Diagonal::shared_ptr corner_noise_model = gt::noiseModel::Isotropic::Sigma(3,1e-4); // m
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 7; col++)
        {
            gt::Point3 marker_point(0.0351f * (row+1), 0.0351f * (col+1), 0.0);
            int corner_num = row + (5*col);
            initial_values_.insert(L(corner_num), marker_point);
            graph_.add(gt::PriorFactor<gt::Point3>(L(corner_num), marker_point ,corner_noise_model));
        }
    }

    // We use the sensor specs to build the noise model for the IMU factor.
    double accel_noise_sigma = 0.0003924;
    double gyro_noise_sigma = 0.000205689024915;
    double accel_bias_rw_sigma = 0.004905;
    double gyro_bias_rw_sigma = 0.000001454441043;
    gt::Matrix33 measured_acc_cov = gt::Matrix33::Identity(3,3) * pow(accel_noise_sigma,2);
    gt::Matrix33 measured_omega_cov = gt::Matrix33::Identity(3,3) * pow(gyro_noise_sigma,2);
    gt::Matrix33 integration_error_cov = gt::Matrix33::Identity(3,3)*1e-8; // error committed in integrating position from velocities
    gt::Matrix33 bias_acc_cov = gt::Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2);
    gt::Matrix33 bias_omega_cov = gt::Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2);
    gt::Matrix66 bias_acc_omega_int = gt::Matrix::Identity(6,6)*1e-5; // error in the bias used for preintegration

    // Create a NED preintegration shared parameter
    boost::shared_ptr<gt::PreintegratedCombinedMeasurements::Params> p = gt::PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);
    // PreintegrationBase params:
    p->accelerometerCovariance = measured_acc_cov; // acc white noise in continuous
    p->integrationCovariance = integration_error_cov; // integration uncertainty continuous
    // should be using 2nd order integration
    // PreintegratedRotation params:
    p->gyroscopeCovariance = measured_omega_cov; // gyro white noise in continuous
    // PreintegrationCombinedMeasurements params:
    p->biasAccCovariance = bias_acc_cov; // acc bias in continuous
    p->biasOmegaCovariance = bias_omega_cov; // gyro bias in continuous
    p->biasAccOmegaInt = bias_acc_omega_int;


    imu_preintegrated_ = new gt::PreintegratedCombinedMeasurements(p, prior_imu_bias);


    // Store previous state for the imu integration and the latest predicted outcome.
    NavState prev_state(prior_pose, prior_velocity);
    NavState prop_state = prev_state;
    imuBias::ConstantBias prev_bias = prior_imu_bias;





}


void Calibrator::add_measurement_to_graph(const Vector3d& dtheta, const Vector3d& dvel, const vector<Point2f>& corners, const vector<int>& ids)
{

}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "camera_imu_calibrator");
    Calibrator thing;
    ros::spin();
    return 0;
}

