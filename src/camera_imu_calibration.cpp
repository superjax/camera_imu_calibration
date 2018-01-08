#include "camera_imu_calibration.h"

using namespace std;
using namespace cv;

Calibrator::Calibrator() :
    nh_private_("~"),
    it_(nh_)
{
    imu_sub_ = nh_.subscribe("imu/data", 100, &Calibrator::imu_callback, this);
    odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);
    image_sub_ = it_.subscribe("cv_camera/image_raw", 1, &Calibrator::image_callback, this);
    output_pub_ = it_.advertise("image_converter/tracked", 1);

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

    initialize_graph();
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

    add_measurement_to_graph(charucoCorners, charucoIds);
}

void Calibrator::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  if (last_imu_time_nsec == 0)
  {
    last_imu_time_nsec = msg->header.stamp.toNSec();
    imu_preintegrated_->resetIntegration();
    return;
  }

  uint64_t now = msg->header.stamp.toNSec();
  uint64_t dt = now - last_imu_time_nsec;
  last_imu_time_nsec = now;
  imu_preintegrated_->integrateMeasurement(gt::Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
                                           gt::Vector3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                           dt*1e-9);
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


void Calibrator::initialize_graph()
{
  // Initial guess and uncertainty for pose
  gt::Pose3 init_pose(gt::Rot3::Quaternion(1.0, 0.0, 0.0, 0.0), gt::Point3(0.0, 0.0, 0.0));
  gt::noiseModel::Diagonal::shared_ptr init_pose_cov = gt::noiseModel::Diagonal::Sigmas((gt::Vector(6) << 5*M_PI/180.0, 5*M_PI/180.0, 5*M_PI/180.0, 20, 20, 0.1).finished());

  // Initial Guess and uncertainty for Camera Intrinsics
  gt::Cal3DS2 init_K((gt::Vector(9) << 400, 400, 0, 320, 240, 0, 0, 0, 0).finished());
  gt::noiseModel::Diagonal::shared_ptr init_K_cov = gt::noiseModel::Diagonal::Sigmas((gt::Vector(9) << 20.0, 20.0, 0.001, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0).finished());

  // Initial Guess and uncertainty for velocity
  gt::Vector3 init_vel(0.0, 0.0, 0.0);
  gt::noiseModel::Diagonal::shared_ptr init_vel_cov = gt::noiseModel::Isotropic::Sigma(3, 0.1);

  // Initial Guess and uncertainty for IMU bias (will be used again for the second pose)
  init_imu_bias_ = gt::imuBias::ConstantBias(gt::Vector3::Zero(), gt::Vector3::Zero());
  imu_bias_init_cov_ = gt::noiseModel::Diagonal::Sigmas((gt::Vector(6) << 0.1, 0.1, 0.1, 5e-5, 5e-5, 5e-5).finished());
  gt::noiseModel::Diagonal::shared_ptr imu_bias_prop_cov = gt::noiseModel::Isotropic::Sigma(6, 5e-5);

  // IMU Preintegration Parameters
  boost::shared_ptr<gt::PreintegratedCombinedMeasurements::Params> IMU_params = boost::make_shared<gt::PreintegratedCombinedMeasurements::Params>(gt::Vector3(0, 0, 9.80665));
  IMU_params->setAccelerometerCovariance(gt::Matrix33::Identity() * pow(1e-2, 2));
  IMU_params->setGyroscopeCovariance(gt::Matrix33::Identity() * pow(1e-2, 2));
  IMU_params->setIntegrationCovariance(gt::Matrix33::Identity() * pow(1e-1, 2));
  IMU_params->biasAccCovariance = gt::Matrix33::Identity() * pow(0.004905,2);
  IMU_params->biasOmegaCovariance = gt::Matrix33::Identity() * pow(0.000001454441043 ,2);
  IMU_params->biasAccOmegaInt = gt::Matrix::Identity(6,6)*1e-5;
  imu_preintegrated_ = new gt::PreintegratedCombinedMeasurements(IMU_params, init_imu_bias_);

  // Noise Models
  l_cov_ = gt::noiseModel::Isotropic::Sigma(3, 0.0005); // landmark location noise (meters)
  z_cov_ = gt::noiseModel::Isotropic::Sigma(2, 1.0); // pixel measurement noise (pixels)

  // Initialize the isam solver
  gt::ISAM2Params isam_params;
  isam_params.setFactorization("QR");
  isam_ = new gt::ISAM2(isam_params);

  // add initial values for pose, velocity, imu bias, camera calibration and landmarks
  initial_values_.insert(X(0), init_pose);
  initial_values_.insert(V(0), init_vel);
  initial_values_.insert(B(0), current_estimated_imu_bias_);
  initial_values_.insert(K(0), init_K);
  fg_.add(gt::PriorFactor<gt::Pose3>(X(0), init_pose, init_pose_cov));
  fg_.add(gt::PriorFactor<gt::Vector3>(V(0), init_vel, init_vel_cov));
  fg_.add(gt::PriorFactor<gt::imuBias::ConstantBias>(B(0), init_imu_bias_, imu_bias_init_cov_));
  fg_.add(gt::PriorFactor<gt::Cal3DS2>(K(0), init_K, init_K_cov));
  for (int row = 0; row < 4; row++)
  {
    for (int col = 0; col < 6; col++)
    {
      gt::Point3 landmark((row+1) * 0.0351f, (col+1) * 0.0351f, 0.0f);
      int l_id = row + 4 * col;
      initial_values_.insert(L(l_id), landmark);
      fg_.add(gt::PriorFactor<gt::Point3>(L(l_id), landmark, l_cov_));
    }
  }


  // Save graph, just for fun
  ofstream file;
  file.open(ros::package::getPath("camera_imu_calibration") + "/saved_graph.gv", ios::out);
  fg_.saveGraph(file, initial_values_);

  // Initialize estimates
  current_estimated_state_ = gt::NavState(init_pose, init_vel);
  current_estimated_imu_bias_ = init_imu_bias_;
  num_poses_ = 1;
}


void Calibrator::add_measurement_to_graph(const vector<Point2f>& corners, const vector<int>& ids)
{
  if (num_poses_ == 0)
  {
    ROS_FATAL("attempting to add measurements to uninitialized graph");
    return; // We shouldn't ever get here
  }

  // Add IMU priors on the second pose as well
  if (num_poses_ == 1)
  {
    fg_.add(gt::PriorFactor<gt::imuBias::ConstantBias>(B(1), init_imu_bias_, imu_bias_init_cov_));
  }

  // Add the IMU preintegration factor to the graph
  gt::CombinedImuFactor imu_factor(X(num_poses_-1), V(num_poses_-1),
                                   X(num_poses_  ), V(num_poses_  ),
                                   B(num_poses_-1), B(num_poses_  ),
                                   *imu_preintegrated_);
  fg_.add(imu_factor);
  // Best guess to initialize the new pose, velocity and IMU bias nodes
  current_estimated_state_ = imu_preintegrated_->predict(current_estimated_state_, current_estimated_imu_bias_);
  initial_values_.insert(X(num_poses_), current_estimated_state_.pose());
  initial_values_.insert(V(num_poses_), current_estimated_state_.v());
  initial_values_.insert(B(num_poses_), current_estimated_imu_bias_);

  // Reset IMU preintegrator
  imu_preintegrated_->resetIntegration();

  // Add pixel measurements to graph
  for (int feat_idx = 0 ; feat_idx < ids.size(); feat_idx++)
  {
    gt::Point2 z(corners[feat_idx].x, corners[feat_idx].y);
    int id = ids[feat_idx];
    fg_.add(gt::ProjectionFactorPPPC<gt::Pose3, gt::Point3, gt::Cal3DS2>(gt::Point2(z), z_cov_, Z(id), X(num_poses_), L(id), K(0)));
  }

  // Optimize with ISAM
//  isam_->update(fg_, initial_values_);
//  gt::Values result = isam_->calculateEstimate();

//  // Get current estimates
//  current_estimated_state_ = gt::NavState(result.at<gt::Pose3>(X(num_poses_)),
//                                          result.at<gt::Vector3>(V(num_poses_)));
//  current_estimated_imu_bias_ = result.at<gt::imuBias::ConstantBias>(B(num_poses_));

//  cout << "Current_State Estimate: Q:\n " << current_estimated_state_.pose().rotation().quaternion() << "\nT:\n" << current_estimated_state_.pose().translation() << "\n";

  num_poses_++;

  // Publish current odometry estimate.
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = ros::Time::now();
  odom_msg.pose.pose.position.x = current_estimated_state_.pose().translation().x();
  odom_msg.pose.pose.position.y = current_estimated_state_.pose().translation().y();
  odom_msg.pose.pose.position.z = current_estimated_state_.pose().translation().z();
  odom_msg.pose.pose.orientation.w = current_estimated_state_.pose().rotation().quaternion()[0];
  odom_msg.pose.pose.orientation.x = current_estimated_state_.pose().rotation().quaternion()[1];
  odom_msg.pose.pose.orientation.y = current_estimated_state_.pose().rotation().quaternion()[2];
  odom_msg.pose.pose.orientation.z = current_estimated_state_.pose().rotation().quaternion()[3];
  odometry_pub_.publish(odom_msg);
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "camera_imu_calibrator");
    Calibrator thing;
    ros::spin();
    return 0;
}

