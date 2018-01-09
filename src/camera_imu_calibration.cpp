#include "camera_imu_calibration.h"

using namespace std;
using namespace cv;

Calibrator::Calibrator() :
  nh_private_("~"),
  it_(nh_)
{
  imu_sub_ = nh_.subscribe("imu/data", 500, &Calibrator::imu_callback, this);
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
    fs["distCoeff"] >> distCoeffs_;
  }

  // Check that intrinsics were loaded properly
  if (camMatrix_.cols != 3 || camMatrix_.rows != 3 || distCoeffs_.rows != 1 || distCoeffs_.cols != 5)
  {
    ROS_FATAL("unable to load camera intrinsics at %s", intrinsics_path.c_str());
  }

  // Create a bunch of aruco objects
  detectorParams_ = aruco::DetectorParameters::create();
  dictionary_ = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(aruco::DICT_6X6_50));

  // create charuco board object
  charucoboard_ = aruco::CharucoBoard::create(5, 7, 0.0351f, 0.0222f, dictionary_);
  board_ = charucoboard_.staticCast<aruco::Board>();

  cout << "\n\n\n\n\n\n\n\n";
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
  Vec3d tvec;
  Mat Rot;
  get_corners(cv_ptr->image, charucoCorners, charucoIds, tracked, Rot, tvec);

  Matrix3d R, camK;
  Vector3d T;
  Matrix<double, 5, 1> D;
  if (charucoCorners.size() > 5)
  {
    cv2eigen(Rot, R);
    cv2eigen(tvec, T);
    cv2eigen(camMatrix_, camK);
    cv2eigen(distCoeffs_, D);

    // Convert to World Coordinates (get_corners returns the transform to the charuco board
    // wrt the camera frame, we just need to invert the transform)
    T = R.transpose() * -T;
    R.transposeInPlace();

    if (num_poses_ == 0 && charucoCorners.size() == 4 * 6)
    {
      initialize_graph(R, T, camK, D);
      return;
    }
  }

  if (num_poses_ > 0)
  {
    add_measurement_to_graph(charucoCorners, charucoIds, T);
  }

  // Output modified video stream
  cv_ptr->image = tracked;
  output_pub_.publish(cv_ptr->toImageMsg());

}

void Calibrator::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  // Just record timestamps if we haven't initialized the graph
  if (last_imu_time_nsec == 0 || num_poses_ == 0)
  {
    last_imu_time_nsec = msg->header.stamp.toNSec();
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
                             OutputArray tracked=noArray(), OutputArray& R=noArray(), OutputArray& tvec=noArray())
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
    Mat rvec;
    if ( aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard_,
                                         camMatrix_, distCoeffs_, rvec, tvec) )
    {
      Rodrigues(rvec, R);
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


void Calibrator::initialize_graph(Matrix3d& Rot, Vector3d& trans, Matrix3d& camK, Matrix<double, 5, 1>& camD)
{
  // Initial guess and uncertainty for pose
  gt::Rot3 rot(Rot);
  gt::Pose3 init_pose(rot, gt::Point3(trans));
  gt::noiseModel::Diagonal::shared_ptr init_pose_cov = gt::noiseModel::Diagonal::Sigmas((gt::Vector(6) << 5*M_PI/180.0, 5*M_PI/180.0, 5*M_PI/180.0, 0.05, 0.05, 0.05).finished());

  // Initial Guess and uncertainty for Camera Intrinsics
  gt::Cal3DS2 init_K((gt::Vector(9) << camK(0, 0), camK(1, 1), camK(0, 1), camK(0, 2), camK(1, 2), camD(0), camD(1), camD(2), camD(3)).finished());
  gt::noiseModel::Diagonal::shared_ptr init_K_cov = gt::noiseModel::Diagonal::Sigmas((gt::Vector(9) << 5.0, 5.0, 0.001, 5.0, 5.0, 0.005, 0.005, 0.0005, 0.0005).finished());

  // Initial Guess and uncertainty for Camera-IMU Extrinsics
  gt::Pose3 init_cam_transform(gt::Rot3::Quaternion(1.0, 0.0, 0.0, 0.0), gt::Point3(0.0, 0.0, 0.0));
  gt::noiseModel::Diagonal::shared_ptr init_cam_transform_cov = gt::noiseModel::Diagonal::Sigmas((gt::Vector(6) << 5*M_PI/180.0, 5*M_PI/180.0, 5*M_PI/180.0, 0.05, 0.05, 0.05).finished());

  // Initial Guess and uncertainty for velocity
  gt::Vector3 init_vel(0.0, 0.0, 0.0);
  gt::noiseModel::Diagonal::shared_ptr init_vel_cov = gt::noiseModel::Isotropic::Sigma(3, 0.1);

  // Initial Guess and uncertainty for IMU bias (will be used again for the second pose)
  init_imu_bias_ = gt::imuBias::ConstantBias(gt::Vector3::Zero(), gt::Vector3::Zero());
  imu_bias_init_cov_ = gt::noiseModel::Diagonal::Sigmas((gt::Vector(6) << 0.1, 0.1, 0.1, 5e-5, 5e-5, 5e-5).finished());

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

  // add initial values for pose, velocity, imu bias, camera calibration, camera transform and landmarks
  initial_values_.insert(X(0), init_pose);
  initial_values_.insert(V(0), init_vel);
  initial_values_.insert(B(0), current_estimated_imu_bias_);
  initial_values_.insert(K(0), init_K);
  initial_values_.insert(T(0), init_cam_transform);
  fg_.add(gt::PriorFactor<gt::Pose3>(X(0), init_pose, init_pose_cov));
  fg_.add(gt::PriorFactor<gt::Vector3>(V(0), init_vel, init_vel_cov));
  fg_.add(gt::PriorFactor<gt::imuBias::ConstantBias>(B(0), init_imu_bias_, imu_bias_init_cov_));
  fg_.add(gt::PriorFactor<gt::Cal3DS2>(K(0), init_K, init_K_cov));
  fg_.add(gt::PriorFactor<gt::Pose3>(T(0), init_cam_transform, init_cam_transform_cov));
  for (int col = 0; col < 4; col++)
  {
    for (int row = 0; row < 6; row++)
    {
      gt::Point3 landmark((col+1) * 0.035f, (row+1) * 0.035f, 0.0f);
      int l_id = col + 4 * row;
      initial_values_.insert(L(l_id), landmark);
      fg_.add(gt::PriorFactor<gt::Point3>(L(l_id), landmark, l_cov_));
    }
  }

  // Initialize estimates
  current_estimated_state_ = gt::NavState(init_pose, init_vel);
  current_estimated_imu_bias_ = init_imu_bias_;
  num_poses_ = 1;
}


void Calibrator::add_measurement_to_graph(const vector<Point2f>& corners, const vector<int>& ids, const Vector3d pos)
{
  if (num_poses_ == 0)
  {
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
  gt::NavState new_estimated_state = imu_preintegrated_->predict(current_estimated_state_, current_estimated_imu_bias_);
  initial_values_.insert(X(num_poses_), new_estimated_state.pose());
  initial_values_.insert(V(num_poses_), new_estimated_state.v());
  initial_values_.insert(B(num_poses_), current_estimated_imu_bias_);
  current_estimated_state_ = new_estimated_state;

  // Reset IMU preintegrator
  imu_preintegrated_->resetIntegration();

  // Add position measurement to graph
  gt::noiseModel::Diagonal::shared_ptr pos_noise = gt::noiseModel::Isotropic::Sigma(3,0.05);
  fg_.add(gt::GPSFactor(X(num_poses_), gt::Point3(pos(0), pos(1), pos(2)), pos_noise));

  // Add pixel measurements to graph
  for (int feat_idx = 0 ; feat_idx < ids.size(); feat_idx++)
  {
    gt::Point2 z(corners[feat_idx].x, corners[feat_idx].y);
    int id = ids[feat_idx];
    fg_.add(gt::ProjectionFactorPPPC<gt::Pose3, gt::Point3, gt::Cal3DS2>(
              gt::Point2(z), z_cov_, X(num_poses_), T(0), L(id), K(0)));
  }

//  gt::ProjectionFactorPPP



  // Optimize with ISAM
  gt::Values result;
  try
  {
    isam_->update(fg_, initial_values_);
    result = isam_->calculateEstimate();
  }
  catch(...)
  {
    // Save graph, just for fun
    ROS_FATAL("Busted");
    isam_->saveGraph(ros::package::getPath("camera_imu_calibration") + "/saved_graph.gv");
    throw;
  }

  // Reset the factor graph and initial values container for the next iteration
  fg_.resize(0);
  initial_values_.clear();

  // Get current estimates
  current_estimated_state_ = gt::NavState(result.at<gt::Pose3>(X(num_poses_)),
                                          result.at<gt::Vector3>(V(num_poses_)));
  current_estimated_imu_bias_ = result.at<gt::imuBias::ConstantBias>(B(num_poses_));

  gt::Pose3 camera_transform = result.at<gt::Pose3>(T(0));
  gt::Cal3DS2 camera_intrinsics = result.at<gt::Cal3DS2>(K(0));
  gt::imuBias::ConstantBias imu_bias = result.at<gt::imuBias::ConstantBias>(B(num_poses_));

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

  // Print progress to screen
  static int progress = 0;
  static char timer[] = {'\\', '-', '/','|'};
  cout << timer[++progress % 4] << "\n";
//  cout << "\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A";
  cout << "q:\t" << current_estimated_state_.pose().rotation().quaternion().transpose() << "          \n";
  cout << "p:\t" << current_estimated_state_.pose().translation().transpose() << "          \n";
  cout << "v:\t" << current_estimated_state_.v().transpose() << "            \n";
  cout << "K:\t" << camera_intrinsics.vector().transpose() << "           \n";
  cout << "B:\t" << imu_bias.vector().transpose() << "          \n";
  cout << "C_r:\t" << camera_transform.rotation().quaternion().transpose() << "         \n";
  cout << "C_t:\t" << camera_transform.translation().transpose() << "           \n\n";
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "camera_imu_calibrator");
  Calibrator thing;
  ros::spin();
  return 0;
}

