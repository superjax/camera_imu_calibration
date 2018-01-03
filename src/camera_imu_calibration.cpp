#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{


    VideoCapture input_video;
    input_video.open(0);
    if (!input_video.isOpened())
    {
        cout << "no camera at /dev/video0\n";
        return -1;
    }
    else
    {
        cout << "found camera\n";
    }

    Mat camMatrix;
    Mat distCoeffs;

    // ELP Web Cam
//    camMatrix = (Mat1d(3, 3) <<  469.462257, 0.000000, 325.442775,
//                                 0.000000, 470.773506, 230.956417,
//                                 0.000000, 0.000000, 1.000000);
//    distCoeffs = (Mat1d(1, 5) << -0.390234, 0.136303, -0.002855, 0.002160, 0.000000);

    // Create a bunch of aruco objects
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(aruco::DICT_6X6_50));

    // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create(5, 7, 0.0351f, 0.0222f, dictionary);
    Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

    // length of axis when drawing pose
    float axisLength = 0.5f * (5 * 0.0351);

    bool draw_rejected = false;


    while(input_video.grab())
    {
        Mat image, imageCopy;
        input_video.retrieve(image);

        vector<int> markerIds, charucoIds;
        vector<vector<Point2f>> markerCorners, rejectedMarkers;
        vector<Point2f> charucoCorners;
        Vec3d rvec, tvec;

        // detect markers
        aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams,
                             rejectedMarkers);

        // refind strategy to detect more markers
        aruco::refineDetectedMarkers(image, board, markerCorners, markerIds, rejectedMarkers,
                                         camMatrix, distCoeffs);

        // interpolate charuco corners
        int interpolatedCorners = 0;
        if(markerIds.size() > 0)
            interpolatedCorners =
                aruco::interpolateCornersCharuco(markerCorners, markerIds, image, charucoboard,
                                                 charucoCorners, charucoIds, camMatrix, distCoeffs);

        // estimate charuco board pose (requires camera intrinsics)
        bool validPose = false;
        if(camMatrix.total() != 0)
            validPose = aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard,
                                                        camMatrix, distCoeffs, rvec, tvec);

        // draw results
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

        if(validPose)
            aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

        imshow("out", imageCopy);
        char key = (char)waitKey(1);
        if(key == 'q') break;
    }

    return 0;
}
