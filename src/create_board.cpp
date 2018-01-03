#include <iostream>

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main()
{
    Ptr<aruco::Dictionary> dictionary =
            aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(aruco::DICT_6X6_50));
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(5, 7, (float)(203.2/5), (float)(203.2/8), dictionary);
    Mat board_img;
    board->draw(Size(1280, 720), board_img, 10, 1);
    imshow("New Image", board_img);
    waitKey(0);

    return 0;
}
