/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextDetection.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "ImagePreprocessing.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {

int ImagePreprocessing::preprocess_forTextDetection(cv::InputArray i_src, cv::OutputArray o_dst)
{
    if(i_src.empty()) {
        logging_error("i_src cannot be empty");
        return -1;
    }
    if(3 != i_src.channels()) {
        logging_error("i_src must have 3 channels");
        return -1;
    }
    if(CV_8U != i_src.depth()) {
        logging_error("i_src must have a 8bits detph (CV_8U)");
        return -1;
    }

    cv::Mat src = i_src.getMat();
    o_dst.create(i_src.size(), i_src.type());
    cv::Mat& dst = o_dst.getMatRef();

    // Apply local contrast enhancement
    cv::Mat image_lce; // lce for local contrast enhancement
    {
        // Switch to HSV
        cv::Mat hsv;
        cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

        // Split channels
        std::vector<cv::Mat> hsv_split;
        cv::split(hsv, hsv_split);

        // Work on V
        cv::Mat lum = hsv_split[2];
        {
            // Apply local histogram equalization
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(6, cv::Size(16, 16));
            cv::Mat enhancedLum;
            clahe->apply(lum, enhancedLum);

            hsv_split[2] = enhancedLum;
        }

        // Merge back channels
        cv::merge(hsv_split, hsv);

        // Switch back to BGR
        cv::cvtColor(hsv, image_lce, cv::COLOR_HSV2BGR);
    }

    image_lce.copyTo(dst);

    return 0;
}

} /* namespace TIN_DR */
