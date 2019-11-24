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

#include "TextDetection.hpp"
#include "TextDetection_Algo.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {

TextDetection::TextDetection(const std::string i_eastModelPath,
                             const float       i_confidenceThreshold,
                             const float       i_nmsThreshold,
                             const bool        i_useSlidingWindow,
                             const int         i_padding)
{
    m_algo.reset(new TextDetection_Algo(i_eastModelPath, i_confidenceThreshold, i_nmsThreshold, i_useSlidingWindow, i_padding));
}

TextDetection::~TextDetection()
{
    m_algo.reset();
}

bool TextDetection::get_isInitialized()
{
    return m_algo->get_isInitialized();
}

int TextDetection::run(cv::InputArray i_src, std::vector<cv::RotatedRect>& o_detectedText_boundingBox)
{
    int32_t res(0);

    if(false == m_algo->get_isInitialized()) {
        logging_error("This instance was not correctly initialized.");
        return -1;
    }

    // Tests on i_src
    if(i_src.empty()) {
        logging_error("i_src is empty.");
        return -1;
    }
    if(3 != i_src.channels()) {
        logging_error("i_src does not have 3 channels.");
        return -1;
    }
    if(CV_8U != i_src.depth()) {
        logging_error("i_src can only be CV_8UC3 (Depth of 8bits unsigned int).");
        return -1;
    }

    // Actual call to algorithm
    res = m_algo->run(i_src.getMat(), o_detectedText_boundingBox);
    if(0 > res) {
        logging_error("m_algo->run() failed.");
        return -1;
    }

    return res;
}

void TextDetection::draw(cv::Mat &io_image, const std::vector<cv::RotatedRect>& i_detectedText_boundingBox, const cv::Scalar i_color_bgr)
{
    if(io_image.empty()) {
        logging_warning("io_image is empty");
        return;
    }
    if(1 == io_image.channels()) {
        logging_warning("io_image is monochrome, it will be converted to BGR");
        cv::cvtColor(io_image, io_image, cv::COLOR_GRAY2BGR);
    }
    if(3 != io_image.channels()) {
        logging_warning("io_image does not have 3 channels");
        return;
    }

    for(auto& box : i_detectedText_boundingBox) {
        cv::Point2f vertices[4];
        box.points(vertices);

        for(int idx = 0; idx < 4; ++idx) {
            cv::line(io_image, vertices[idx], vertices[(idx + 1) % 4], i_color_bgr, 1);
        }
    }
}

} /* namespace TIN_DR */
