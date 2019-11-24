/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextBlob.hpp

 */
/*============================================================================*/

#ifndef TEXTBLOB_HPP_
#define TEXTBLOB_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {


/*============================================================================*/
/* Class Description                                                       */
/*============================================================================*/
/**
 * 	\brief       Class performing ...
 *
 */
/*============================================================================*/
class TextBlob {
private:

    static void apply_centeredRotation(const cv::Mat&     i_fullImageSrc,
                                             cv::Mat&     o_roiDst,
                                       const cv::Matx33f& i_rotation_homography,
                                       const cv::Point    i_roi_topLeft_inFullImage,
                                       const cv::Size     i_size_roiSrc,
                                       const cv::Size     i_size_roiDst);

    static cv::Matx33f compute_rotation_homography(const float i_angle_deg);
    static cv::Size compute_rotated_size(const cv::Matx33f& i_H, const cv::Size i_srcSize, const cv::RotatedRect i_rotatedRect);
public:

    cv::RotatedRect m_original_rotated_boundingBox;
    cv::Rect m_original_RoI;
    cv::Mat m_original_image;
    cv::Mat m_upright_image;

    TextBlob();
    virtual ~TextBlob();


    static int extractTextBlob(const cv::Mat& i_image, const cv::RotatedRect i_rotatedRect, TextBlob &o_textBlob);
    static void filter_TextBlobs(std::vector<TextBlob> &io_textBlobs);
};

} /* namespace TIN_DR */
#endif /* TEXTBLOB_HPP_ */
