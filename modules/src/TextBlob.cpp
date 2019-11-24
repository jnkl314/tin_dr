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

#include "TextBlob.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {

TextBlob::TextBlob()
{

}

TextBlob::~TextBlob()
{

}

cv::Matx33f TextBlob::compute_rotation_homography(const float i_angle_deg)
{
    constexpr float deg2Rad = CV_PI/180.f;
    const float angle_rad = i_angle_deg * deg2Rad;
    cv::Matx33f H = cv::Matx33f::eye();
    H(0, 0) =  std::cos(angle_rad);
    H(0, 1) = -std::sin(angle_rad);
    H(1, 0) =  std::sin(angle_rad);
    H(1, 1) =  std::cos(angle_rad);

    return H;
}

cv::Size TextBlob::compute_rotated_size(const cv::Matx33f& i_H, const cv::Size i_srcSize, const cv::RotatedRect i_rotatedRect)
{
    // Compute new size
//    cv::Matx31f tl = i_H * cv::Matx31f(0.,              0.,               1.);
//    cv::Matx31f tr = i_H * cv::Matx31f(i_srcSize.width, 0.,               1.);
//    cv::Matx31f br = i_H * cv::Matx31f(i_srcSize.width, i_srcSize.height, 1.);
//    cv::Matx31f bl = i_H * cv::Matx31f(0.,              i_srcSize.height, 1.);
    cv::Point2f points[4];
    //The order is bottomLeft, topLeft, topRight, bottomRight.
    i_rotatedRect.points(points);
    cv::Matx31f bl = i_H * cv::Matx31f(points[0].x, points[0].y, 1.);
    cv::Matx31f tl = i_H * cv::Matx31f(points[1].x, points[1].y, 1.);
    cv::Matx31f tr = i_H * cv::Matx31f(points[2].x, points[2].y, 1.);
    cv::Matx31f br = i_H * cv::Matx31f(points[3].x, points[3].y, 1.);

    std::vector<cv::Point2f> vertices;
    vertices.push_back(cv::Point2f(tl(0), tl(1)));
    vertices.push_back(cv::Point2f(tr(0), tr(1)));
    vertices.push_back(cv::Point2f(br(0), br(1)));
    vertices.push_back(cv::Point2f(bl(0), bl(1)));

    cv::Size rotatedSize = cv::boundingRect(vertices).size();

    return rotatedSize;
}

void TextBlob::apply_centeredRotation(const cv::Mat&     i_fullImageSrc,
                                            cv::Mat&     o_roiDst,
                                      const cv::Matx33f& i_rotation_homography,
                                      const cv::Point    i_roi_topLeft_inFullImage,
                                      const cv::Size     i_size_roiSrc,
                                      const cv::Size     i_size_roiDst)
{


    const float Cx_dst = (i_size_roiDst.width - 1.f)/2;
    const float Cy_dst = (i_size_roiDst.height - 1.f)/2;
    const float Cx_src = (i_size_roiSrc.width - 1.f)/2;
    const float Cy_src = (i_size_roiSrc.height - 1.f)/2;

    auto& H = i_rotation_homography;
    cv::Mat_<float> mapx, mapy;
    mapx = cv::Mat_<float>(i_size_roiDst);
    mapy = cv::Mat_<float>(i_size_roiDst);
    for(int r = 0 ; r < i_size_roiDst.height ; ++r) {
        for(int c = 0 ; c < i_size_roiDst.width ; ++c) {

            const float x = H(0,0)*(c - Cx_dst) + H(0,1)*(r - Cy_dst) + H(0,2);
            const float y = H(1,0)*(c - Cx_dst) + H(1,1)*(r - Cy_dst) + H(1,2);
            const float z = H(2,0)*(c - Cx_dst) + H(2,1)*(r - Cy_dst) + H(2,2);

            mapx(r, c) = i_roi_topLeft_inFullImage.x + x/z + Cx_src;
            mapy(r, c) = i_roi_topLeft_inFullImage.y + y/z + Cy_src;
        }
    }

    cv::remap(i_fullImageSrc, o_roiDst, mapx, mapy, cv::INTER_CUBIC, cv::BORDER_REFLECT101);
}

int TextBlob::extractTextBlob(const cv::Mat& i_image, const cv::RotatedRect i_rotatedRect, TextBlob &o_textBlob)
{
    const float angle = i_rotatedRect.angle;
    cv::Rect bbRoI = i_rotatedRect.boundingRect();

    // Compute effective RoI
    cv::Rect effectiveRoI = bbRoI;
    {
        effectiveRoI.x = std::max(effectiveRoI.x, 0);
        effectiveRoI.y = std::max(effectiveRoI.y, 0);
        effectiveRoI.width  = std::min(i_image.cols - effectiveRoI.x, effectiveRoI.width);
        effectiveRoI.height = std::min(i_image.rows - effectiveRoI.y, effectiveRoI.height);
    }
    // Extract orignal Image
    cv::Mat original_image = i_image(effectiveRoI);


    // Compute Rotation Homography and destination size
    cv::Matx33f H = compute_rotation_homography(angle);
    cv::Size sizeSrc = bbRoI.size();
    cv::Size sizeDst = compute_rotated_size(H, sizeSrc, i_rotatedRect);

    // Rotate RoI
    cv::Mat upright_image;
    apply_centeredRotation(i_image, upright_image,
                           H, bbRoI.tl(),
                           bbRoI.size(),
                           sizeDst);


    // Assign images and rotatedRect to output
    o_textBlob.m_original_rotated_boundingBox = i_rotatedRect;
    o_textBlob.m_original_RoI                 = effectiveRoI;
    o_textBlob.m_original_image               = original_image;
    o_textBlob.m_upright_image                = upright_image;


    return 0;
}

static double computeMedian(const cv::Mat_<double>& i_mat)
{
    double medianValue = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> valueVector(i_mat.begin(), i_mat.end());

    if(i_mat.empty()) {
        return medianValue;
    }

//    std::list<double> valueVector;
//    for(auto& v : i_mat) {
//        valueVector.push_back(v);
//    }
//    valueList.sort();

    if(valueVector.size() % 2 == 0) {
        const auto median_it1 = valueVector.begin() + valueVector.size() / 2 - 1;
        const auto median_it2 = valueVector.begin() + valueVector.size() / 2;

        std::nth_element(valueVector.begin(), median_it1 , valueVector.end());
        const auto e1 = *median_it1;

        std::nth_element(valueVector.begin(), median_it2 , valueVector.end());
        const auto e2 = *median_it2;

        medianValue = (e1 + e2) / 2;

    } else {
        const auto median_it = valueVector.begin() + valueVector.size() / 2;
        std::nth_element(valueVector.begin(), median_it , valueVector.end());
        medianValue = *median_it;
    }

    return medianValue;
}

static cv::Scalar compute_mean_stddev_median(const cv::Mat_<double>& i_mat)
{
    cv::Scalar mean, stddev;
    double median;

    cv::meanStdDev(i_mat,  mean, stddev);
    median = computeMedian(i_mat);

    cv::Scalar msm = cv::Scalar(mean(0), stddev(0), median);

    return msm;
}

void TextBlob::filter_TextBlobs(std::vector<TextBlob> &io_textBlobs)
{
    const uint N = io_textBlobs.size();
    // Compute statistics on the blobs
    cv::Mat_<double> w_h_srdo_y(4, N); // Width, Height, Sum Row Distance to Others, Y

    for(uint i = 0 ; i < N ; ++i) {
        cv::Mat& blob = io_textBlobs[i].m_upright_image;
        w_h_srdo_y(0, i) = blob.cols;
        w_h_srdo_y(1, i) = blob.rows;
        w_h_srdo_y(2, i) = 0.;
        for(uint j = 0 ; j < N ; ++j) {
            if(j != i) {
                const double absdiff = std::abs(io_textBlobs[i].m_original_rotated_boundingBox.center.y -
                                                io_textBlobs[j].m_original_rotated_boundingBox.center.y);
                w_h_srdo_y(2, i) += absdiff;
            }
        }
        w_h_srdo_y(3, i) = io_textBlobs[i].m_original_rotated_boundingBox.center.y;
    }

    // Compute means, variances and median
    cv::Scalar msm[4];
    // Width
    for(int i = 0 ; i < 4 ; ++i) {
        msm[i] = compute_mean_stddev_median(w_h_srdo_y.row(i));
    }

    // Keep only blobs which have all three stats in [-2*stddev;+2*stddev]
    std::vector<TextBlob> kept_textBlobs;
    for(uint n = 0 ; n < N ; ++n) {
        bool keep = true;
        for(uint i = 0 ; i < 4 ; ++i) {
            const double absdiff = std::abs(msm[i](2) - w_h_srdo_y(i, n));
            if(absdiff > msm[i](1)) {
                keep = false;
                break;
            }
        }
        if(keep) {
            kept_textBlobs.push_back(io_textBlobs[n]);
        }
    }
    io_textBlobs.swap(kept_textBlobs);
}

} /* namespace TIN_DR */
