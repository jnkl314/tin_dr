/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextDetection.hpp

 */
/*============================================================================*/

#ifndef TEXTDETECTION_HPP_
#define TEXTDETECTION_HPP_

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
/* Forward Declaration                                                        */
/*============================================================================*/
class TextDetection_Algo;

/*============================================================================*/
/* Class Description                                                       */
/*============================================================================*/
/**
 * 	\brief       Class performing ...
 *
 */
/*============================================================================*/
class TextDetection {
private:
    std::unique_ptr<TextDetection_Algo> m_algo;

public:

    TextDetection(const std::string i_eastModelPath,
                  const float       i_confidenceThreshold,
                  const float       i_nmsThreshold,
                  const bool        i_useSlidingWindow,
                  const int         i_padding);
    virtual ~TextDetection();

    bool get_isInitialized();

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run the text detection
     * @param[in] 		i_src                       : Input image to process
     * @param[out] 		o_detectedText_boundingBox	: Vector of bounding box of the detected text area
     *
     */
    /*============================================================================*/
    int run(cv::InputArray i_src, std::vector<cv::RotatedRect>& o_detectedText_boundingBox);

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Draw detected text on image
     * @param[in,out]   io_image                     : Image to draw on
     * @param[in] 		i_detectedText_boundingBox	 : Vector of bounding box of the detected text area
     *
     */
    /*============================================================================*/
    static void draw(cv::Mat &io_image, const std::vector<cv::RotatedRect>& i_detectedText_boundingBox, const cv::Scalar i_color_bgr = cv::Scalar(0, 0, 127));

};

} /* namespace TIN_DR */
#endif /* TEXTDETECTION_HPP_ */
