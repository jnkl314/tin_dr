/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextDetection_Algo.hpp

 */
/*============================================================================*/

#ifndef TEXTDETECTION_ALGO_H_
#define TEXTDETECTION_ALGO_H_

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

/*============================================================================*/
/* Class Description                                                          */
/*============================================================================*/
/**
 * 	\brief       Class performing
 *
 */
/*============================================================================*/
class TextDetection_Algo {
private:
    bool m_isInitialized = false;
    cv::dnn::Net m_eastNet;
    cv::Scalar m_model_mean = cv::Scalar(123.68, 116.78, 103.94);
    const cv::Size m_model_inputSize = cv::Size(320, 320);
    std::vector<cv::String> m_model_outNames;

    const float m_confidenceThreshold;
    const float m_nmsThreshold;
    const bool  m_useSlidingWindow;
    const int   m_padding;

    // Method extracted from OpenCV samples
    int decode(const cv::Mat&                i_scores,
               const cv::Mat&                i_geometry,
               float                         i_scoreThresh,
               std::vector<cv::RotatedRect>& o_detections,
               std::vector<float>&           o_confidences);

    float compute_overlap(const cv::RotatedRect& i_a, const cv::RotatedRect& i_b);
    cv::RotatedRect merge_a_and_b(const cv::RotatedRect& i_a, const cv::RotatedRect& i_b);

    int detect(const cv::Mat &i_src, std::vector<cv::RotatedRect> &o_detectedText_boundingBox);
public:

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run the text detection
     * @param[in] 		i_eastModelPath         : Path to a binary .pb containing the trained network
     * @param[in] 		i_confidenceThreshold	: Confidence threshold
     * @param[in] 		i_nmsThreshold          : Non-maximum suppression threshold
     *
     */
    /*============================================================================*/
    TextDetection_Algo(const std::string i_eastModelPath,
                       const float       i_confidenceThreshold,
                       const float       i_nmsThreshold,
                       const bool        i_useSlidingWindow,
                       const int         i_padding);
    virtual ~TextDetection_Algo();

    bool get_isInitialized() {
        return m_isInitialized;
    }


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
    int run(const cv::Mat &i_src, std::vector<cv::RotatedRect> &o_detectedText_boundingBox);

};

} /* namespace TIN_DR */
#endif /* TEXTDETECTION_ALGO_H_ */
