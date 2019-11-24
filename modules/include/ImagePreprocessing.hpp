/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        ImagePreprocessing.hpp

 */
/*============================================================================*/

#ifndef IMAGEPREPROCESSING_HPP_
#define IMAGEPREPROCESSING_HPP_

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
class ImagePreprocessing {
public:

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run preprocessing on image to prepare it for text detection
     * @param[in] 		i_src : Input image
     * @param[out] 		o_dst : output image
     *
     */
    /*============================================================================*/
    static int preprocess_forTextDetection(cv::InputArray i_src, cv::OutputArray o_dst);
};

} /* namespace TIN_DR */
#endif /* IMAGEPREPROCESSING_HPP_ */