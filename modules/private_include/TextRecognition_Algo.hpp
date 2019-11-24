/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextRecognition_Algo.hpp

 */
/*============================================================================*/

#ifndef TEXTRECOGNITION_ALGO_H_
#define TEXTRECOGNITION_ALGO_H_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <opencv2/text.hpp>

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
class TextRecognition_Algo {
private:
    bool m_isInitialized = false;
    std::vector<cv::Ptr<cv::text::OCRTesseract>> m_ocr;

    void extract_characters(const cv::Mat &i_src, std::vector<cv::Mat>& o_characterPatches);
    void classify_characters(std::vector<cv::Mat>& io_characterPatches, std::string& o_text, const int i_subset);
public:

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run the text detection
     *
     */
    /*============================================================================*/
    TextRecognition_Algo();
    virtual ~TextRecognition_Algo();

    bool get_isInitialized() {
        return m_isInitialized;
    }


    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run the text recognition
     * @param[in] 		i_src  : Input image to process
     * @param[out] 		o_text : Text
     *
     */
    /*============================================================================*/
    int run(const cv::Mat &i_src, std::string& o_text, const int i_subset);

};

} /* namespace TIN_DR */
#endif /* TEXTRECOGNITION_ALGO_H_ */
