/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextRecognition.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "TextRecognition.hpp"
#include "TextRecognition_Algo.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {

TextRecognition::TextRecognition()
{
    m_algo.reset(new TextRecognition_Algo());
}

TextRecognition::~TextRecognition()
{
    m_algo.reset();
}

bool TextRecognition::get_isInitialized()
{
    return m_algo->get_isInitialized();
}

int TextRecognition::run(cv::InputArray i_src, std::string& o_text, const int i_subset)
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
    if(1 != i_src.channels()) {
        logging_error("i_src does not have 1 channel.");
        return -1;
    }
    if(CV_8U != i_src.depth()) {
        logging_error("i_src can only be CV_8U (Depth of 8bits unsigned int).");
        return -1;
    }
    if(i_subset < AlphaNum || i_subset > DotPattern) {
        logging_error("i_subset has an invalid value.");
        return -1;
    }

    // Actual call to algorithm
    res = m_algo->run(i_src.getMat(), o_text,i_subset);
    if(0 > res) {
        logging_error("m_algo->run() failed.");
        return -1;
    }

    return res;
}

} /* namespace TIN_DR */
