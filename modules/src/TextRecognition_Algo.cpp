/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextRecognition_Algo.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include <opencv2/text.hpp>

#include "Utils_Logging.hpp"

#include "TextRecognition.hpp"
#include "TextRecognition_Algo.hpp"

/*============================================================================*/
/* Defines                                                                  */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {

TextRecognition_Algo::TextRecognition_Algo()
{


    std::vector<std::string> whitelists(4);
    whitelists[TextRecognition::AlphaNum  ] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    whitelists[TextRecognition::Alpha     ] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    whitelists[TextRecognition::Num       ] = "0123456789";
    whitelists[TextRecognition::DotPattern] = "DOT";

    m_ocr.resize(4);
    for(uint idx = 0 ; idx < 4 ; ++idx) {
        m_ocr[idx] = cv::text::OCRTesseract::create(nullptr, nullptr,
                                                    whitelists[idx].c_str(),
                                                    cv::text::OEM_TESSERACT_ONLY,
                                                    cv::text::PSM_SINGLE_CHAR);
    }

    m_isInitialized = true;
}

TextRecognition_Algo::~TextRecognition_Algo()
{
    for(auto& ocr : m_ocr) {
        ocr.reset();
    }
}

void TextRecognition_Algo::extract_characters(const cv::Mat &i_src, std::vector<cv::Mat> &o_characterPatches)
{
    // Resize image to be 100 pixels wide
    cv::Mat image;
    const float im_width = 100;
    const float resizeRatio = im_width/i_src.cols;
    const float im_height = resizeRatio*i_src.rows;
    cv::resize(i_src, image, cv::Size(im_width, im_height), 0, 0, cv::INTER_AREA);

    // Normalize image
    cv::Mat image_normalized;
    {
        double minV, maxV;
        cv::minMaxIdx(image, &minV, &maxV);
        if(std::abs(maxV - minV) < 1e-5) {
            image.convertTo(image_normalized, CV_32F, 1./255.);
        } else {
            double alpha = 1. / (maxV - minV);
            double beta = -alpha/(maxV - minV);
            image.convertTo(image_normalized, CV_32F, alpha, beta);
        }
    }

    // cv::imshow("image_normalized", image_normalized);

    // Remove background
    {
        cv::Mat im = image_normalized.clone();
        cv::Scalar globalMean_sc, globalStddev_sc;
        cv::meanStdDev(im, globalMean_sc, globalStddev_sc);
        constexpr int kSize = 3;
        constexpr int half_kSize = kSize/2;
        for(int y = half_kSize ; y < (image_normalized.rows - half_kSize) ; ++y) {
            for(int x = half_kSize ; x < (image_normalized.cols - half_kSize) ; ++x) {
                cv::Scalar mean_sc, stddev_sc;
                cv::Mat roi = im(cv::Rect(x-half_kSize, y-half_kSize, kSize, kSize));
                cv::meanStdDev(roi, mean_sc, stddev_sc);

                if(stddev_sc.val[0] < globalStddev_sc(0)) {
                    image_normalized.at<float>(y, x) = 0;
                }
            }
        }
        image_normalized.rowRange(0, half_kSize).setTo(0);
        image_normalized.rowRange(image_normalized.rows-half_kSize, image_normalized.rows).setTo(0);

        image_normalized.colRange(0, half_kSize).setTo(0);
        image_normalized.colRange(image_normalized.cols-half_kSize, image_normalized.cols).setTo(0);
    }

//    cv::imshow("image", image);
//    cv::imshow("image_normalized", image_normalized);

    // Compute gradients
    cv::Mat grad_x, grad_y;
    cv::filter2D(image_normalized, grad_x, -1, cv::Matx13f(-0.5f, 0.f, 0.5f));
    cv::filter2D(image_normalized, grad_y, -1, cv::Matx31f(-0.5f, 0.f, 0.5f));


    // Accumulate square gradient in both directions
    cv::Mat accu_grad_y = cv::Mat::zeros(grad_x.rows, 1, CV_32F);
    for(int x = 0 ; x < grad_x.cols ; ++x) {
        accu_grad_y = accu_grad_y + grad_x.col(x).mul(grad_x.col(x));
    }
    cv::Mat accu_grad_x = cv::Mat::zeros(1, grad_y.cols, CV_32F);
    for(int y = 0 ; y < grad_y.rows ; ++y) {
        accu_grad_x = accu_grad_x + grad_y.row(y).mul(grad_y.row(y));
    }

    // Normalize accu_grad_y and accu_grad_x
    {
        double maxV;
        cv::minMaxIdx(accu_grad_x, nullptr, &maxV);
        accu_grad_x *= 1./maxV;
        cv::minMaxIdx(accu_grad_y, nullptr, &maxV);
        accu_grad_y *= 1./maxV;
    }

    // Compute square cross intensity
    cv::Mat_<float> intensity(accu_grad_y.rows, accu_grad_x.cols);
    for(int y = 0 ; y < intensity.rows ; ++y) {
        for(int x = 0 ; x < intensity.cols ; ++x) {
            const float intens = accu_grad_y.at<float>(y, 0) * accu_grad_x.at<float>(0, x);
            intensity(y, x) = intens * intens;
        }
    }

//    cv::imshow("intensity", intensity);

    // Compute local variance
    cv::Mat variance(intensity.size(), CV_32F);
    {
        constexpr int kSize = 7;
        constexpr int half_kSize = kSize/2;
        for(int y = half_kSize ; y < (intensity.rows - half_kSize) ; ++y) {
            for(int x = half_kSize ; x < (intensity.cols - half_kSize) ; ++x) {
                cv::Scalar mean_sc, stddev_sc;
                cv::Mat roi = image_normalized(cv::Rect(x-half_kSize, y-half_kSize, kSize, kSize));
                cv::meanStdDev(roi, mean_sc, stddev_sc);

                variance.at<float>(y, x) = stddev_sc(0);
            }
        }
        variance.rowRange(0, half_kSize).setTo(0);
        variance.rowRange(variance.rows-half_kSize, variance.rows).setTo(0);

        variance.colRange(0, half_kSize).setTo(0);
        variance.colRange(variance.cols-half_kSize, variance.cols).setTo(0);
    }
    // Normalize variance
    {
        double maxV;
        cv::minMaxIdx(variance, nullptr, &maxV);
        variance *= 1./maxV;
    }

//    cv::imshow("variance", variance);

    intensity = intensity.mul(variance);

//    cv::imshow("intensity .* variance", intensity);

    {

        cv::Scalar mean_sc, stddev_sc;
        cv::meanStdDev(intensity, mean_sc, stddev_sc);
//        std::cout << "Mean = " << mean_sc(0) << std::endl;
//        std::cout << "Stddev = " << stddev_sc(0) << std::endl;

        cv::threshold(intensity, intensity, stddev_sc(0), 1.f, cv::THRESH_BINARY);
    }

//    cv::imshow("intensity thresh", intensity);

    cv::Mat intensity_thresholded;
    intensity.convertTo(intensity_thresholded, CV_8U, 255.);

//    // Erode
//    cv::erode(intensity_thresholded, intensity_thresholded,
//              cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1),
//              2, cv::BORDER_CONSTANT);

//    // Dilate
    cv::dilate(intensity_thresholded, intensity_thresholded,
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1),
               2, cv::BORDER_CONSTANT);

//    cv::imshow("intensity dilate", intensity_thresholded);

    std::vector<std::vector<cv::Point>> contoursVect;
    cv::findContours(intensity_thresholded, contoursVect, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> rects;
    rects.reserve(contoursVect.size());
    constexpr int widthMin = 5;
    constexpr int heightMin = 20;

    for(auto& contours : contoursVect)
    {
        cv::Rect rect = cv::boundingRect(contours);
        if(rect.width >= widthMin && rect.height >= heightMin) {
            rects.push_back(rect);
        }
    }

    // Reorder from left to right
    {
        std::list<cv::Rect> rects_list;
        for(auto& rect : rects) {
            rects_list.push_back(rect);
        }
        rects_list.sort([](const cv::Rect& a, const cv::Rect& b)
        {
            return a.x < b.x;
        });

        rects.clear();
        rects.reserve(rects_list.size());
        for(auto& rect : rects_list) {
            rects.push_back(rect);
        }
    }

    // Add padding to Rects
    constexpr int padding = 4;
    for(auto& rect : rects) {
        rect.x = std::max(rect.x - padding, 0);
        rect.y = std::max(rect.y - padding, 0);
        rect.width  += 2*padding;
        rect.height += 2*padding;
        rect.width  = std::min(image.cols - rect.x, rect.width);
        rect.height = std::min(image.rows - rect.y, rect.height);
    }

    // Display
    // {
    //     cv::Mat display;
    //     cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    //     for(auto& rect : rects) {
    //         cv::rectangle(display, rect, cv::Scalar(0, 255, 127), 1);
    //     }

    //     cv::imshow("contours", display);
    //     while(cv::waitKey(10) & 0xff != 'n');
    // }

    // Extract character image patches
    o_characterPatches.reserve(rects.size());
    for(auto& rect : rects) {
        // Resize rect to size of i_src
        cv::Rect fullRect = rect;
        fullRect.x      /= resizeRatio;
        fullRect.y      /= resizeRatio;
        fullRect.width  /= resizeRatio;
        fullRect.height /= resizeRatio;

        // Check boundaries
        fullRect.x = std::min(fullRect.x, i_src.cols);
        fullRect.y = std::min(fullRect.y, i_src.rows);
        fullRect.width  = std::min(i_src.cols - fullRect.x, fullRect.width);
        fullRect.height = std::min(i_src.rows - fullRect.y, fullRect.height);

        o_characterPatches.push_back(i_src(fullRect));
    }

//    // Display characters
//    {
//        int cnt = 0 ;
//        for(auto& characterPatch : o_characterPatches) {
//            std::ostringstream oss;
//            oss << "Character #" << cnt++;
//            cv::imshow(oss.str(), characterPatch);
//        }
//    }
}

void TextRecognition_Algo::classify_characters(std::vector<cv::Mat>& io_characterPatches, std::string& o_text, const int i_subset)
{
    // Preprocess character patches
    for(auto& characterPatch : io_characterPatches) {

        // Remove background
        {
            cv::Mat im = characterPatch.clone();
            cv::Scalar globalMean_sc, globalStddev_sc;
            cv::meanStdDev(im, globalMean_sc, globalStddev_sc);
            constexpr int kSize = 7;
            constexpr int half_kSize = kSize/2;
            for(int y = half_kSize ; y < (characterPatch.rows - half_kSize) ; ++y) {
                for(int x = half_kSize ; x < (characterPatch.cols - half_kSize) ; ++x) {
                    cv::Scalar mean_sc, stddev_sc;
                    cv::Mat roi = im(cv::Rect(x-half_kSize, y-half_kSize, kSize, kSize));
                    cv::meanStdDev(roi, mean_sc, stddev_sc);

                    if(stddev_sc.val[0] < globalStddev_sc(0)) {
                        characterPatch.at<uint8_t>(y, x) = 0;
                    }
                }
            }
            characterPatch.rowRange(0, half_kSize).setTo(0);
            characterPatch.rowRange(characterPatch.rows-half_kSize, characterPatch.rows).setTo(0);

            characterPatch.colRange(0, half_kSize).setTo(0);
            characterPatch.colRange(characterPatch.cols-half_kSize, characterPatch.cols).setTo(0);
        }
    }



    int cnt = 0;
    std::ostringstream oss;
    for(auto& characterPatch : io_characterPatches) {

        std::vector<std::string> characters;
        std::vector<float>  confidences;
        std::string text;
        m_ocr[i_subset]->run(characterPatch, text, nullptr, &characters, &confidences, cv::text::OCR_LEVEL_WORD);

        if(!characters.empty()) {
            std::cout << "Character = " << characters[0] << " confidence = " << confidences[0] << std::endl;
            oss << characters[0];
        } else {
            oss << "_";
        }


//        std::ostringstream oss;
//        oss << "Inner Character #" << cnt++;
//        cv::imshow(oss.str(), characterPatch);
//        cv::waitKey(0);
    }

    o_text = oss.str();
}

int TextRecognition_Algo::run(const cv::Mat &i_src, std::string& o_text, const int i_subset)
{
    std::vector<cv::Mat> characterPatches;
    extract_characters(i_src, characterPatches);
    classify_characters(characterPatches, o_text, i_subset);

    return 0;
}

} /* namespace TIN_DR */
