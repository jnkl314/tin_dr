/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        TextDetection_Algo.cpp

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
/* Defines                                                                  */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace TIN_DR {

TextDetection_Algo::TextDetection_Algo(const std::string i_eastModelPath,
                                       const float       i_confidenceThreshold,
                                       const float       i_nmsThreshold,
                                       const bool        i_useSlidingWindow,
                                       const int         i_padding)
    : m_confidenceThreshold(i_confidenceThreshold),
      m_nmsThreshold(i_nmsThreshold),
      m_useSlidingWindow(i_useSlidingWindow),
      m_padding(i_padding)
{
    try {
        logging_info("Read net : " << i_eastModelPath);
        m_eastNet = cv::dnn::readNet(i_eastModelPath);
    } catch(cv::Exception& e) {
        logging_error("Failed to read dnn net : " << i_eastModelPath << std::endl << e.msg);
        return;
    }

    m_model_outNames.resize(2);
    m_model_outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    m_model_outNames[1] = "feature_fusion/concat_3";

    m_isInitialized = true;
}

TextDetection_Algo::~TextDetection_Algo()
{

}

int TextDetection_Algo::decode(const cv::Mat&                i_scores,
                               const cv::Mat&                i_geometry,
                               float                         i_scoreThresh,
                               std::vector<cv::RotatedRect>& o_detections,
                               std::vector<float>&           o_confidences)
{
    o_detections.clear();
    o_confidences.clear();
    try {
        CV_Assert(i_scores.dims == 4); CV_Assert(i_geometry.dims == 4); CV_Assert(i_scores.size[0] == 1);
        CV_Assert(i_geometry.size[0] == 1); CV_Assert(i_scores.size[1] == 1); CV_Assert(i_geometry.size[1] == 5);
        CV_Assert(i_scores.size[2] == i_geometry.size[2]); CV_Assert(i_scores.size[3] == i_geometry.size[3]);
    } catch(cv::Exception &e) {
        logging_error("Assertions failed" << std::endl << e.msg);
        return -1;
    }
    const int height = i_scores.size[2];
    const int width = i_scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = i_scores.ptr<float>(0, 0, y);
        const float* x0_data = i_geometry.ptr<float>(0, 0, y);
        const float* x1_data = i_geometry.ptr<float>(0, 1, y);
        const float* x2_data = i_geometry.ptr<float>(0, 2, y);
        const float* x3_data = i_geometry.ptr<float>(0, 3, y);
        const float* anglesData = i_geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < i_scoreThresh)
                continue;
            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];
            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            o_detections.push_back(r);
            o_confidences.push_back(score);
        }
    }

    return 0;
}

int TextDetection_Algo::detect(const cv::Mat &i_src, std::vector<cv::RotatedRect> &o_detectedText_boundingBox)
{
    int res(0);

    o_detectedText_boundingBox.clear();

    if(i_src.size() != m_model_inputSize) {
        logging_error("i_src must be " << m_model_inputSize);
        return -1;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(i_src, blob, 1.0, cv::Size(), m_model_mean, true, false);
    m_eastNet.setInput(blob);

    std::vector<cv::Mat> outs;

    m_eastNet.forward(outs, m_model_outNames);

    cv::Mat scores = outs[0];
    cv::Mat geometry = outs[1];
    // Decode predicted bounding boxes.
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    res = decode(scores, geometry, m_confidenceThreshold, boxes, confidences);
    if(0 > res) {
        logging_error("decode() failed.");
        return -1;
    }
    // Apply non-maximum suppression procedure.
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, m_confidenceThreshold, m_nmsThreshold, indexes);

    // Populate o_detectedText_boundingBox
    o_detectedText_boundingBox.reserve(indexes.size());
    for(auto& index : indexes) {
        // Add padding
        cv::RotatedRect rr = boxes[index];
        rr.size.width  += 2 *m_padding;
        rr.size.height += 2 *m_padding;
        // Copy box
        o_detectedText_boundingBox.push_back(rr);
    }

    return 0;
}


float TextDetection_Algo::compute_overlap(const cv::RotatedRect& i_a, const cv::RotatedRect& i_b)
{
    // Check first if the smallest BB is included into the largest
    const auto bb_a = i_a.boundingRect();
    const auto bb_b = i_b.boundingRect();
    const int area_a = bb_a.area();
    const int area_b = bb_b.area();
    const cv::Rect *smallest, *largest;
    if(area_a > area_b) {
        largest  = &bb_a;
        smallest = &bb_b;
    } else {
        largest  = &bb_b;
        smallest = &bb_a;
    }

    if(largest->contains(smallest->tl()) && largest->contains(smallest->br())) {
        return 1.f;
    }

    // Otherwise, compute overlap
    const int commonArea = (bb_a & bb_b).area();
    return (0 < commonArea ? 2.f*commonArea/(area_a+area_b) : 0.f);
}


cv::RotatedRect TextDetection_Algo::merge_a_and_b(const cv::RotatedRect& i_a, const cv::RotatedRect& i_b)
{
    cv::RotatedRect merged;
    merged.angle  = 0.5f * (i_a.angle + i_b.angle);
    merged.center = 0.5f * (i_a.center + i_b.center);

    // The order is bottomLeft, topLeft, topRight, bottomRight.
    cv::Point2f pts_a[4];
    i_a.points(pts_a);
    cv::Point2f pts_b[4];
    i_b.points(pts_b);
    cv::Point2f pts_m[4];
    // Bottom Left
    pts_m[0].x = std::min(pts_a[0].x, pts_b[0].x);
    pts_m[0].y = std::max(pts_a[0].y, pts_b[0].y);
    // Top Left
    pts_m[1].x = std::min(pts_a[1].x, pts_b[1].x);
    pts_m[1].y = std::min(pts_a[1].y, pts_b[1].y);
    // Top Right
    pts_m[2].x = std::max(pts_a[2].x, pts_b[2].x);
    pts_m[2].y = std::min(pts_a[2].y, pts_b[2].y);
    // Bottom Right
    pts_m[3].x = std::min(pts_a[3].x, pts_b[3].x);
    pts_m[3].y = std::min(pts_a[3].y, pts_b[3].y);

    int width  = std::max(cv::norm(pts_m[1] - pts_m[2]), cv::norm(pts_m[0] - pts_m[3]));
    int height = std::max(cv::norm(pts_m[0] - pts_m[1]), cv::norm(pts_m[2] - pts_m[3]));
    merged.size = cv::Size(width, height);

    return merged;
}

int TextDetection_Algo::run(const cv::Mat &i_src, std::vector<cv::RotatedRect>& o_detectedText_boundingBox)
{
    int res(0);

    o_detectedText_boundingBox.clear();

    if(m_useSlidingWindow) {
        // First, if one dimension is less than 320, extend the original image with a reflection101
        cv::Mat image;
        const int& width = m_model_inputSize.width;
        const int& height = m_model_inputSize.height;

        if(width > i_src.cols || height > i_src.rows) {
            int bottomExtension  = height > i_src.rows ? height - i_src.rows : 0;
            int rightExtension = width  > i_src.cols ? width  - i_src.cols : 0;
            cv::copyMakeBorder(i_src, image, 0, bottomExtension, 0, rightExtension, cv::BORDER_REFLECT101);
        } else {
            image = i_src;
        }

        // Create vector of windows (cv::Mat for the RoI and cv::Point for the position offset)
        std::vector<std::pair<cv::Mat, cv::Point>> windowImages;
        // If the image, after extension, is exactly 320x320
        if(image.size() == m_model_inputSize) {
            // Then there is only one image in the vector
            windowImages.push_back({image, cv::Point(0, 0)});
        }
        // Otherwise, create all RoIs
        else {
            constexpr int coverSize = 96;
            for(int y = 0 ; y < image.rows ; y+=height) {
                // Remove coverSize
                y = 0 == y ? y : y-coverSize;
                // Ensure y stays in the image boundaries

                const int y_eff = (y+height) <= image.rows ? y : image.rows-height;
                for(int x = 0 ; x < image.cols ; x+=width) {

                    // Remove coverSize
                    x = 0 == x ? x : x-coverSize;

                    // Ensure x stays in the image boundaries
                    const int x_eff = (x+width)  <= image.cols ? x : image.cols-width ;

//                    cv::Mat roi = image.rowRange(y_eff, height).colRange(x_eff, width);
                    cv::Mat roi = image(cv::Rect(x_eff, y_eff, width, height));

                    windowImages.push_back({roi, cv::Point(x_eff, y_eff)});

                    x = x_eff;
                }
                y = y_eff;
            }
        }

        for(auto& pair : windowImages) {
            cv::Mat&   roi    = pair.first;
            cv::Point& offset = pair.second;


            std::vector<cv::RotatedRect> boxes;
            res = detect(roi, boxes);
            if(0 > res) {
                logging_error("detect() failed.");
            }

//            {
//                cv::Mat visu = roi.clone();
//                TextDetection::draw(visu, boxes);
//                cv::imshow("roi", visu);
//                cv::waitKey(0);
//            }

            // Apply offset on boxes
            for(auto& box : boxes) {
                // Apply ratio
                box.center.x    += offset.x;
                box.center.y    += offset.y;
            }

            constexpr float overlap_threshold = 0.5f;

            std::vector<cv::RotatedRect> remaining_boxes;
            // Merge boxes with o_detectedText_boundingBox when their overlap is greater than overlap_threshold (0.5)
            for(auto& newBox : boxes) {
                bool hasBeenMerged = false;
                for(auto& existingBox : o_detectedText_boundingBox) {
                    const float overlap = compute_overlap(newBox, existingBox);

                    // If one is included in the other
                    if(1e-5f > (1.f - overlap)) {
                        // Keep only the largest
                        if(existingBox.boundingRect().area() < newBox.boundingRect().area()) {
                            existingBox = newBox;
                        }
                        hasBeenMerged = true;
                        break;
                    }
                    // If the overlap is above a threshold
                    if(overlap_threshold < overlap) {
                        existingBox = merge_a_and_b(newBox, existingBox);

                        hasBeenMerged = true;
                        break;
                    }
                }
                // Otherwise, push the newBox in the vector of remaining boxes
                if(!hasBeenMerged) {
                    remaining_boxes.push_back(newBox);
                }
            }

            // Append the remaining boxes to o_detectedText_boundingBox
            for(auto& newBox : remaining_boxes) {
                o_detectedText_boundingBox.push_back(newBox);
            }

        }

    }
    // Resize input
    else {
        cv::Mat image;
        cv::resize(i_src, image, m_model_inputSize, 0, 0, cv::INTER_AREA);
        res = detect(image, o_detectedText_boundingBox);
        if(0 > res) {
            logging_error("detect() failed.");
        }
        // Apply ratio on o_detectedText_boundingBox
        cv::Point2f ratio(static_cast<float>(i_src.cols) / m_model_inputSize.width,
                          static_cast<float>(i_src.rows) / m_model_inputSize.height);
        for(auto& box : o_detectedText_boundingBox) {
            // Apply ratio
            box.center.x    *= ratio.x;
            box.center.y    *= ratio.y;
            box.size.width  *= ratio.x;
            box.size.height *= ratio.y;
        }
    }


    return 0;
}

} /* namespace TIN_DR */
