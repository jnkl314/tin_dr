/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        main.cpp

 */
/*============================================================================*/
#include <iostream>
#include <thread>
#include <cmath>
#include <chrono>
#include <ctime>

#include <tclap/CmdLine.h>

#include <Utils_Logging.hpp>
#include <ImagePreprocessing.hpp>
#include <TextDetection.hpp>
#include <TextBlob.hpp>
#include <TextRecognition.hpp>


////// APPLICATION ARGUMENTS //////
struct {
    std::string inputPath;
    std::string eastModelPath;
    float       confidenceThreshold;
    float       nmsThreshold;
    bool        useSlidingWindow;
    int         padding;
    std::string outputPath;

} typedef CmdArguments;

int initializeAndParseArguments(int argc, char **argv, CmdArguments& o_cmdArguments)
{
    ////*** Beginning of Arguments Handling ***////

    // Create and attach TCLAP arguments to cmd
    TCLAP::CmdLine cmd("TextDetection", ' ', "1.0");
    std::vector<std::shared_ptr<TCLAP::Arg> > tclap_args;
    // Add some custom parameter
    try {
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("i", "inputPath",
                                                                                          "Path to an image",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("m", "eastModelPath",
                                                                                          "Path to a binary .pb containing the trained network from :\n"
                                                                                          "https://github.com/argman/EAST => "
                                                                                          "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<float>       ("c", "confidenceThreshold",
                                                                                          "Confidence threshold",
                                                                                          false, 0.5f, "float", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<float>       ("n", "nmsThreshold",
                                                                                          "Non-maximum suppression threshold",
                                                                                          false, 0.3f, "float", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::SwitchArg             ("s", "useSlidingWindow",
                                                                                          "Instead of resizing the input image to 320x320, use a sliding window and merge detections",
                                                                                          cmd, false)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>         ("p", "padding",
                                                                                          "Add padding to detected area",
                                                                                          false, 0, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("o", "outputPath",
                                                                                          "Path to a directory to save detected text patches",
                                                                                          false, "", "string", cmd)));

    } catch(TCLAP::ArgException &e) {  // catch any exceptions
        logging_error("Failed to create TCLAP arguments" << std::endl <<
                      "TCLAP error: " << e.error() << " for arg " << e.argId());
        return -1;
    }

    // Parse all arguments
    try {
        cmd.setExceptionHandling(false);
        cmd.parse(argc, argv);
    } catch(TCLAP::ArgException &e) {
        logging_error("Failed to parse tclap arguments" << std::endl <<
                     "TCLAP error: " << e.error() << " for arg " << e.argId());
        return -1;
    } catch(TCLAP::ExitException &e) {
        exit(0);
    }

    ////*** End of Arguments Handling ***////

    // Dispatch arguments value in o_cmdArguments
    uint idx = 0;
    // Fetch custom argument value
    o_cmdArguments.inputPath           = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.eastModelPath       = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.confidenceThreshold = dynamic_cast<TCLAP::ValueArg<float>*>      (tclap_args[idx++].get())->getValue();
    o_cmdArguments.nmsThreshold        = dynamic_cast<TCLAP::ValueArg<float>*>      (tclap_args[idx++].get())->getValue();
    o_cmdArguments.useSlidingWindow    = dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue();
    o_cmdArguments.padding             = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    o_cmdArguments.outputPath          = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();

    return 0;
}



////// MAIN //////
int main(int argc, char **argv)
{
    int res(0);

    CmdArguments cmdArguments;

    res = initializeAndParseArguments(argc, argv, cmdArguments);
    if(0 > res) {
        logging_error("initializeAndParseArguments() failed");
        return EXIT_FAILURE;
    }

    // Create and initialize TextDetection
    std::unique_ptr<TIN_DR::TextDetection> TD(new TIN_DR::TextDetection(cmdArguments.eastModelPath,
                                                                        cmdArguments.confidenceThreshold,
                                                                        cmdArguments.nmsThreshold,
                                                                        cmdArguments.useSlidingWindow,
                                                                        cmdArguments.padding));
    if(false == TD->get_isInitialized()) {
        logging_error("TIN_DR::TextDetection was not correctly initialized");
        return EXIT_FAILURE;
    }

    // Create and initialize TextRecognition
    std::unique_ptr<TIN_DR::TextRecognition> TR(new TIN_DR::TextRecognition());
    if(false == TR->get_isInitialized()) {
        logging_error("TIN_DR::TextRecognition was not correctly initialized");
        return EXIT_FAILURE;
    }

    // Run Text Detection
    {
        // Load image
        logging_info("Read image : " << cmdArguments.inputPath);
        cv::Mat image = cv::imread(cmdArguments.inputPath);
        if(image.empty()) {
            logging_error("Failed to read image : " << cmdArguments.inputPath);
            return EXIT_FAILURE;
        } else {
            logging_info("Image of type " << cv::typeToString(image.type()) << " and size " << image.size());
        }

        // Apply some preprocessing on image
        logging_info("Apply preprocessing on input image ...");
        cv::Mat image_preprocessed;
        res = TIN_DR::ImagePreprocessing::preprocess_forTextDetection(image, image_preprocessed);
        if(0 > res) {
            logging_error("TIN_DR::ImagePreprocessing::preprocess_forTextDetection() failed.");
            return EXIT_FAILURE;
        }
//        cv::imshow("image_preprocessed", image_preprocessed);

        // Run detection
        logging_info("Run text detection ...");
        std::vector<cv::RotatedRect> detectedText_boundingBox;
        res = TD->run(image_preprocessed, detectedText_boundingBox);
        if(0 > res) {
            logging_error("TD->run() failed.");
            return EXIT_FAILURE;
        }

        // Draw detection
        cv::Mat resultImage = image.clone();
        TIN_DR::TextDetection::draw(resultImage, detectedText_boundingBox);


        // Extract TextBlobs from the image
        logging_info("Extract text blobs ...");
        std::vector<TIN_DR::TextBlob> textBlobs;
        textBlobs.reserve(detectedText_boundingBox.size());
        for(auto& box : detectedText_boundingBox) {
            TIN_DR::TextBlob tb;
            if(0 == TIN_DR::TextBlob::extractTextBlob(image_preprocessed, box, tb)) {
                textBlobs.push_back(tb);
            }
        }

        TIN_DR::TextBlob::filter_TextBlobs(textBlobs);


        for(auto& textBlob : textBlobs) {

            // Run recognition
            logging_info("Run text recognition ...");
            cv::Mat& im = textBlob.m_upright_image;
            cv::Mat image_mono;
            if(3 == im.channels()) {
                cv::cvtColor(im, image_mono, cv::COLOR_BGR2GRAY);
            } else {
                image_mono = im;
            }
            std::string text;
            res = TR->run(image_mono, text, TIN_DR::TextRecognition::AlphaNum);
            if(0 > res) {
                logging_error("TD->run() failed.");
                return EXIT_FAILURE;
            }

            // Write text on image
            cv::putText(resultImage, text, textBlob.m_original_RoI.tl(), cv::FONT_HERSHEY_COMPLEX, 2.0, cv::Scalar(0, 0, 127), 2);
        }


        // Display detection
        cv::imshow("Result", resultImage);


        logging_info("Push ESC or Q to exit");
        int key = 0;
        while(27 != key && 'q' != key){
            key = cv::waitKey(10) & 0xff;
        }

        if(!cmdArguments.outputPath.empty()) {
            logging_info("Saving results in : " << cmdArguments.outputPath);
//            auto now = std::chrono::system_clock::now();
//            std::time_t now_time = std::chrono::system_clock::to_time_t(now);
//            auto now_str = std::ctime(&now_time);

            std::string now_str;
            {
                std::ostringstream oss;
                std::time_t t = std::time(0);   // get time now
                std::tm* now = std::localtime(&t);
                oss << std::setfill('0')
                    << (now->tm_year + 1900) << "_"
                    << std::setw(2)
                    << (now->tm_mon + 1) << "_"
                    <<  now->tm_mday << "_"
                     << now->tm_hour << "h"
                     << now->tm_min << "m"
                     << now->tm_sec << "s";
                now_str = oss.str();
            }

            int cnt = 0;
            for(auto& textBlob : textBlobs) {
                std::ostringstream oss;
                oss << cmdArguments.outputPath << "/" << now_str << "_" << std::setw(4) << std::setfill('0') << cnt++ << ".png";
                std::string path = oss.str();
                logging_info("Writing : " << path);
                cv::imwrite(path, textBlob.m_upright_image);
            }
        }
    }


    // Manually reset (and delete content of) pointer
    TD.reset();

    return res;
}
