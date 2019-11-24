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
#include <TextRecognition.hpp>

////// APPLICATION ARGUMENTS //////
struct {
    std::string inputPath;

} typedef CmdArguments;

int initializeAndParseArguments(int argc, char **argv, CmdArguments& o_cmdArguments)
{
    ////*** Beginning of Arguments Handling ***////

    // Create and attach TCLAP arguments to cmd
    TCLAP::CmdLine cmd("TextRecognition", ' ', "1.0");
    std::vector<std::shared_ptr<TCLAP::Arg> > tclap_args;
    // Add some custom parameter
    try {
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("i", "inputPath",
                                                                                          "Path to an image",
                                                                                          true, "", "string", cmd)));

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
        cv::imshow("image", image);

        // Run recognition
        logging_info("Run text recognition ...");
        if(3 == image.channels()) {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }
        std::string text;
        res = TR->run(image, text);
        if(0 > res) {
            logging_error("TD->run() failed.");
            return EXIT_FAILURE;
        }

        // Display text
        std::cout << text << std::endl;


        logging_info("Push ESC or Q to exit");
        int key = 0;
        while(27 != key && 'q' != key){
            key = cv::waitKey(10) & 0xff;
        }
    }


    // Manually reset (and delete content of) pointer
    TR.reset();

    return res;
}
