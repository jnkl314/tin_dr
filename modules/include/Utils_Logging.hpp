/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        Utils_Logging.hpp

 */
/*============================================================================*/

#ifndef UTILS_LOGGING_HPP_
#define UTILS_LOGGING_HPP_
#include <iostream>
#include <typeinfo>

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/
#ifdef TIN_DR_ENABLE_VERBOSE
#define logging_error(message) \
    std::cerr << "Error in " << __PRETTY_FUNCTION__ << "\n\t: " << message <<  std::endl;
#define logging_warning(message) \
    std::cerr << "Warning in " << __PRETTY_FUNCTION__ << "\n\t: " << message <<  std::endl;
#define logging_info(message) \
    std::cout << "Info in " << __PRETTY_FUNCTION__ << "\n\t: " << message <<  std::endl;
#else
#define logging_error(message)
#define logging_warning(message)
#define logging_info(message)
#endif

#endif /* UTILS_LOGGING_HPP_ */
