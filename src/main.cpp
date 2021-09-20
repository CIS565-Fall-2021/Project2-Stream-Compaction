/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include "testing_helpers.hpp"
#include "recording_helpers.hpp"

#define RECORD_PERFORMANCE_ANALYSIS 0//1
#define RECORD_PERFORMANCE_ANALYSIS_SAMPLE 1000

#if RECORD_PERFORMANCE_ANALYSIS
const int SIZE = 1 << 24;
#else // RECORD_PERFORMANCE_ANALYSIS
const int SIZE = 1 << 20;//1 << 24;//1 << 22;//1 << 20;//1 << 18;//1 << 16;//1 << 12;//1 << 3;//1 << 8; // feel free to change the size of array
#endif // RECORD_PERFORMANCE_ANALYSIS
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
    testMain();
#if RECORD_PERFORMANCE_ANALYSIS
    if (!RecordingHelpers::recordingMain(RECORD_PERFORMANCE_ANALYSIS_SAMPLE)) {
        exit(1);
    }
#endif // RECORD_PERFORMANCE_ANALYSIS
    //system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
