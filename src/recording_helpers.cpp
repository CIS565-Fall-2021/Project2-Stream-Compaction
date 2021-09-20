#include <fstream>
#include "recording_helpers.hpp"
#include "testing_helpers.hpp"
#include "main.hpp"

#define CHECK_BODY(precondition, body, file, line) (precondition && body && (printf("    At line %d in '%s'\n", line, file) || 1))

RecordingHelpers::Recorder& RecordingHelpers::Recorder::get() {
    static Recorder recorder;
    return recorder;
}

void RecordingHelpers::Recorder::reset() {
    samplesSizesFileToEventToTime.empty();
    logSizes.empty();
    fileNames.empty();
}

void RecordingHelpers::Recorder::reserveSampleCount(size_t count) {
    samplesSizesFileToEventToTime.reserve(count);
}

void RecordingHelpers::Recorder::increaseSampleCount() {
    samplesSizesFileToEventToTime.emplace_back();
    samplesSizesFileToEventToTime.back().resize(logSizes.size());
}

size_t RecordingHelpers::Recorder::recordLogSize(int logSize) {
    auto it = logSizeToIndex.find(logSize);
    if (it != logSizeToIndex.end()) {
        return it->second;
    }
    size_t result = logSizes.size();
    logSizes.push_back(logSize);
    logSizeToIndex[logSize] = result;
    return result;
}

void RecordingHelpers::Recorder::recordFileEventSizeDuration(const std::string& fileName, const std::string& eventName, int logSize, float duration) {
    std::vector<std::unordered_map<std::string, std::unordered_map<std::string, float>>>& SizesFileToEventToTime = samplesSizesFileToEventToTime.back();
    std::unordered_map<std::string, std::unordered_map<std::string, float>>& fileToEventToTime = SizesFileToEventToTime[recordLogSize(logSize)];
    std::unordered_map<std::string, float>& eventToTime = registerEventToTimeInternal(fileName, fileToEventToTime);
    eventToTime[eventName] = duration;
}

const std::vector<int>& RecordingHelpers::Recorder::getLogSizes() {
    return logSizes;
}

void RecordingHelpers::Recorder::writeToFiles(const std::string& baseDir) const {
    if (samplesSizesFileToEventToTime.size() == 0 || samplesSizesFileToEventToTime[0].size() == 0) {
        return;
    }
    for (const std::string& fileName : fileNames) {
        std::vector<std::string> eventNames;
        auto& firstEventToTime = samplesSizesFileToEventToTime[0][0].at(fileName);
        eventNames.reserve(firstEventToTime.size());
        for (auto& eventTimePair : firstEventToTime) {
            eventNames.push_back(eventTimePair.first);
        }

        std::vector<std::vector<std::string>> samplesHeaders(samplesSizesFileToEventToTime.size() + 3);

        for (size_t sample = 0; sample < samplesSizesFileToEventToTime.size(); ++sample) {
            samplesHeaders[sample].push_back("sample-" + std::to_string(sample));
        }

        samplesHeaders[samplesSizesFileToEventToTime.size()].push_back("average");
        samplesHeaders[samplesSizesFileToEventToTime.size() + 1].push_back("max");
        samplesHeaders[samplesSizesFileToEventToTime.size() + 2].push_back("min");

        for (std::string& eventName : eventNames) {
            for (auto& headers : samplesHeaders) {
                headers.push_back(eventName);
            }
        }

        std::vector<std::vector<std::vector<double>>> samplesSizesTimes(samplesSizesFileToEventToTime.size() + 3, std::vector<std::vector<double>>(logSizes.size(), std::vector<double>(eventNames.size())));
        double invSampleCount = 1. / samplesSizesFileToEventToTime.size();

        for (size_t sample = 0; sample < samplesSizesFileToEventToTime.size(); ++sample) {
            for (size_t sizeIdx = 0; sizeIdx < logSizes.size(); ++sizeIdx) {
                int logSize = logSizes[sizeIdx];
                for (size_t eventIdx = 0; eventIdx < eventNames.size(); ++eventIdx) {
                    const std::string& eventName = eventNames[eventIdx];
                    double value = samplesSizesFileToEventToTime[sample][sizeIdx].at(fileName).at(eventName);
                    samplesSizesTimes[sample][sizeIdx][eventIdx] = value;
                    samplesSizesTimes[samplesSizesFileToEventToTime.size()][sizeIdx][eventIdx] += value * invSampleCount;
                    if (value > samplesSizesTimes[samplesSizesFileToEventToTime.size() + 1][sizeIdx][eventIdx]) {
                        samplesSizesTimes[samplesSizesFileToEventToTime.size() + 1][sizeIdx][eventIdx] = value;
                    }
                    if (value < samplesSizesTimes[samplesSizesFileToEventToTime.size() + 2][sizeIdx][eventIdx] || samplesSizesTimes[samplesSizesFileToEventToTime.size() + 2][sizeIdx][eventIdx] == 0.) {
                        samplesSizesTimes[samplesSizesFileToEventToTime.size() + 2][sizeIdx][eventIdx] = value;
                    }
                }
            }
        }

        std::ofstream fout;
        fout.open(baseDir + fileName);
        
        for (size_t i = 0; i < samplesSizesTimes.size(); ++i) {
            size_t block = (i < 3) ? samplesSizesTimes.size() - 3 + i : i - 3;
            auto& headers = samplesHeaders[block];
            for (size_t j = 0; j + 1 < headers.size(); ++j) {
                fout << headers[j] << ',';
            }
            fout << headers.back() << std::endl;

            auto& sizesTimes = samplesSizesTimes[block];
            for (size_t sizeIdx = 0; sizeIdx < logSizes.size(); ++sizeIdx) {
                fout << (1 << logSizes[sizeIdx]);
                for (size_t j = 0; j < sizesTimes[sizeIdx].size(); ++j) {
                    fout << ',' << sizesTimes[sizeIdx][j];
                }
                fout << std::endl;
            }

            fout << std::endl;
        }

        fout.close();
    }
}

std::unordered_map<std::string, float>& RecordingHelpers::Recorder::registerEventToTimeInternal(const std::string& fileName, std::unordered_map<std::string, std::unordered_map<std::string, float>>& fileToEventToTime)
{
    auto it = fileToEventToTime.find(fileName);
    if (it != fileToEventToTime.end()) {
        return it->second;
    }
    fileToEventToTime[fileName] = {};
    fileNames.insert(fileName);
    return fileToEventToTime[fileName];
}

inline bool needToCheckResult(int sampleCount) {
    return true;
    //return false;
}

bool recordingAux(int logSize, const std::string& scanDir, const std::string& compactDir, const std::string& sortDir, int sampleCount) {
    RecordingHelpers::Recorder& recorder = RecordingHelpers::Recorder::get();
    int SIZE = 1 << logSize;
    int NPOT = SIZE - 3;

    genArray(SIZE, a, 50);
    //a[SIZE - 1] = 0;
    //printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    //printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + scanDir, "cpu", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    //printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + scanDir, "cpu", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(NPOT, b, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + scanDir, "naive", logSize, StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(SIZE, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    //printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + scanDir, "naive", logSize, StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + scanDir, "work-efficient", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(SIZE, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + scanDir, "work-efficient", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + scanDir, "thrust", logSize, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(SIZE, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    //printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + scanDir, "thrust", logSize, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(NPOT, b, c);

    /////////////////////////////////////////
    //if (SIZE <= (1 << 7)) {
    //    zeroArray(SIZE, c);
    //    printDesc("work-efficient-test scan, power-of-two");
    //    StreamCompaction::EfficientTest::scan(SIZE, c, a);
    //    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //    //printArray(SIZE, c, true);//
    //    printCmpResult(SIZE, b, c);
    //}

    //zeroArray(SIZE, c);
    //printDesc("work-efficient-sh-mem-opt-onesplit scan, power-of-two");
    //StreamCompaction::Efficient::scanBatch(SIZE, c, a, true);
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    ////printArray(SIZE, c, true);//
    //printCmpResult(SIZE, b, c);

    //zeroArray(SIZE, c);
    //printDesc("work-efficient-sh-mem-opt-onesplit scan, non-power-of-two");
    //StreamCompaction::Efficient::scanBatch(NPOT, c, a, true);
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    ////printArray(SIZE, c, true);//
    //printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient-hierarchical scan, power-of-two");
    StreamCompaction::Efficient::scanBatch(SIZE, c, a, false);
    recorder.recordFileEventSizeDuration("power_of_two_" + scanDir, "work-efficient-hierarchical", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(SIZE, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient-hierarchical scan, non-power-of-two");
    StreamCompaction::Efficient::scanBatch(NPOT, c, a, false);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + scanDir, "work-efficient-hierarchical", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(NPOT, b, c);
    /////////////////////////////////////////

    //printf("\n");
    //printf("****************************************\n");
    //printf("** STREAM COMPACTION TESTS %010d **\n", SIZE);
    //printf("****************************************\n");

    // Compaction tests

    genArray(SIZE, a, 4);  // Leave a 0 at the end to test that edge case
    //a[SIZE - 1] = 3;//3;//0;
    //printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    //printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + compactDir, "cpu-without-scan", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    //printArray(count, b, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedCount, b, b), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    //printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + compactDir, "cpu-without-scan", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    //printArray(count, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedNPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("cpu compact with scan, power-of-two");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + compactDir, "cpu-with-scan", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(count, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedCount, b, b), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    //printDesc("cpu compact with scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + compactDir, "cpu-with-scan", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(count, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedNPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + compactDir, "work-efficient", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedCount, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + compactDir, "work-efficient", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedNPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("thrust compact, power-of-two");
    count = StreamCompaction::Thrust::compact(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + compactDir, "thrust", logSize, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedCount, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    //printDesc("thrust compact, non-power-of-two");
    count = StreamCompaction::Thrust::compact(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + compactDir, "thrust", logSize, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedNPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedNPOT, b, c);

    /////////////////////////////////////////

    //zeroArray(SIZE, c);
    //printDesc("work-efficient-sh-mem-opt-onesplit compact, power-of-two");
    //count = StreamCompaction::Efficient::compactBatch(SIZE, c, a, true);
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    ////printArray(SIZE, c, true);//
    //printCmpLenResult(count, expectedNPOT, b, c);

    //zeroArray(SIZE, c);
    //printDesc("work-efficient-sh-mem-opt-onesplit compact, non-power-of-two");
    //count = StreamCompaction::Efficient::compactBatch(NPOT, c, a, true);
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    ////printArray(SIZE, c, true);//
    //printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient-hierarchical compact, power-of-two");
    count = StreamCompaction::Efficient::compactBatch(SIZE, c, a, false);
    recorder.recordFileEventSizeDuration("power_of_two_" + compactDir, "work-efficient-hierarchical", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedCount, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient-hierarchical compact, non-power-of-two");
    count = StreamCompaction::Efficient::compactBatch(NPOT, c, a, false);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + compactDir, "work-efficient-hierarchical", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpLenResult(count, expectedNPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpLenResult(count, expectedNPOT, b, c);
    /////////////////////////////////////////

    //printf("\n");
    //printf("***************************\n");
    //printf("** SORT TESTS %010d **\n", SIZE);
    //printf("***************************\n");

    genArray(SIZE, a, SIZE >> 2);  // Leave a 0 at the end to test that edge case
    //a[SIZE - 1] = 0;//9999;//0;
    //printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    //printDesc("cpu sort, power-of-two");
    StreamCompaction::CPU::sort(SIZE, b, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + sortDir, "cpu", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    //printDesc("thrust sort, power-of-two");
    StreamCompaction::Thrust::sort(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + sortDir, "thrust", logSize, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(SIZE, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient sort, power-of-two");
    StreamCompaction::Efficient::sort(SIZE, c, a);
    recorder.recordFileEventSizeDuration("power_of_two_" + sortDir, "work-efficient-hierarchical", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(SIZE, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    zeroArray(SIZE, b);
    //printDesc("cpu sort, non-power-of-two");
    StreamCompaction::CPU::sort(NPOT, b, a);
    //printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    recorder.recordFileEventSizeDuration("non_power_of_two_" + sortDir, "cpu", logSize, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    //printArray(NPOT, b, true);

    zeroArray(SIZE, c);
    //printDesc("thrust sort, non-power-of-two");
    StreamCompaction::Thrust::sort(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + sortDir, "thrust", logSize, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    //printDesc("work-efficient sort, non-power-of-two");
    StreamCompaction::Efficient::sort(NPOT, c, a);
    recorder.recordFileEventSizeDuration("non_power_of_two_" + sortDir, "work-efficient-hierarchical", logSize, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    //printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);//
    if (CHECK_BODY(needToCheckResult(sampleCount), !RecordingHelpers::checkCmpResult(NPOT, b, c), __FILE__, __LINE__)) {
        return false;
    }
    //printCmpResult(NPOT, b, c);

    return true;
}

bool RecordingHelpers::recordingMain(int maxCount) {
    printf("\n");
    printf("******************************************\n");
    printf("** RECORD PERFORMANCE FOR %6d TIMES **\n", maxCount);
    printf("******************************************\n");

    auto startTime = std::chrono::high_resolution_clock::now();

    std::string baseDir = "../profile/";
    std::string scanFile = "scan.csv";
    std::string compactFile = "compact.csv";
    std::string sortFile = "sort.csv";

    Recorder& recorder = Recorder::get();

    recorder.reset();
    recorder.reserveSampleCount(maxCount);

    for (int logSize = 12; (1 << logSize) <= SIZE; logSize += 2) {
        recorder.recordLogSize(logSize);
    }

    for (int count = 0; count < maxCount; ++count) {
        recorder.increaseSampleCount();
        for (int logSize : recorder.getLogSizes()) {
            if (!recordingAux(logSize, scanFile, compactFile, sortFile, count)) {
                return false;
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duroSecond = endTime - startTime;
        printf("Complete count %d, %lf second elpased.\n", count, duroSecond.count());
    }
    recorder.writeToFiles(baseDir);

    printf("\n");
    printf("*****************\n");
    printf("** FILES SAVED **\n");
    printf("*****************\n");
    return true;
}

