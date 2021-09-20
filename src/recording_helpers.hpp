#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace RecordingHelpers {
    template<typename T>
    int cmpArrays(int n, T *a, T *b) {
        for (int i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
                return 1;
            }
        }
        return 0;
    }

    template<typename T>
    bool checkCmpResult(int n, T *a, T *b) {
        if (cmpArrays(n, a, b)) {
            printf("    FAIL VALUE \n");
            return false;
        }
        return true;
    }

    template<typename T>
    bool checkCmpLenResult(int n, int expN, T *a, T *b) {
        if (n != expN) {
            printf("    expected %d elements, got %d\n", expN, n);
            return false;
        }
        if (cmpArrays(n, a, b)) {
            printf("    FAIL VALUE \n");
            return false;
        }
        return true;
    }

    class Recorder {
    public:
        static Recorder& get();

        void reset();

        void reserveSampleCount(size_t count);

        void increaseSampleCount();

        size_t recordLogSize(int size);

        void recordFileEventSizeDuration(const std::string& fileName, const std::string& eventName, int logSize, float duration);

        const std::vector<int>& getLogSizes();

        void writeToFiles(const std::string& baseDir = "./") const;

    protected:
        std::unordered_map<std::string, float>& registerEventToTimeInternal(const std::string& fileName, std::unordered_map<std::string, std::unordered_map<std::string, float>>& fileToEventToTime);

    private:
        std::vector<std::vector<std::unordered_map<std::string, std::unordered_map<std::string, float>>>> samplesSizesFileToEventToTime;

        std::unordered_map<int, size_t> logSizeToIndex;
        std::vector<int> logSizes;
        std::unordered_set<std::string> fileNames;
    };

    bool recordingMain(int maxCount = 1000);
}
