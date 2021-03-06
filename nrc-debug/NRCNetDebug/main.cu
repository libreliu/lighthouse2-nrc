#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define getcwd _getcwd // stupid MSFT "deprecation" warning
#elif
#include <unistd.h>
#endif

#include "nrc_settings.h"

void NRCNet_Init(cudaStream_t training_stream, cudaStream_t inference_stream);
float NRCNet_TrainCPU(float *trainInputBuffer, float *trainTargetBuffer, size_t numTrainSamples);
float NRCNet_Train(float *trainInputBuffer, float *trainTargetBuffer, size_t numTrainSamples);
void NRCNet_Inference(float *inferenceInputBuffer, float *inferenceOutputBuffer, size_t numInferenceSamples);
void NRCNet_Destroy();

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)

#define ERR_EXIT(msg)         \
    do                        \
    {                         \
        fprintf(stderr, msg "\n"); \
        exit(-1);             \
    } while (0)

#define CHK_CUDA(expr)                                                                  \
    do                                                                                  \
    {                                                                                   \
        cudaError_t cudaStatus = expr;                                                  \
        if (cudaStatus != cudaSuccess)                                                  \
            ERR_EXIT("cuda operation failed at " __FILE__ ":" STRINGIZE(__LINE__) "."); \
    } while (0)

// From CUDATools
int _ConvertSMVer2Cores(int major, int minor)
{
    typedef struct
    {
        int SM, Cores;
    } sSMtoCores;
    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64}, {0x61, 128}, {0x62, 128}, {0x70, 64}, {0x72, 64}, {0x75, 64}, {-1, -1}};
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            return nGpuArchCoresPerSM[index].Cores;
        index++;
    }
    return nGpuArchCoresPerSM[index - 1].Cores;
}

// from CUDATools::FastestDevice
int FastestDevice()
{
    int curdev = 0,
        smperproc = 0, fastest = 0, count = 0, prohibited = 0;
    uint64_t max_perf = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&count);
    if (count == 0)
        ERR_EXIT("No CUDA devices found.\nIf you do have an NVIDIA GPU, consider updating your drivers.");
    while (curdev < count)
    {
        cudaGetDeviceProperties(&deviceProp, curdev);
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                smperproc = 1;
            else
                smperproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            uint64_t compute_perf = (uint64_t)deviceProp.multiProcessorCount * smperproc * deviceProp.clockRate;
            if (compute_perf > max_perf)
            {
                max_perf = compute_perf;
                fastest = curdev;
            }
        }
        else
            prohibited++;
        ++curdev;
    }
    if (prohibited == count)
        ERR_EXIT("All CUDA devices are prohibited from use.");
    return fastest;
}

inline float randUniform(float low, float high) {
    return low + ((float)rand() / RAND_MAX) * high; 
}

void test_learn(void) {
    const int numSamples = NRC_MAXTRAINPATHLENGTH * NRC_NUMTRAINRAYS;
    const int numTrainSteps = 1;

    float trainInput[numSamples][NRC_INPUTDIM];
    float trainTarget[numSamples][3];

    // regression on y=kx+b
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < NRC_INPUTDIM; j++) {
            trainInput[i][j] = randUniform(0, 10);
        }

        for (int j = 0; j < 3; j++) {
            trainTarget[i][j] = randUniform(0, 10);
        }
    }

    for (int i = 0; i < numTrainSteps; i++) {
        float loss = NRCNet_TrainCPU((float *)trainInput, (float *)trainTarget, numSamples);
        printf("Round %d: train loss=%f\n", i, loss);
    }

}

void test_inference(void) {
    
}

void test_nrcNetDumpTrain(const std::string &pathPrefix, const std::string &trainInputFilename, const std::string &trainTargetFilename, int numSamples) {
    std::string inputFilePath = pathPrefix + trainInputFilename;
    std::string targetFilePath = pathPrefix + trainTargetFilename;
    printf("trainInput=%s, trainTarget=%s\n", inputFilePath.c_str(), targetFilePath.c_str());

    FILE* inputfp = fopen(inputFilePath.c_str(), "rb");
    FILE* targetfp = fopen(targetFilePath.c_str(), "rb");
    if (!inputfp || !targetfp) {
        ERR_EXIT("Can't open file provided");
    }

    std::vector<float> inputBuf(numSamples * NRC_INPUTDIM, 0.0f);
    std::vector<float> targetBuf(numSamples * 3, 0.0f);

    int ret = fread(inputBuf.data(), sizeof(float), numSamples * NRC_INPUTDIM, inputfp);
    if (ret != numSamples * NRC_INPUTDIM) {
        printf("Error: trainInputBuf: Expected %d, got %d\n", numSamples * NRC_INPUTDIM, ret);
        ERR_EXIT("Can't read file provided");
    }

    ret = fread(targetBuf.data(), sizeof(float), numSamples * 3, targetfp);
    if (ret != numSamples * 3) {
        printf("Error: trainTargetBuf: Expected %d, got %d\n", numSamples * 3, ret);
        ERR_EXIT("Can't read file provided");
    }

    float loss = NRCNet_TrainCPU(inputBuf.data(), targetBuf.data(), numSamples);
    printf("Loss = %f\n", loss);

    fclose(inputfp);
    fclose(targetfp);
}

void displayCwd() {
    char buffer[4096];
    char* answer = getcwd(buffer, sizeof(buffer));
    std::string s_cwd;
    if (answer)
    {
        s_cwd = answer;
    }

    std::cout << "displayCwd: " << s_cwd << std::endl;
}

int main(int argc, char *argv[])
{
    displayCwd();

    unsigned int device = FastestDevice();
    CHK_CUDA(cudaSetDevice(device));

    cudaStream_t inferenceStream, trainingStream;
    CHK_CUDA(cudaStreamCreate(&trainingStream));
    inferenceStream = trainingStream;

    // NOTE: *can't* use stream 0, since tiny-cuda-nn uses cuda graph underneath
    NRCNet_Init(trainingStream, inferenceStream);

    //test_learn();
    test_nrcNetDumpTrain(NRC_DUMP_PATH, "matrix.1646637806_0.data", "matrix.1646637806_1.data", 9);

    NRCNet_Destroy();

    //CHK_CUDA(cudaDeviceReset());
}