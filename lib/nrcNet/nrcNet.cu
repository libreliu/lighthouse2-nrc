#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/encodings/oneblob.h>

#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/trainer.h>

#include <memory>
#include "nrc_settings.h"

using network_precision_t = float;

struct nrcNetContext {
	std::shared_ptr<tcnn::Loss<network_precision_t>> loss;
	std::shared_ptr<tcnn::Optimizer<network_precision_t>> optimizer;
	std::shared_ptr<tcnn::Network<network_precision_t>> network;
	std::shared_ptr<tcnn::Trainer<float, network_precision_t, network_precision_t>> trainer;

	cudaStream_t inference_stream;
	cudaStream_t training_stream;

#ifdef NRC_ENABLE_DEBUG_DUMP_TO_DISK
	// debug features
	int seed;
	FILE *logfp;

	int lastDumpIdx;
#endif
} nrcNetCtx;

void NRCNet_Init(cudaStream_t training_stream, cudaStream_t inference_stream) {
	// TODO: try exponential moving average

	tcnn::json config = {
		{"loss", {
			{"otype", "RelativeL2"}
		}},
		{"optimizer", {
			{"otype", "Adam"},
			{"learning_rate", 1e-2},
			{"beta1", 0.9f},
			{"beta2", 0.99f},
			{"l2_reg", 0.0f},
		}},
		{"network", {
			{"otype", "CutlassMLP"},
			{"n_input_dims", 64},
			{"n_output_dims", 3},
			{"n_neurons", 64},
			{"n_hidden_layers", 4},
			{"activation", "ReLU"},
			{"output_activation", "None"},
		}},
	};

	tcnn::json loss_opts = config.value("loss", tcnn::json::object());
	tcnn::json optimizer_opts = config.value("optimizer", tcnn::json::object());
	tcnn::json network_opts = config.value("network", tcnn::json::object());

	nrcNetCtx.loss.reset(
		tcnn::create_loss<network_precision_t>(loss_opts)
	);
	nrcNetCtx.optimizer.reset(
		tcnn::create_optimizer<network_precision_t>(optimizer_opts)
	);
	nrcNetCtx.network.reset(
		tcnn::create_network<network_precision_t>(network_opts)
	);
	nrcNetCtx.trainer = std::make_shared<
		tcnn::Trainer<float, network_precision_t, network_precision_t>
	>(
		nrcNetCtx.network, nrcNetCtx.optimizer, nrcNetCtx.loss
	);

	nrcNetCtx.inference_stream = inference_stream;
	nrcNetCtx.training_stream = training_stream;

#ifdef NRC_ENABLE_DEBUG_DUMP_TO_DISK
	nrcNetCtx.seed = time(NULL);
	std::string logFilePath = std::string(NRC_DUMP_PATH "/dump.log.") + std::to_string(nrcNetCtx.seed) + ".py";

	nrcNetCtx.logfp = fopen(logFilePath.c_str(), "w");
	if (!nrcNetCtx.logfp) {
		NRC_DUMP_WARN("Can't open dump log in disk, exit");
		exit(1);
	}

	nrcNetCtx.lastDumpIdx = 0;
#endif
}

#ifdef NRC_ENABLE_DEBUG_DUMP_TO_DISK
void dumpBufferToDisk(void *buffer, size_t sizeInBytes, const char *prefix, std::string &fileName) {
	std::string filePath = std::string(prefix) + fileName;
	FILE *fp = fopen(filePath.c_str(), "wb");
	if (!fp) {
		NRC_DUMP_WARN("Can't open file %s for write, exit", filePath.c_str());
		exit(1);
	}

	size_t ret = fwrite(buffer, 1, sizeInBytes, fp);
	if (ret != sizeInBytes) {
		NRC_DUMP_WARN("Can't write to file %s, exit", filePath.c_str());
		exit(1);
	}

	fclose(fp);
}
#endif

void NRCNet_Destroy() {
	nrcNetCtx.loss.reset();
	nrcNetCtx.optimizer.reset();
	nrcNetCtx.network.reset();
	nrcNetCtx.trainer.reset();

#ifdef NRC_ENABLE_DEBUG_DUMP_TO_DISK
	fclose(nrcNetCtx.logfp);
#endif
}

__global__ void gpu_copy_float(uint32_t n_elements, float* __restrict__ src, float* __restrict__ dst) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	dst[i] = src[i];
}

// Retrieve from cpu memory buffer
float NRCNet_TrainCPU(
	float* trainInputBuffer,
	float* trainTargetBuffer,
	size_t numTrainSamples
) {
	tcnn::GPUMemory<float> trainInputAux(numTrainSamples * NRC_INPUTDIM);
	tcnn::GPUMemory<float> trainTargetAux(numTrainSamples * 3);

#ifdef NRC_ENABLE_DEBUG_DUMP_TO_DISK
	std::string trainInputFilename = "matrix." + std::to_string(nrcNetCtx.seed) + "_" + std::to_string(nrcNetCtx.lastDumpIdx++) + ".data";
	fprintf(nrcNetCtx.logfp, "log.append({'type': 'trainInput', 'numSamples': %zd, 'fileName': '%s'})\n",
		numTrainSamples,
		trainInputFilename.c_str()
	);
	fflush(nrcNetCtx.logfp);
	dumpBufferToDisk(trainInputBuffer, numTrainSamples * NRC_INPUTDIM * sizeof(float), NRC_DUMP_PATH, trainInputFilename);

	std::string trainTargetFilename = "matrix." + std::to_string(nrcNetCtx.seed) + "_" + std::to_string(nrcNetCtx.lastDumpIdx++) + ".data";
	fprintf(nrcNetCtx.logfp, "log.append({'type': 'trainTarget', 'numSamples': %zd, 'fileName': '%s'})\n",
		numTrainSamples,
		trainTargetFilename.c_str()
	);
	fflush(nrcNetCtx.logfp);
	dumpBufferToDisk(trainTargetBuffer, numTrainSamples * 3 * sizeof(float), NRC_DUMP_PATH, trainTargetFilename);
#endif

	trainInputAux.copy_from_host(trainInputBuffer);
	trainTargetAux.copy_from_host(trainTargetBuffer);

	tcnn::GPUMatrix<float, tcnn::CM> trainInput(NRC_INPUTDIM, numTrainSamples);
	tcnn::GPUMatrix<float, tcnn::CM> trainTarget(3, numTrainSamples);

	// TODO: check stream it resides in
	tcnn::linear_kernel(gpu_copy_float, 0, nrcNetCtx.training_stream, numTrainSamples * NRC_INPUTDIM, trainInputAux.data(), trainInput.data());
	tcnn::linear_kernel(gpu_copy_float, 0, nrcNetCtx.training_stream, numTrainSamples * 3, trainTargetAux.data(), trainTarget.data());

	cudaStreamSynchronize(nrcNetCtx.training_stream);

	float loss_value;
	nrcNetCtx.trainer->training_step(nrcNetCtx.training_stream, trainInput, trainTarget, &loss_value, nullptr);

	return loss_value;
}

float NRCNet_Train(
	float* trainInputBuffer,
	float* trainTargetBuffer,
	size_t numTrainSamples
) {
	tcnn::GPUMatrix<float, tcnn::CM> trainInput(trainInputBuffer, NRC_INPUTDIM, numTrainSamples);
	tcnn::GPUMatrix<float, tcnn::CM> trainTarget(trainTargetBuffer, 3, numTrainSamples);

	float loss_value;
	nrcNetCtx.trainer->training_step(nrcNetCtx.training_stream, trainInput, trainTarget, &loss_value, nullptr);

	return loss_value;
}

void NRCNet_Inference(
	float* inferenceInputBuffer,
	float* inferenceOutputBuffer,
	size_t numInferenceSamples
) {
	// That's strange, but only CM works
	tcnn::GPUMatrix<float, tcnn::CM> inferenceInput(inferenceInputBuffer, NRC_INPUTDIM, numInferenceSamples);
	tcnn::GPUMatrix<float, tcnn::CM> inferenceTarget(inferenceOutputBuffer, 3, numInferenceSamples);

	nrcNetCtx.network->inference(
		nrcNetCtx.inference_stream,
		inferenceInput,
		inferenceTarget
	);
}