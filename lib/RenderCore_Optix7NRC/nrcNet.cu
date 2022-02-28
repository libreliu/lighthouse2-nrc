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
}

float NRCNet_Train(
	float* trainInputBuffer,
	float* trainTargetBuffer,
	size_t numTrainSamples
) {
	tcnn::GPUMatrix<float, tcnn::RM> trainInput(trainInputBuffer, NRC_INPUTDIM, numTrainSamples);
	tcnn::GPUMatrix<float, tcnn::RM> trainTarget(trainTargetBuffer, 3, numTrainSamples);

	float loss_value;
	nrcNetCtx.trainer->training_step(nrcNetCtx.training_stream, trainInput, trainTarget, &loss_value, nullptr);

	return loss_value;
}

void NRCNet_Inference(
	float* inferenceInputBuffer,
	float* inferenceOutputBuffer,
	size_t numInferenceSamples
) {
	// Do we need to pad?
	tcnn::GPUMatrix<float, tcnn::RM> inferenceInput(inferenceInputBuffer, NRC_INPUTDIM, numInferenceSamples);
	tcnn::GPUMatrix<float, tcnn::RM> inferenceTarget(inferenceOutputBuffer, 3, numInferenceSamples);

	nrcNetCtx.network->inference(
		nrcNetCtx.inference_stream,
		inferenceInput,
		inferenceTarget
	);
}