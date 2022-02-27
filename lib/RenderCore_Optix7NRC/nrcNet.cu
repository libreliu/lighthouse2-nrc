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

using network_precision_t = float;

static struct nrcNetContext {
	std::unique_ptr<tcnn::Loss<network_precision_t>> loss;
	std::unique_ptr<tcnn::Optimizer<network_precision_t>> optimizer;
	std::unique_ptr<tcnn::NetworkWithInputEncoding<network_precision_t>> network;

	std::unique_ptr<tcnn::Trainer<float, network_precision_t, network_precision_t>> trainer;
} nrcNetCtx;

__host__ void NRCNet_Init() {

}