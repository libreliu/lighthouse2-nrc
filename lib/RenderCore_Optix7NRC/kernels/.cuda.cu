/* .cuda.cu - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include ".cuda.h"

namespace lh2core
{

// path tracing buffers and global variables
__constant__ CoreInstanceDesc* instanceDescriptors;
__constant__ CUDAMaterial* materials;
__constant__ CoreLightTri* triLights;
__constant__ CorePointLight* pointLights;
__constant__ CoreSpotLight* spotLights;
__constant__ CoreDirectionalLight* directionalLights;
__constant__ int4 lightCounts;			// area, point, spot, directional
__constant__ uchar4* argb32;
__constant__ float4* argb128;
__constant__ uchar4* nrm32;
__constant__ float4* skyPixels;
__constant__ int skywidth;
__constant__ int skyheight;
__constant__ PathState* pathStates;
__constant__ float4* debugData;
__constant__ LightCluster* lightTree;

__constant__ mat4 worldToSky;

// path tracer settings
__constant__ __device__ float geometryEpsilon;
__constant__ __device__ float clampValue;

// staging: copies will be batched and carried out after rendering completes, 
// to allow the CPU to update the scene concurrently with GPU rendering.

enum { INSTS = 0, MATS, TLGHTS, PLGHTS, SLGHTS, DLGHTS, LCNTS, RGB32, RGBH, NRMLS, SKYPIX, SKYW, SKYH, SMAT, DBGDAT, GEPS, CLMPV, LTREE };

// device pointers are not real pointers for nvcc, so we need a bit of a hack.

struct StagedPtr { void* p; int id; };
struct StagedInt { int v; int id; };
struct StagedInt4 { int4 v; int id; };
struct StagedFloat3 { float3 v; int id; };
struct StagedMat { mat4 v; int id; };
struct StagedF32 { float v; int id; };
struct StagedCpy { void* d; void* s; int n; };
static std::vector<StagedPtr> stagedPtr;
static std::vector<StagedInt> stagedInt;
static std::vector<StagedInt4> stagedInt4;
static std::vector<StagedFloat3> stagedFloat3;
static std::vector<StagedMat> stagedMat;
static std::vector<StagedF32> stagedF32;
static std::vector<StagedCpy> stagedCpy;

__host__ static void pushPtrCpy( int id, void* p )
{
	if (id == INSTS) cudaMemcpyToSymbol( instanceDescriptors, &p, sizeof( void* ) );
	if (id == MATS) cudaMemcpyToSymbol( materials, &p, sizeof( void* ) );
	if (id == TLGHTS) cudaMemcpyToSymbol( triLights, &p, sizeof( void* ) );
	if (id == PLGHTS) cudaMemcpyToSymbol( pointLights, &p, sizeof( void* ) );
	if (id == SLGHTS) cudaMemcpyToSymbol( spotLights, &p, sizeof( void* ) );
	if (id == DLGHTS) cudaMemcpyToSymbol( directionalLights, &p, sizeof( void* ) );
	if (id == RGB32) cudaMemcpyToSymbol( argb32, &p, sizeof( void* ) );
	if (id == RGBH) cudaMemcpyToSymbol( argb128, &p, sizeof( void* ) );
	if (id == NRMLS) cudaMemcpyToSymbol( nrm32, &p, sizeof( void* ) );
	if (id == SKYPIX) cudaMemcpyToSymbol( skyPixels, &p, sizeof( void* ) );
	if (id == DBGDAT) cudaMemcpyToSymbol( debugData, &p, sizeof( void* ) );
	if (id == LTREE) cudaMemcpyToSymbol( lightTree, &p, sizeof( void* ) );
}
__host__ static void pushIntCpy( int id, const int v )
{
	if (id == SKYW) cudaMemcpyToSymbol( skywidth, &v, sizeof( int ) );
	if (id == SKYH) cudaMemcpyToSymbol( skyheight, &v, sizeof( int ) );
}
__host__ static void pushF32Cpy( int id, const float v )
{
	if (id == GEPS) cudaMemcpyToSymbol( geometryEpsilon, &v, sizeof( float ) );
	if (id == CLMPV) cudaMemcpyToSymbol( clampValue, &v, sizeof( int ) );
}
__host__ static void pushMatCpy( int id, const mat4& m )
{
	if (id == SMAT) cudaMemcpyToSymbol( worldToSky, &m, sizeof( mat4 ) );
}
__host__ static void pushInt4Cpy( int id, const int4& v )
{
	if (id == LCNTS) cudaMemcpyToSymbol( lightCounts, &v, sizeof( int4 ) );
}
__host__ static void pushFloat3Cpy( int id, const float3& v )
{
	// nothing here yet
}

#define MAXVARS	32
static void* prevPtr[MAXVARS] = {};
static int prevInt[MAXVARS] = {};
static float prevFloat[MAXVARS] = {};
static int4 prevInt4[MAXVARS] = {};
// static float3 prevFloat3[MAXVARS] = {};
static bool prevValSet[MAXVARS] = {};

__host__ static void stagePtrCpy( int id, void* p )
{
	if (prevPtr[id] == p) return; // not changed
	StagedPtr n = { p, id };
	stagedPtr.push_back( n );
	prevPtr[id] = p;
}
__host__ static void stageIntCpy( int id, const int v )
{
	if (prevValSet[id] == true && prevInt[id] == v) return;
	StagedInt n = { v, id };
	stagedInt.push_back( n );
	prevValSet[id] = true;
	prevInt[id] = v;
}
__host__ static void stageF32Cpy( int id, const float v )
{
	if (prevValSet[id] == true && prevFloat[id] == v) return;
	StagedF32 n = { v, id };
	stagedF32.push_back( n );
	prevValSet[id] = true;
	prevFloat[id] = v;
}
__host__ static void stageMatCpy( int id, const mat4& m ) { StagedMat n = { m, id }; stagedMat.push_back( n ); }
__host__ static void stageInt4Cpy( int id, const int4& v )
{
	if (prevValSet[id] == true && prevInt4[id].x == v.x && prevInt4[id].y == v.y && prevInt4[id].z == v.z && prevInt4[id].w == v.w) return;
	StagedInt4 n = { v, id };
	stagedInt4.push_back( n );
	prevValSet[id] = true;
	prevInt4[id] = v;
}
/* __host__ static void stageFloat3Cpy( int id, const float3& v )
{
	if (prevValSet[id] == true && prevFloat3[id].x == v.x && prevFloat3[id].y == v.y && prevFloat3[id].z == v.z) return;
	StagedFloat3 n = { v, id };
	stagedFloat3.push_back( n );
	prevValSet[id] = true;
	prevFloat3[id] = v;
} */

__host__ void stageMemcpy( void* d, void* s, int n ) { StagedCpy c = { d, s, n }; stagedCpy.push_back( c ); }

__host__ void stageInstanceDescriptors( CoreInstanceDesc* p ) { stagePtrCpy( INSTS /* instanceDescriptors */, p ); }
__host__ void stageMaterialList( CUDAMaterial* p ) { stagePtrCpy( MATS /* materials */, p ); }
__host__ void stageTriLights( CoreLightTri* p ) { stagePtrCpy( TLGHTS /* triLights */, p ); }
__host__ void stagePointLights( CorePointLight* p ) { stagePtrCpy( PLGHTS /* pointLights */, p ); }
__host__ void stageSpotLights( CoreSpotLight* p ) { stagePtrCpy( SLGHTS /* spotLights */, p ); }
__host__ void stageDirectionalLights( CoreDirectionalLight* p ) { stagePtrCpy( DLGHTS /* directionalLights */, p ); }
__host__ void stageARGB32Pixels( uint* p ) { stagePtrCpy( RGB32 /* argb32 */, p ); }
__host__ void stageARGB128Pixels( float4* p ) { stagePtrCpy( RGBH /* argb128 */, p ); }
__host__ void stageNRM32Pixels( uint* p ) { stagePtrCpy( NRMLS /* nrm32 */, p ); }
__host__ void stageSkyPixels( float4* p ) { stagePtrCpy( SKYPIX /* skyPixels */, p ); }
__host__ void stageSkySize( int w, int h ) { stageIntCpy( SKYW /* skywidth */, w ); stageIntCpy( SKYH /* skyheight */, h ); }
__host__ void stageWorldToSky( const mat4& worldToLight ) { stageMatCpy( SMAT /* worldToSky */, worldToLight ); }
__host__ void stageDebugData( float4* p ) { stagePtrCpy( DBGDAT /* debugData */, p ); }
__host__ void stageGeometryEpsilon( float e ) { stageF32Cpy( GEPS /* geometryEpsilon */, e ); }
__host__ void stageClampValue( float c ) { stageF32Cpy( CLMPV /* clampValue */, c ); }
__host__ void stageLightTree( LightCluster* t ) { stagePtrCpy( LTREE /* light tree */, t ); }
__host__ void stageLightCounts( int tri, int point, int spot, int directional )
{
	const int4 counts = make_int4( tri, point, spot, directional );
	stageInt4Cpy( LCNTS /* lightCounts */, counts );
}

__host__ void pushStagedCopies()
{
	for (auto c : stagedCpy) cudaMemcpy( c.d, c.s, c.n, cudaMemcpyHostToDevice ); stagedCpy.clear();
	for (auto n : stagedPtr) pushPtrCpy( n.id, n.p ); stagedPtr.clear();
	for (auto n : stagedInt) pushIntCpy( n.id, n.v ); stagedInt.clear();
	for (auto n : stagedInt4) pushInt4Cpy( n.id, n.v ); stagedInt4.clear();
	for (auto n : stagedFloat3) pushFloat3Cpy( n.id, n.v ); stagedFloat3.clear();
	for (auto n : stagedF32) pushF32Cpy( n.id, n.v ); stagedF32.clear();
	for (auto n : stagedMat) pushMatCpy( n.id, n.v ); stagedMat.clear();
}

// counters for persistent threads
static __device__ Counters* counters;
__global__ void InitCountersForExtend_Kernel( int pathCount )
{
	if (threadIdx.x != 0) return;
	counters->activePaths = pathCount;	// remaining active paths
	counters->shaded = 0;				// persistent thread atomic for shade kernel
	counters->generated = 0;			// persistent thread atomic for generate in .optix.cu
	counters->extensionRays = 0;		// compaction counter for extension rays
	counters->shadowRays = 0;			// compaction counter for connections
	counters->connected = 0;
	counters->totalExtensionRays = pathCount;
	counters->totalShadowRays = 0;
}
__host__ void InitCountersForExtend( int pathCount ) { InitCountersForExtend_Kernel << <1, 32 >> > (pathCount); }
__global__ void InitCountersSubsequent_Kernel()
{
	if (threadIdx.x != 0) return;
	counters->totalExtensionRays += counters->extensionRays;
	counters->activePaths = counters->extensionRays;	// remaining active paths
	counters->extended = 0;				// persistent thread atomic for genSecond in .optix.cu
	counters->shaded = 0;				// persistent thread atomic for shade kernel
	counters->extensionRays = 0;		// compaction counter for extension rays
}
__host__ void InitCountersSubsequent() { InitCountersSubsequent_Kernel << <1, 32 >> > (); }
__host__ void SetCounters( Counters* p ) { cudaMemcpyToSymbol( counters, &p, sizeof( void* ) ); }

// nrc auxiliary counters
static __device__ NRCCounters* nrcCounters;
__host__ void SetNRCCounters( NRCCounters* p ) { cudaMemcpyToSymbol(nrcCounters, &p, sizeof(void*)); }


inline __device__ void EncodeNRCInput(
	float* bufStart,
	const float3 rayIsect,
	const float roughness,
	const float2 rayDir,
	const float2 normalDir,
	const float3 diffuseRefl,
	const float3 specularRefl	
) {
	const size_t position_encoded_offset = 0;
	const size_t direction_encoded_offset = 3 * 12;
	const size_t normal_encoded_offset = direction_encoded_offset + 2 * 4;
	const size_t roughness_encoded_offset = normal_encoded_offset + 2 * 4;
	const size_t diffuse_encoded_offset = roughness_encoded_offset + 4;
	const size_t specular_encoded_offset = diffuse_encoded_offset + 3;

	// Position - Frequency - 3->3x12
	 for (uint i = 0; i < 6; i++) {
	 	bufStart[position_encoded_offset + i] = sinf(powf(2, i) * PI * rayIsect.x);
	 	bufStart[position_encoded_offset + i + 6] = cosf(powf(2, i) * PI * rayIsect.x);
	 }
	 for (uint i = 0; i < 6; i++) {
	 	bufStart[position_encoded_offset + i + 12] = sinf(powf(2, i) * PI * rayIsect.y);
	 	bufStart[position_encoded_offset + i + 12 + 6] = cosf(powf(2, i) * PI * rayIsect.y);
	 }
	 for (uint i = 0; i < 6; i++) {
	 	bufStart[position_encoded_offset + i + 24] = sinf(powf(2, i) * PI * rayIsect.z);
	 	bufStart[position_encoded_offset + i + 24 + 6] = cosf(powf(2, i) * PI * rayIsect.z);
	 }
	//for (uint i = 0; i < 6; i++) {
	//	bufStart[position_encoded_offset + i] = rayIsect.x;
	//	bufStart[position_encoded_offset + i + 6] = rayIsect.x;
	//}
	//for (uint i = 0; i < 6; i++) {
	//	bufStart[position_encoded_offset + i + 12] = rayIsect.y;
	//	bufStart[position_encoded_offset + i + 12 + 6] = rayIsect.y;
	//}
	//for (uint i = 0; i < 6; i++) {
	//	bufStart[position_encoded_offset + i + 24] = rayIsect.z;
	//	bufStart[position_encoded_offset + i + 24 + 6] = rayIsect.z;
	//}

	// Direction - OneBlob - 2->2x4 (k=4, same for below)
	// dir_sph: { 0 to 1, 0 to 1 }
	const float2 dir_sph = make_float2(rayDir.x / (2 * PI) + 0.5f, rayDir.y / PI + 0.5f);
	for (uint i = 0; i < 4; i++) {      // OneBlob size
		for (uint j = 0; j < 2; j++) {  // Input demension
			const float sigma = 1.f / 4.f;
			float x = (i / 4.f) - (j == 0 ? dir_sph.x : dir_sph.y);
			bufStart[direction_encoded_offset + j * 4 + i] =
				1.f / (sqrtf(2.f * PI) * sigma) *
				expf((-x * x / (2.f * sigma * sigma)));
		}
	}

	// Normal - OneBlob
	const float2 normal_sph = make_float2(normalDir.x / (2 * PI) + 0.5f, normalDir.y / PI + 0.5f);
	for (uint i = 0; i < 4; i++) {      // OneBlob size
		for (uint j = 0; j < 2; j++) {  // Input demension
			const float sigma = 1.f / 4.f;
			float x = (i / 4.f) - (j == 0 ? normal_sph.x : normal_sph.y);
			bufStart[normal_encoded_offset + j * 4 + i] =
				1.f / (sqrtf(2.f * PI) * sigma) *
				expf((-x * x / (2.f * sigma * sigma)));
		}
	}

	// Roughness - OneBlob
	for (uint i = 0; i < 4; i++) {  // OneBlob size
		const float sigma = 1.f / 4.f;
		float x = (i / 4.f) - (1 - exp(-roughness));
		bufStart[roughness_encoded_offset + i] =
			1.f / (sqrtf(2.f * PI) * sigma) *
			expf((-x * x / (2.f * sigma * sigma)));
	}
	// Diffuse
	bufStart[diffuse_encoded_offset + 0] = diffuseRefl.x;
	bufStart[diffuse_encoded_offset + 1] = diffuseRefl.y;
	bufStart[diffuse_encoded_offset + 2] = diffuseRefl.z;

	// Specular
	bufStart[specular_encoded_offset + 0] = specularRefl.x;
	bufStart[specular_encoded_offset + 1] = specularRefl.y;
	bufStart[specular_encoded_offset + 2] = specularRefl.z;
}

#define NRC_TRAININPUTBUF(ray_segment_idx, idx) trainInputBuf[(NRC_INPUTDIM * (ray_segment_idx)) + (idx)]

// functional blocks
#include "tools_shared.h"
#include "sampling_shared.h"
#include "material_shared.h"
#include "lights_shared.h"
#include "bsdf.h"
#include "pathtracer.h"
#include "finalize_shared.h"


#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__(128 /* max block size */, 2 /* min blocks per sm TURING */)
#else
__global__  __launch_bounds__(256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */)
#endif
__global__ void PrepareNRCTrainData_Kernel(
	const float4* trainBuf, float* trainInputBuf, float* trainTargetBuf,
	float4* debugView, const uint trainingMode, const uint visualizeMode
) {
	uint jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= NRC_NUMTRAINRAYS) {
		return;
	}

	NRC_DUMP("[trainData] jobIndex=%d", jobIndex);

	float3 luminances[NRC_MAXTRAINPATHLENGTH];
	bool luminanceTrained[NRC_MAXTRAINPATHLENGTH];
	for (uint i = 0; i < NRC_MAXTRAINPATHLENGTH; i++) {
		luminanceTrained[i] = false;
	}

	bool previousDataValid = false;
	uint lastValidPathLength = 0;

	for (uint pathLength = NRC_MAXTRAINPATHLENGTH; pathLength >= 1; pathLength--) {
		const float4 data0 = NRC_TRAINBUF(jobIndex, pathLength, 0);
		const float4 data1 = NRC_TRAINBUF(jobIndex, pathLength, 1);
		const float4 data2 = NRC_TRAINBUF(jobIndex, pathLength, 2);
		const float4 data3 = NRC_TRAINBUF(jobIndex, pathLength, 3);
		const float4 data4 = NRC_TRAINBUF(jobIndex, pathLength, 4);
		const float4 data5 = NRC_TRAINBUF(jobIndex, pathLength, 5);

		const uint flags = __float_as_uint(data4.w);

		// NRC_DUMP("[trainData] jobIndex=%d, pathLen=%d, flags=%x, %d, %d", jobIndex, pathLength, flags, (flags & S_NRC_DATA_VALID), (flags & S_NRC_TRAINING_DISCARD));
		NRC_DUMP(
			"[trainData] trainSlotIdx=%d, pathLen=%d\n"
			"RayIsect=(%f,%f,%f), roughness=%f, Direction=(%f,%f), Normal=(%f,%f)\n"
			"diffuseRefl=(%f,%f,%f), SpecularRefl=(%f,%f,%f), pixelIdx=%d\n"
			"directLo=(%f,%f,%f), flag=%d, throughput=(%f,%f,%f), postponedBsdfPdf=%f\n",
			jobIndex, pathLength,
			data0.x, data0.y, data0.z, data0.w, data1.x, data1.y, data1.z, data1.w,
			data2.x, data2.y, data2.z, data3.x, data3.y, data3.z, __float_as_uint(data2.w),
			data4.x, data4.y, data4.z, __float_as_uint(data4.w), data5.x, data5.y, data5.z, data5.w
			);

		if ((flags & S_NRC_DATA_VALID) > 0 && !previousDataValid) {
			// This is the last bounce, reason:
			// 1. NRC_MAXTRAINPATHLENGTH exceed, or killed by russian roulette
			//   (in this case no luminances other than direct lighting will occur)
			// 2. hit emissive material
			// 3. hit skybox

			// NOTE: possible sources
			// 1. emissive
			// 2. hit skybox
			// 3. direct lighting from one of the lights
			float3 directLuminance = make_float3(data4.x, data4.y, data4.z);
			
			luminances[pathLength - 1] = directLuminance;

			// // short-circuit for convenience
			// if (trainingMode == 1 && pathLength == 1) {
			// 	luminanceTrained[0] = true;
			// } else {
				luminanceTrained[pathLength - 1] = ((flags & S_NRC_TRAINING_DISCARD) == 0);
			// }
			previousDataValid = true;
			lastValidPathLength = pathLength;

			NRC_DUMP("[trainData] jobIndex=%d, pathLen=%d, last bounce", jobIndex, pathLength);

#ifdef NRC_ENABLE_DEBUG_VIEW
			if (visualizeMode == 1 && pathLength == 1) {
				// VISUALIZE_TRAIN_TARGET_FIRSTBOUNCE_DIRECT
				const uint pixelIdx = float_as_uint(data2.w);
				// printf("Visualize %d: (%f,%f,%f)\n", pixelIdx, directLuminance.x, directLuminance.y, directLuminance.z);
				debugView[pixelIdx] += make_float4(directLuminance, 0.0f);
			}
#endif
		}
		else if ((flags & S_NRC_DATA_VALID) > 0 && previousDataValid) {
			// NOTE: only direct lighting from one of the lights are possible here
			float3 directLuminance = make_float3(data4.x, data4.y, data4.z);

			float3 segmentThroughput = make_float3(data5.x, data5.y, data5.z);
			float3 indirectLuminance = segmentThroughput * luminances[pathLength];
			luminances[pathLength - 1] = directLuminance + indirectLuminance;
			luminanceTrained[pathLength - 1] = ((flags & S_NRC_TRAINING_DISCARD) == 0);

			NRC_DUMP("[trainData] jobIndex=%d, pathLen=%d, not last bounce", jobIndex, pathLength);

#ifdef NRC_ENABLE_DEBUG_VIEW
			if (visualizeMode == 1 && pathLength == 1) {
				// VISUALIZE_TRAIN_TARGET_FIRSTBOUNCE_DIRECT
				const uint pixelIdx = float_as_uint(data2.w);
				debugView[pixelIdx] += make_float4(directLuminance, 0.0f);
			} else if (visualizeMode == 2 && pathLength == 1) {
				const uint pixelIdx = float_as_int(data2.w);
				debugView[pixelIdx] += make_float4(indirectLuminance, 0.0f);
			}
#endif
		}
		else if ((flags & S_NRC_DATA_VALID) == 0 && previousDataValid) {
			// illegal data encountered, TODO: error recovery
			NRC_DUMP_WARN("[WARN] illegal data, jobIndex=%d, current pathLength=%d", jobIndex, pathLength);
			return;
		}
	}
	// TODO: debugView

	if (!previousDataValid) {
		// TODO: error recovery
		NRC_DUMP_WARN("[WARN] no valid data, jobIndex=%d", jobIndex);
		return;
	}

	for (uint pathLength = lastValidPathLength; pathLength >= 1; pathLength--) {
		const float4 data0 = NRC_TRAINBUF(jobIndex, pathLength, 0);
		const float4 data1 = NRC_TRAINBUF(jobIndex, pathLength, 1);
		const float4 data2 = NRC_TRAINBUF(jobIndex, pathLength, 2);
		const float4 data3 = NRC_TRAINBUF(jobIndex, pathLength, 3);
		const float4 data4 = NRC_TRAINBUF(jobIndex, pathLength, 4);
		const float4 data5 = NRC_TRAINBUF(jobIndex, pathLength, 5);

		if (!luminanceTrained[pathLength - 1]) {
			NRC_DUMP_WARN("[trainData] jobIndex=%d, pathLen=%d discarded", jobIndex, pathLength);
			// mark as red
			const uint pixelIdx = __float_as_int(data2.w);
			const uint flags = __float_as_uint(data4.w);
			
			// if ((flags & S_NRC_TRAINING_DISCARD) == 0) {
			// 	debugView[pixelIdx] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
			// } else {
			// 	debugView[pixelIdx] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
			// }


			continue;
		}

		const uint raySegmentIdx = atomicAdd( &nrcCounters->nrcActualTrainRays, 1);

		const float3 rayIsect = make_float3(data0.x, data0.y, data0.z);
		const float roughness = data0.w;
		const float2 rayDir = make_float2(data1.x, data1.y);
		const float2 normalDir = make_float2(data1.x, data1.y);
		const float3 diffuseRefl = make_float3(data2.x, data2.y, data2.z);
		const float3 specularRefl = make_float3(data3.x, data3.y, data3.z);
		const float3 luminance = luminances[pathLength - 1];

		EncodeNRCInput(
			&trainInputBuf[raySegmentIdx * NRC_INPUTDIM],
			rayIsect, roughness, rayDir, normalDir, diffuseRefl, specularRefl
		);

		// TODO: add luminance into target buffer
		trainTargetBuf[raySegmentIdx * 3 + 0] = luminance.x;
		trainTargetBuf[raySegmentIdx * 3 + 1] = luminance.y;
		trainTargetBuf[raySegmentIdx * 3 + 2] = luminance.z;
	}
}

__host__ void PrepareNRCTrainData(
	const float4* trainBuf, float* trainInputBuf, float* trainTargetBuf,
	float4* debugView, const uint trainingMode, const uint visualizeMode
) {
	const uint numBlocks = NEXTMULTIPLEOF(NRC_NUMTRAINRAYS, 128) / 128;
	//printf("numBlocks=%d\n", numBlocks);
	PrepareNRCTrainData_Kernel << <numBlocks, 128 >> > (
		trainBuf, trainInputBuf, trainTargetBuf,
		debugView, trainingMode, visualizeMode
	);
}

#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__(128 /* max block size */, 2 /* min blocks per sm TURING */)
#else
__global__  __launch_bounds__(256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */)
#endif
__global__ void NRCNetResultAdd_Kernel(float4* accumulator, const float* inferenceOutputBuffer, const float* inferenceAuxiliaryBuffer, uint numSamples) {
	uint jobIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (jobIndex >= numSamples) {
		return;
	}

	const float4 throughput = make_float4(
		inferenceAuxiliaryBuffer[jobIndex * 4 + 0],
		inferenceAuxiliaryBuffer[jobIndex * 4 + 1],
		inferenceAuxiliaryBuffer[jobIndex * 4 + 2],
		0.0f
	);

	const uint pixelIdx = __float_as_int(inferenceAuxiliaryBuffer[jobIndex * 4 + 3]);
	accumulator[pixelIdx] += throughput * make_float4(
		inferenceOutputBuffer[jobIndex * 3 + 0],
		inferenceOutputBuffer[jobIndex * 3 + 1],
		inferenceOutputBuffer[jobIndex * 3 + 2],
		0.0f
	);
}

__host__ void NRCNetResultAdd(float4* accumulator, const float* inferenceOutputBuffer, const float* inferenceAuxiliaryBuffer, uint numSamples) {
	const uint numBlocks = NEXTMULTIPLEOF(numSamples, 128) / 128;
	NRCNetResultAdd_Kernel << <numBlocks, 128 >> > (accumulator, inferenceOutputBuffer, inferenceAuxiliaryBuffer, numSamples);
}

#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__( 128 /* max block size */, 2 /* min blocks per sm TURING */ )
#else
__global__  __launch_bounds__( 256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */ )
#endif
void shadePrimaryKernel( float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const uint pathCount, float* inferenceInputBuffer, float* inferenceAuxiliaryBuffer )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// gather data by reading sets of four floats for optimal throughput
	const float4 O4 = pathStates[jobIndex];				// ray origin xyz, w can be ignored
	const float4 D4 = pathStates[jobIndex + stride];	// ray direction xyz
	float4 T4 = pathLength == 1 ? make_float4( 1 ) /* faster */ : pathStates[jobIndex + stride * 2]; // path thoughput rgb
	const float4 hitData = hits[jobIndex];
	hits[jobIndex].z = __int_as_float( -1 ); // reset for next query
	const float bsdfPdf = T4.w;

	// derived data
	uint data = __float_as_uint( O4.w ); // prob.density of the last sampled dir, postponed because of MIS
	const float3 D = make_float3( D4 );
	float3 throughput = make_float3( T4 );
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;
	const uint pathIdx = PATHIDX;
	const uint pixelIdx = pathIdx % (w * h);
	const uint sampleIdx = pathIdx / (w * h) + pass;

	// initialize depth in accumulator for DOF shader
	// if (pathLength == 1) accumulator[pixelIdx].w += PRIMIDX == NOHIT ? 10000 : HIT_T;

	// use skydome if we didn't hit any geometry
	if (PRIMIDX == NOHIT)
	{
		float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = FLAGS & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		float3 contribution = throughput * skyPixel * (1.0f / bsdfPdf);
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );
		if (accumulator != nullptr)
			accumulator[pixelIdx] += make_float4( contribution, 0 );
		return;
	}

	// object picking
	if (pixelIdx == probePixelIdx && pathLength == 1)
	{
		counters->probedInstid = INSTANCEIDX;	// record instace id at the selected pixel
		counters->probedTriid = PRIMIDX;		// record primitive id at the selected pixel
		counters->probedDist = HIT_T;			// record primary ray hit distance
	}

	// get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = RAY_O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

	// add to inference buffer
	const uint inferenceRayIdx = atomicAdd(&nrcCounters->nrcNumInferenceRays, 1);
	EncodeNRCInput(
		&inferenceInputBuffer[NRC_INPUTDIM * inferenceRayIdx],
		I, ROUGHNESS, toSphericalCoord(D), toSphericalCoord(N), shadingData.color, shadingData.color
	);

	// Write auxiliary buffer
	// [throughput.x throughput.y throughput.z pixelIdx]
	// the throughput should consider previous pdf (already done in apply postponed bsdf pdf)
	inferenceAuxiliaryBuffer[inferenceRayIdx * 4 + 0] = throughput.x;
	inferenceAuxiliaryBuffer[inferenceRayIdx * 4 + 1] = throughput.y;
	inferenceAuxiliaryBuffer[inferenceRayIdx * 4 + 2] = throughput.z;
	inferenceAuxiliaryBuffer[inferenceRayIdx * 4 + 3] = __uint_as_float( pixelIdx );

}

__host__ void shadePrimary( const int pathCount, float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int scrwidth, const int scrheight,
	const float spreadAngle, float* inferenceInputBuffer, float* inferenceAuxiliaryBuffer )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
	shadePrimaryKernel << <gridDim.x, 128 >> > (accumulator, stride, pathStates, hits, connections, R0, shift, blueNoise,
		pass, probePixelIdx, pathLength, scrwidth, scrheight, spreadAngle, pathCount, inferenceInputBuffer, inferenceAuxiliaryBuffer);
}


} // namespace lh2core

// EOF