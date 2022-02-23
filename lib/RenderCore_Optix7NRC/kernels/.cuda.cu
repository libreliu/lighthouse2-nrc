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
__global__ void PrepareNRCTrainData_Kernel( float4* trainBuf, float4* trainInputBuf, float4* debugView ) {
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= NRC_NUMTRAINRAYS) {
		return;
	}

	float3 luminances[NRC_MAXTRAINPATHLENGTH];
	bool previousDataValid = false;
	uint lastValidPathLength = 0;

	for (uint pathLength = NRC_MAXTRAINPATHLENGTH; pathLength >= 1; pathLength--) {
		const float4 data0 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 0];
		const float4 data1 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 1];
		const float4 data2 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 2];
		const float4 data3 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 3];
		const float4 data4 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 4];
		const float4 data5 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 5];

		const uint flags = __float_as_uint(data4.w);

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
			previousDataValid = true;
			lastValidPathLength = pathLength;
		}
		else if ((flags & S_NRC_DATA_VALID) > 0 && previousDataValid) {
			// NOTE: only direct lighting from one of the lights are possible here
			float3 directLuminance = make_float3(data4.x, data4.y, data4.z);

			float3 segmentThroughput = make_float3(data5.x, data5.y, data5.z);
			float3 indirectLuminance = segmentThroughput * luminances[pathLength];
			luminances[pathLength - 1] = directLuminance + indirectLuminance;
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
		const uint raySegmentIdx = atomicAdd( &nrcCounters->nrcActualTrainRays, 1);

		const float4 data0 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 0];
		const float4 data1 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 1];
		const float4 data2 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 2];
		const float4 data3 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 3];
		const float4 data4 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 4];
		const float4 data5 = trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * jobIndex + (pathLength - 1) * NRC_TRAINCOMPONENTSIZE + 5];

		// TODO: ray origin => ray intersection point
		// float3 intersection
	}
}

__host__ void PrepareNRCTrainData() {}

} // namespace lh2core

// EOF