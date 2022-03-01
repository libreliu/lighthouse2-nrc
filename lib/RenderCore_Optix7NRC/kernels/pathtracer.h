/* pathtracer.cu - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file implements the shading stage of the wavefront algorithm.
   It takes a buffer of hit results and populates a new buffer with
   extension rays. Shadow rays are added with 'potential contributions'
   as fire-and-forget rays, to be traced later. Streams are compacted
   using simple atomics. The kernel is a 'persistent kernel': a fixed
   number of threads fights for food by atomically decreasing a counter.

   The implemented path tracer is deliberately simple.
   This file is as similar as possible to the one in OptixPrime_B.
*/

#include "noerrors.h"

// path state flags
#define S_SPECULAR		1	// previous path vertex was specular
#define S_BOUNCED		2	// path encountered a diffuse vertex
#define S_VIASPECULAR	4	// path has seen at least one specular vertex
#define S_BOUNCEDTWICE	8	// this core will stop after two diffuse bounces
#define S_NRC_TRAINING_PATH_ENDED 16  // (NRC) this training path has ended here
#define S_NRC_DATA_VALID 32           // (NRC) the data presented is valid
#define S_NRC_TRAINING_DISCARD 64     // (NRC) this training sample should be discarded
#define ENOUGH_BOUNCES	S_BOUNCED // or S_BOUNCEDTWICE

// readability defines; data layout is optimized for 128-bit accesses
#define PRIMIDX __float_as_int( hitData.z )
#define INSTANCEIDX __float_as_int( hitData.y )
#define HIT_U ((__float_as_uint( hitData.x ) & 65535) * (1.0f / 65535.0f))
#define HIT_V ((__float_as_uint( hitData.x ) >> 16) * (1.0f / 65535.0f))
#define HIT_T hitData.w
#define RAY_O make_float3( O4 )
#define FLAGS data
#define PATHIDX (data >> 6)

// nrc readability defines
#define NRC_TRAINBUF(train_slot_idx, path_length, index) \
	trainBuf[(NRC_MAXTRAINPATHLENGTH * NRC_TRAINCOMPONENTSIZE) * (train_slot_idx) + ((path_length) - 1) * NRC_TRAINCOMPONENTSIZE + (index)]

// Full sphere instead of half, different from Spherical* utility functions
LH2_DEVFUNC float2 toSphericalCoord(const float3& v)
{
  /* -PI ~ PI */
  const float theta = std::atan2(v.y, v.x);

  /* -PI/2 ~ PI/2 */
  const float phi = std::asin(clamp(v.z, -1.f, 1.f));
  return make_float2(theta, phi);
}

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Implements the shade phase of the wavefront path tracer.             LH2'19|
//  +-----------------------------------------------------------------------------+
#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__( 128 /* max block size */, 2 /* min blocks per sm TURING */ )
#else
__global__  __launch_bounds__( 256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */ )
#endif
void shadeKernel( float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const uint pathCount )
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

	// we need to detect alpha in the shading code.
	if (shadingData.flags & 1)
	{
		if (pathLength < MAXPATHLENGTH)
		{
			const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
			pathStates[extensionRayIdx] = make_float4( I + D * geometryEpsilon, O4.w );
			pathStates[extensionRayIdx + stride] = D4;
			if (!(isfinite( T4.x + T4.y + T4.z ))) T4 = make_float4( 0, 0, 0, T4.w );
			pathStates[extensionRayIdx + stride * 2] = T4;
		}
		return;
	}

	// stop on light
	if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
	{
		const float DdotNL = -dot( D, N );
		float3 contribution = make_float3( 0 ); // initialization required.
		if (DdotNL > 0 /* lights are not double sided */)
		{
			if (pathLength == 1 || (FLAGS & S_SPECULAR) > 0 || connections == 0)
			{
				// accept light contribution if previous vertex was specular
				contribution = shadingData.color;
			}
			else
			{
				// last vertex was not specular: apply MIS
				const float3 lastN = UnpackNormal( __float_as_uint( D4.w ) );
				const CoreTri& tri = (const CoreTri&)instanceTriangles[PRIMIDX];
				const float lightPdf = CalculateLightPDF( D, HIT_T, tri.area, N );
				const float pickProb = LightPickProb( tri.ltriIdx, RAY_O, lastN, I /* the N at the previous vertex */ );
				// const float pickProb = LightPickProbLTree( tri.ltriIdx, RAY_O, lastN, I /* the N at the previous vertex */, seed );
				if ((bsdfPdf + lightPdf * pickProb) > 0) contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
			}
			CLAMPINTENSITY;
			FIXNAN_FLOAT3( contribution );
			accumulator[pixelIdx] += make_float4( contribution, 0 );
		}
		return;
	}

	// path regularization
	if (FLAGS & S_BOUNCED) shadingData.parameters.x |= 255u << 24; // set roughness to 1 after a bounce

	// detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.5f) FLAGS |= S_SPECULAR; /* detect pure speculars; skip NEE for these */ else FLAGS &= ~S_SPECULAR;

	// normal alignment for backfacing polygons
	const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

	// apply postponed bsdf pdf
	throughput *= 1.0f / bsdfPdf;

	// prepare random numbers
	float4 r4;
	if (sampleIdx < 64)
	{
		const uint x = ((pathIdx % w) + (shift & 127)) & 127;
		const uint y = ((pathIdx / w) + (shift >> 24)) & 127;
		r4 = blueNoiseSampler4( blueNoise, x, y, sampleIdx, 4 * pathLength - 4 );
	}
	else
	{
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}

	// next event estimation: connect eye path to light
	if ((FLAGS & S_SPECULAR) == 0 && connections != 0) // skip for specular vertices
	{
		float pickProb, lightPdf = 0;
		float3 lightColor, L = RandomPointOnLight( r4.x, r4.y, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		// float3 lightColor, L = RandomPointOnLightLTree( r4.x, r4.y, seed, I, fN * faceDir, pickProb, lightPdf, lightColor, false ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN * faceDir );
		if (NdotL > 0 && lightPdf > 0)
		{
			float bsdfPdf;
		#ifdef BSDF_HAS_PURE_SPECULARS // see note in lambert.h
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf ) * ROUGHNESS;
		#else
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf );
		#endif
			if (bsdfPdf > 0)
			{
				// calculate potential contribution
				float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf + bsdfPdf));
				FIXNAN_FLOAT3( contribution );
				CLAMPINTENSITY;
				// add fire-and-forget shadow ray to the connections buffer
				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
				connections[shadowRayIdx] = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), 0 ); // O4
				connections[shadowRayIdx + stride * 2] = make_float4( L, dist - 2 * geometryEpsilon ); // D4
				connections[shadowRayIdx + stride * 2 * 2] = make_float4( contribution, __int_as_float( pixelIdx ) ); // E4
			}
		}
	}

	// cap at two diffuse bounces, or a maxium path length
	if (FLAGS & ENOUGH_BOUNCES || pathLength == MAXPATHLENGTH) return;

	// evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r4.z, r4.w, RandomFloat( seed ), R, newBsdfPdf, specular );
	if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) return;
	if (specular) FLAGS |= S_SPECULAR;

	// russian roulette
	const float p = ((FLAGS & S_SPECULAR) || ((FLAGS & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );
	if (p < RandomFloat( seed )) return; else throughput *= 1 / p;

	// write extension ray, with compaction. Note: nvcc will aggregate automatically, 
	// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics 
	const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
	const uint packedNormal = PackNormal( fN * faceDir );
	if (!(FLAGS & S_SPECULAR)) FLAGS |= FLAGS & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED; else FLAGS |= S_VIASPECULAR;
	pathStates[extensionRayIdx] = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), __uint_as_float( FLAGS ) );
	pathStates[extensionRayIdx + stride] = make_float4( R, __uint_as_float( packedNormal ) );
	FIXNAN_FLOAT3( throughput );
	pathStates[extensionRayIdx + stride * 2] = make_float4( throughput * bsdf * abs( dot( fN, R ) ), newBsdfPdf );
}

//  +-----------------------------------------------------------------------------+
//  |  shadeTrainKernel                                                           |
//  |  Implements the shade phase of the wavefront path tracer.             LH2'19|
//  +-----------------------------------------------------------------------------+
#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__( 128 /* max block size */, 2 /* min blocks per sm TURING */ )
#else
__global__  __launch_bounds__( 256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */ )
#endif
void shadeTrainKernel( float4* trainBuf, const uint stride,
	float4* trainPathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int pathLength, const int w, const int h, const float spreadAngle,
	const uint pathCount, float4* debugView )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// gather data by reading sets of four floats for optimal throughput
	const float4 O4 = trainPathStates[jobIndex];				// ray origin xyz, w can be ignored
	const float4 D4 = trainPathStates[jobIndex + stride];	// ray direction xyz
	float4 T4 = pathLength == 1 ? make_float4( 1 ) /* faster */ : trainPathStates[jobIndex + stride * 2]; // path thoughput rgb
	const float4 hitData = hits[jobIndex];
	hits[jobIndex].z = __int_as_float( -1 ); // reset for next query
	const float bsdfPdf = T4.w;

	// derived data
	uint data = __float_as_uint( O4.w ); // prob.density of the last sampled dir, postponed because of MIS
	const float3 D = make_float3( D4 );
	const float2 DSphCoord = toSphericalCoord(D);
	//float3 throughput = make_float3( T4 );
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;
	const uint pathIdx = PATHIDX;
	const uint pixelIdx = pathIdx % (w * h);
	const uint sampleIdx = pathIdx / (w * h) + pass;
	const uint nrcTrainingSampleModulus = (w * h) / NRC_NUMTRAINRAYS;
	const uint trainSlotIdx = pixelIdx / nrcTrainingSampleModulus;

	NRC_DUMP("(JobIndex=%d, pathLength=%d) Origin: (%f, %f, %f) Direction: (%f, %f, %f) pixelIdx = %d, trainSlotIdx=%d", jobIndex, pathLength, O4.x, O4.y, O4.z, D4.x, D4.y, D4.z, pixelIdx, trainSlotIdx);

	// initialize depth in accumulator for DOF shader
	// if (pathLength == 1) accumulator[pixelIdx].w += PRIMIDX == NOHIT ? 10000 : HIT_T;

	// use skydome if we didn't hit any geometry
	if (PRIMIDX == NOHIT)
	{
	  NRC_DUMP("(JobIndex=%d, pathLength=%d) didn't hit any geometry", jobIndex, pathLength);
		float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = FLAGS & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );

		NRC_TRAINBUF(trainSlotIdx, pathLength, 0) = make_float4(
			/* Ray intersection - DUMMY */
			0.0f, 0.0f, 0.0f,
			/* roughness */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 1) = make_float4(
			/* Inbound ray direction */
			DSphCoord.x, DSphCoord.y,
			/* Surface normal X & Y, TODO: better val */
			0.0f, 0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 2) = make_float4(
			/* diffuse reflectance */
			-1.0f, -1.0f, -1.0f,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 3) = make_float4(
			/* specular reflectance */
			-1.0f, -1.0f, -1.0f,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 4) = make_float4(
			/* luminance output - represents luminance of this segment; accumulate to get real luminance later */
			skyPixel * (1.0f / bsdfPdf),
			/* flags */
			__uint_as_float(FLAGS | S_NRC_TRAINING_PATH_ENDED | S_NRC_DATA_VALID | S_NRC_TRAINING_DISCARD)
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 5) = make_float4(
			/* throughput factor - useless since no next rays exists; 
			 * (But use S_NRC_TRAINING_PATH_ENDED to determine if we have rays next) */
			0.0f, 0.0f, 0.0f,
			/* postponed bsdfPdf */
			1.0f
		);

#ifdef NRC_ENABLE_DEBUG_VIEW_PRIMARY
		if (debugView != nullptr && pathLength == 1) {
			debugView[pixelIdx] = make_float4(skyPixel * (1.0f / bsdfPdf), 0);
		}
#endif
		
		return;
	}

	// get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = RAY_O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );
	const float2 NSphCoord = toSphericalCoord(N);
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

	// we need to detect alpha in the shading code.
	// TODO: implement me
	// if (shadingData.flags & 1)
	// {
	// 	if (pathLength < NRC_MAXTRAINPATHLENGTH)
	// 	{
	// 		const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
	// 		pathStates[extensionRayIdx] = make_float4( I + D * geometryEpsilon, O4.w );
	// 		pathStates[extensionRayIdx + stride] = D4;
	// 		if (!(isfinite( T4.x + T4.y + T4.z ))) T4 = make_float4( 0, 0, 0, T4.w );  // TODO: why zero throughput??
	// 		pathStates[extensionRayIdx + stride * 2] = T4;
	// 	}
	// 	return;
	// }

	// stop on light
	if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
	{
	  NRC_DUMP("(JobIndex=%d, pathLength=%d) hit light\n", jobIndex, pathLength);
		const float DdotNL = -dot( D, N );
		float3 contribution = make_float3( 0 ); // initialization required.
		if (DdotNL > 0 /* lights are not double sided */)
		{
			if (pathLength == 1 || (FLAGS & S_SPECULAR) > 0 || connections == 0)
			{
				// accept light contribution if previous vertex was specular
				contribution = shadingData.color;
			}
			else
			{
				// last vertex was not specular: apply MIS
				const float3 lastN = UnpackNormal( __float_as_uint( D4.w ) );
				const CoreTri& tri = (const CoreTri&)instanceTriangles[PRIMIDX];
				const float lightPdf = CalculateLightPDF( D, HIT_T, tri.area, N );
				const float pickProb = LightPickProb( tri.ltriIdx, RAY_O, lastN, I /* the N at the previous vertex */ );
				// const float pickProb = LightPickProbLTree( tri.ltriIdx, RAY_O, lastN, I /* the N at the previous vertex */, seed );
				if ((bsdfPdf + lightPdf * pickProb) > 0) contribution = shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
			}
		}

		NRC_TRAINBUF(trainSlotIdx, pathLength, 0) = make_float4(
			/* Ray intersection */ RAY_O, /* roughness */ 0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 1) = make_float4(
			/* Inbound ray direction */
			DSphCoord.x, DSphCoord.y,
			/* Surface normal X & Y, TODO: better val */
			NSphCoord.x, NSphCoord.y
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 2) = make_float4(
			/* diffuse reflectance */
			//shadingData.color,
			contribution,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 3) = make_float4(
			/* specular reflectance */
			-1.0f, -1.0f, -1.0f,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 4) = make_float4(
			/* luminance output - represents luminance of this segment; accumulate to get real luminance later */
			contribution,
			/* flags */
			__uint_as_float(FLAGS | S_NRC_TRAINING_PATH_ENDED | S_NRC_DATA_VALID)
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 5) = make_float4(
			/* throughput factor - useless since no next rays exists; 
			 * (But use S_NRC_TRAINING_PATH_ENDED to determine if we have rays next) */
			0.0f, 0.0f, 0.0f,
			/* postponed bsdfPdf */
			1.0f
		);
		
		return;
	}

	// path regularization
	if (FLAGS & S_BOUNCED) shadingData.parameters.x |= 255u << 24; // set roughness to 1 after a bounce

	// detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.5f) FLAGS |= S_SPECULAR; /* detect pure speculars; skip NEE for these */ else FLAGS &= ~S_SPECULAR;

	// normal alignment for backfacing polygons
	const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

	// prepare random numbers
	float4 r4;
	if (sampleIdx < 64)
	{
		const uint x = ((pathIdx % w) + (shift & 127)) & 127;
		const uint y = ((pathIdx / w) + (shift >> 24)) & 127;
		r4 = blueNoiseSampler4( blueNoise, x, y, sampleIdx, 4 * pathLength - 4 );
	}
	else
	{
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}

	// next event estimation: connect eye path to light
	if ((FLAGS & S_SPECULAR) == 0 && connections != 0) // skip for specular vertices
	{
	  NRC_DUMP("(JobIndex=%d, pathLength=%d) emit shadow ray", jobIndex, pathLength);
		float pickProb, lightPdf = 0;
		float3 lightColor, L = RandomPointOnLight( r4.x, r4.y, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		// float3 lightColor, L = RandomPointOnLightLTree( r4.x, r4.y, seed, I, fN * faceDir, pickProb, lightPdf, lightColor, false ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN * faceDir );
		if (NdotL > 0 && lightPdf > 0)
		{
			float bsdfPdf;
		#ifdef BSDF_HAS_PURE_SPECULARS // see note in lambert.h
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf ) * ROUGHNESS;
		#else
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf );
		#endif
			if (bsdfPdf > 0)
			{
				// calculate potential contribution
				float3 contribution = sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf + bsdfPdf));
				FIXNAN_FLOAT3( contribution );
				CLAMPINTENSITY;
				// add fire-and-forget shadow ray to the connections buffer
				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction; TODO: eliminate this?
				connections[shadowRayIdx] = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), /* pathLength */ __int_as_float(pathLength) ); // O4
				connections[shadowRayIdx + stride * 2] = make_float4( L, dist - 2 * geometryEpsilon ); // D4
				connections[shadowRayIdx + stride * 2 * 2] = make_float4( contribution, __int_as_float( pathIdx ) ); // E4
			}
		}
	}

	// a maxium path length
	if (pathLength == NRC_MAXTRAINPATHLENGTH) {
	  NRC_DUMP("(JobIndex=%d, pathLength=%d) reached maximum length", jobIndex, pathLength);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 0) = make_float4(
			/* Ray intersection - dummy */ 0.0f, 0.0f, 0.0f,
			/* roughness */ ROUGHNESS
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 1) = make_float4(
			/* Inbound ray direction */
			DSphCoord.x, DSphCoord.y,
			/* Surface normal X & Y, TODO: better val */
			NSphCoord.x, NSphCoord.y
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 2) = make_float4(
			/* diffuse reflectance */
			shadingData.color,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 3) = make_float4(
			/* specular reflectance (TODO: figure out since SPECTINT have only one channel) */
			shadingData.color,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 4) = make_float4(
			/* luminance output - represents luminance of this segment; accumulate to get real luminance later */
			0.0f, 0.0f, 0.0f,
			/* flags */
			__uint_as_float(FLAGS | S_NRC_TRAINING_PATH_ENDED | S_NRC_DATA_VALID | S_NRC_TRAINING_DISCARD)
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 5) = make_float4(
			/* throughput factor - useless since no next rays exists; 
			 * (But use S_NRC_TRAINING_PATH_ENDED to determine if we have rays next) */
			0.0f, 0.0f, 0.0f,
			/* postponed bsdf */
			1.0f
		);
		return;
	}

	// evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	bool specular = false;
	
	// TODO: figure out why bsdf is float3
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r4.z, r4.w, RandomFloat( seed ), R, newBsdfPdf, specular );
	if (specular) FLAGS |= S_SPECULAR;

	// russian roulette
	// TODO: check if we can change chance but keep bsdfpdf working
	const float p = ((FLAGS & S_SPECULAR) || ((FLAGS & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );
	if (p < RandomFloat( seed ) || (newBsdfPdf < EPSILON || isnan( newBsdfPdf ))) {
	  NRC_DUMP("(JobIndex=%d, pathLength=%d) killed by russian roulette", jobIndex, pathLength);
		// killed by Russian roulette, or chance for given direction is too low

		NRC_TRAINBUF(trainSlotIdx, pathLength, 0) = make_float4(
			/* Ray intersection - dummy */ 0.0f, 0.0f, 0.0f,
			/* roughness */ ROUGHNESS
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 1) = make_float4(
			/* Inbound ray direction */
			DSphCoord.x, DSphCoord.y,
			/* Surface normal X & Y, TODO: better val */
			NSphCoord.x, NSphCoord.y
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 2) = make_float4(
			/* diffuse reflectance */
			shadingData.color,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 3) = make_float4(
			/* specular reflectance */
			shadingData.color,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 4) = make_float4(
			/* luminance output - represents luminance of this segment; accumulate to get real luminance later */
			0.0f, 0.0f, 0.0f,
			/* flags; also some extra stuff, used to calculate contribution accurately
			*	#define S_SPECULAR		1	// previous path vertex was specular
			*	#define S_BOUNCED		2	// path encountered a diffuse vertex
			*	#define S_VIASPECULAR	4	// path has seen at least one specular vertex
			*	#define S_BOUNCEDTWICE	8	// this core will stop after two diffuse bounces
			*	#define S_NRC_TRAINING_PATH_ENDED 16  // (NRC) this training path has ended
			*/
			__uint_as_float(FLAGS | S_NRC_TRAINING_PATH_ENDED | S_NRC_DATA_VALID | S_NRC_TRAINING_DISCARD)
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 5) = make_float4(
			/* throughput factor - useless since no next rays exists; 
			 * (But use S_NRC_TRAINING_PATH_ENDED to determine if we have rays next) */
			0.0f, 0.0f, 0.0f,
			/* postponed bsdfPdf */
			1.0f
		);
#ifdef NRC_ENABLE_DEBUG_VIEW_PRIMARY
		if (debugView != nullptr && pathLength == 1) {
			debugView[pixelIdx] = make_float4(shadingData.color * (1.0f / bsdfPdf), 0);
		}
#endif

	} else {
		// write extension ray, with compaction. Note: nvcc will aggregate automatically, 
		// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics 
	    NRC_DUMP("(JobIndex=%d, pathLength=%d) extension ray, Isect=(%.2f,%.2f,%.2f), Rdir=(%.2f,%.2f,%.2f)", jobIndex, pathLength, I.x, I.y, I.z, R.x, R.y, R.z);
		const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
		const uint packedNormal = PackNormal( fN * faceDir );
		float3 newThroughput = (1.0f / p) * bsdf * abs( dot( fN, R ) );
		FIXNAN_FLOAT3( newThroughput );

		if (!(FLAGS & S_SPECULAR)) FLAGS |= FLAGS & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED; else FLAGS |= S_VIASPECULAR;
		trainPathStates[extensionRayIdx] = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), __uint_as_float( FLAGS ) );
		trainPathStates[extensionRayIdx + stride] = make_float4( R, __uint_as_float( packedNormal ) );
		trainPathStates[extensionRayIdx + stride * 2] = make_float4( newThroughput, newBsdfPdf );

		// fill trainBuf
		NRC_TRAINBUF(trainSlotIdx, pathLength, 0) = make_float4(
			/* Ray intersection */ I,
			/* roughness */ ROUGHNESS
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 1) = make_float4(
			/* Inbound ray direction */
			DSphCoord.x, DSphCoord.y,
			/* Surface normal X & Y, TODO: better val */
			NSphCoord.x, NSphCoord.y
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 2) = make_float4(
			/* diffuse reflectance */
			shadingData.color,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 3) = make_float4(
			/* specular reflectance */
			shadingData.color,
			/* dummy */
			0.0f
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 4) = make_float4(
			/* luminance output - represents luminance of this segment; accumulate to get real luminance later */
			0.0f, 0.0f, 0.0f,
			/* flags */
			__uint_as_float(FLAGS | S_NRC_DATA_VALID)
		);
		NRC_TRAINBUF(trainSlotIdx, pathLength, 5) = make_float4(
			/* throughput factor - useless since no next rays exists; 
			 * (But use S_NRC_TRAINING_PATH_ENDED to determine if we have rays next) */
			newThroughput,
			/* postponed bsdfPdf */
			newBsdfPdf
		);

#ifdef NRC_ENABLE_DEBUG_VIEW_PRIMARY
		if (debugView != nullptr && pathLength == 1) {
			debugView[pixelIdx] = make_float4(shadingData.color * (1.0f / bsdfPdf), 0);
		}
#endif
	}


}

//  +-----------------------------------------------------------------------------+
//  |  shadeTrain                                                                 |
//  |  Host-side access point for the shadeKernel code.                     LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void shadeTrain( const int pathCount, float4* trainBuf, const uint stride,
	float4* trainPathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass, const int pathLength, const int scrwidth, const int scrheight, const float spreadAngle, float4* debugView)
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );

	NRC_DUMP("[DEBUG] trainBuf=%p, trainPathStates=%p, hits=%p, connections=%p, blueNoise=%p, debugView=%p", trainBuf, trainPathStates, hits, connections, blueNoise, debugView);
	shadeTrainKernel << <gridDim.x, 128 >> > (trainBuf, stride, trainPathStates, hits, connections, R0, shift, blueNoise,
		pass, pathLength, scrwidth, scrheight, spreadAngle, pathCount, debugView);
}


//  +-----------------------------------------------------------------------------+
//  |  shade                                                                      |
//  |  Host-side access point for the shadeKernel code.                     LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void shade( const int pathCount, float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int scrwidth, const int scrheight, const float spreadAngle )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
	shadeKernel << <gridDim.x, 128 >> > (accumulator, stride, pathStates, hits, connections, R0, shift, blueNoise,
		pass, probePixelIdx, pathLength, scrwidth, scrheight, spreadAngle, pathCount);
}

// EOF