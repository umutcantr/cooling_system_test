#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void random_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = ((A[n]*B[n]) * B[n] * A[n]) / (B[n] / A[n]) / B[n];
	B[n] = ((A[n] * B[n]) * B[n] * A[n]) / (B[n] / A[n]) / B[n];
	A[n] = ((A[n] * B[n]) * B[n] * A[n]) / (B[n] / A[n]) / B[n];
	C[n] = ((A[n] * B[n]) * B[n] * A[n]) / (B[n] / A[n]) / B[n];
	C[n] = (A[n] * B[n]) / C[n];
}



);} // ############################################################### end of OpenCL C code #####################################################################