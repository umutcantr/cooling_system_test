// Basic computer cooling system evaluation tool. The program measure first temperature of gpu and gives gpu task. 
// After task finished, it calculates in what time returned to idle or first temperature.
// It is best to execute program that other programs not working.
// Author: Umutcan Genç

#include "opencl.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace std;

// Private functions
void gpu_temp_in_limit(int gpu_temp);
int get_temp_value(string file_name);
void write_command_line(int gpu_temp);
void run_opencl_task();
void gpu_cooling_time();

// Global variables
int gpu_temp_limit = 88;
int gpu_temp;
int first_gpu_temp;
string filename = "nvidia_smi.txt";
string command = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader";
string nvidia_smý_command = command + " > " + filename;

int main() {
    // Get idle gpu temp
    first_gpu_temp = get_temp_value(filename);
    write_command_line(first_gpu_temp);

    run_opencl_task();

    // Last(running) gpu temp
    gpu_temp = get_temp_value(filename);
    write_command_line(gpu_temp);
    gpu_temp_in_limit(gpu_temp);

    // Calculate gpu cooling time
    gpu_cooling_time();

	wait();
	return 0;
}

// Checks temperature in the max limit
void gpu_temp_in_limit(int temp) {
    if (temp > gpu_temp_limit) {
        cout << "|| Gpu temp is high. Consider to maintenance." << endl;
    }
    else {
        cout << "| Gpu temp is normal." << endl;
    }
}

// Get temperature value using nvidia-smý tool
int get_temp_value(string filename) {
    // Sends nvidia-smý command and stores incoming value in txt file
    system(nvidia_smý_command.c_str());

    ifstream file(filename);
    if (file.is_open()) {
        stringstream ss;
        string s;
        int temp;
        ss << file.rdbuf();
        s = ss.str();
        temp = stoi(s);
        file.close();
        return temp;
    }
    else {
        cout << "Cant open the file!" << endl;
        return 0;
    }
}

// Write temperature to command line 
void write_command_line(int gpu_temp) {
    cout << "| Gpu temperature is:" << gpu_temp << endl;
}

// Run gpu task which uses OpenCL-Wrapper project;https://github.com/ProjectPhysX/OpenCL-Wrapper
void run_opencl_task(void) {
    Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

    const uint N = 300000000u; // size of vectors
    Memory<float> A(device, N); // allocate memory on both host and device
    Memory<float> B(device, N);
    Memory<float> C(device, N);

    Kernel random_kernel(device, N, "random_kernel", A, B, C); // kernel that runs on the device

    for (uint n = 0u; n < N; n++) {
        A[n] = 3.0f; // initialize memory
        B[n] = 2.0f;
        C[n] = 1.0f;
    }

    //print_info("Value before kernel execution: C[0] = "+to_string(C[0]));

    A.write_to_device(); // copy data from host memory to device memory
    B.write_to_device();
    random_kernel.run(); // run add_kernel on the device
    C.read_from_device(); // copy data from device memory to host memory

    //print_info("Value after kernel execution: C[0] = "+to_string(C[0]));
}

// Calculates elapsed time by comparing first temp and current temp
void gpu_cooling_time() {
    cout << "| Calculating gpu cooling time..." << endl;
    clock_t begin = clock();

    while (get_temp_value(filename) != (first_gpu_temp + 1));

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "| Elapsed time:" << elapsed_secs << " second" << endl;
}