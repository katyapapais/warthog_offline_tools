#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

static float getFloatFromByteArray(char *byteArray, uint index) {
    return *( (float *)(byteArray + index));
}

int main() {
	std::string path = "/media/keenan/autorontossd1/2020_11_03_2/lidar/1604442643567733662.bin";
	std::ifstream ifs("/media/keenan/autorontossd1/2020_11_03_2/lidar/1604442643567733662.bin", std::ios::binary);
	
	std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});

	int float_offset = 4;
	int fields = 6;
	int N = buffer.size() / (float_offset * fields);	
	int point_step = float_offset * fields;
	Eigen::MatrixXf pc = Eigen::MatrixXf::Zero(N, fields);
	int l = 0;
	for (int i = 0; i < buffer.size(); i += point_step) {
		pc(l, 0) = getFloatFromByteArray(buffer.data(), i);
		pc(l, 1) = getFloatFromByteArray(buffer.data(), i + float_offset);
		pc(l, 2) = getFloatFromByteArray(buffer.data(), i + float_offset * 2);
		pc(l, 3) = getFloatFromByteArray(buffer.data(), i + float_offset * 3);
		pc(l, 4) = getFloatFromByteArray(buffer.data(), i + float_offset * 4);
		pc(l, 5) = getFloatFromByteArray(buffer.data(), i + float_offset * 5);
		l++;
	}

	std::cout << buffer.size() << std::endl;
	std::cout << pc.cols() << " " << pc.rows() << std::endl;
	std::cout << pc.block(0, 0, 1, 6) << std::endl;
	std::cout << pc.block(N - 1, 0, 1, 6) << std::endl;
	return 0;
}
