// LodePNG is used to handle saving PNG images.
#include "lodepng\lodepng.h"
#include "Image.h"
#include <algorithm>

inline void unpack_16bit(int16_t in, uint8_t* dest) {
	dest[0] = static_cast<uint8_t>(in & 0x00ff);
	dest[1] = static_cast<uint8_t>((in & 0xff00) >> 8);
	return;
}

inline void unpack_32bit(int32_t integer, uint8_t* dest) {
	dest[0] = static_cast<uint8_t>((integer & 0x000000ff));
	dest[1] = static_cast<uint8_t>((integer & 0x0000ff00) >> 8);
	dest[2] = static_cast<uint8_t>((integer & 0x00ff0000) >> 16);
	dest[3] = static_cast<uint8_t>((integer & 0xff000000) >> 24);
	return;
}

inline void unpack_float(float in, uint8_t* dest) {
	uint8_t* bytes = reinterpret_cast<uint8_t*>(&in);
	dest[0] = *bytes++;
	dest[1] = *bytes++;
	dest[2] = *bytes++;
	dest[3] = *bytes++;
	return;
}

template<typename T>
inline auto convertRawData(const std::vector<float>& raw_data)->std::vector<T> {
	// We need to scale the data to fit in the range of the desired return type.
	T t_min, t_max;
	t_min = std::numeric_limits<T>::min();
	t_max = std::numeric_limits<T>::max();
	// Declare result vector so we can use std::transform shortly.
	std::vector<T> result;
	result.reserve(raw_data.size());
	// Get min/max values from raw data
	auto min_max = std::minmax_element(raw_data.begin(), raw_data.end());
	float max = *min_max.second;
	float min = *min_max.first;
	// Conversion lambda expression
	auto convert = [min, max, t_min, t_max](const float& val)->T {
		float result = ((val - min) / (min - max) * static_cast<float>(t_max - t_min)) + t_min;
		return result;
	};
	// Convert data
	std::transform(raw_data.begin(), raw_data.end(), std::back_inserter(result), convert);
	return result;
}

template<typename T>
inline auto convertRawData_Ranged(const std::vector<float>& raw_data, const float& lower_bound, const float& upper_bound)->std::vector<T> {
	// Declare result vector so we can use std::transform shortly.
	std::vector<T> result;
	result.reserve(raw_data.size());
	// Get min/max values from raw data
	auto min_max = std::minmax_element(raw_data.begin(), raw_data.end());
	float max = *min_max.second;
	float min = *min_max.first;
	// Conversion lambda expression
	auto convert = [min, max, lower_bound, upper_bound](const float& val)->T {
		float result = (val - min) / (min - max); // Normalize val into 0.0 - 1.0 range.
		result *= upper_bound;
		result += lower_bound;
		return static_cast<T>(result);
	};
	// Convert data
	std::transform(raw_data.begin(), raw_data.end(), std::back_inserter(result), convert);
	return result;
}

ImageWriter::ImageWriter(int _width, int _height) : width(_width), height(_height) {
	// Setup destination for raw data.
	rawData.resize(width * height);
}

void ImageWriter::FreeMemory(){
	rawData.clear();
	rawData.shrink_to_fit();
	pixelData.clear();
	pixelData.shrink_to_fit();
}

void ImageWriter::WriteBMP(const char * filename){
	uint8_t d[4];
	std::ofstream os;
	os.open(filename);
	if (os.fail() || os.bad()) {
		throw;
	}

	// Build header.
	os.write("BM", 2);

}

void ImageWriter::WritePNG(const char * filename, int compression_level){
	std::vector<unsigned char> tmpBuffer = convertRawData<unsigned char>(rawData);
	// Copy values over to pixelData, for a grayscale image.
	pixelData.resize(4 * tmpBuffer.size());
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			size_t idx = 4 * width * y + 4 * x;
			pixelData[idx + 0] = tmpBuffer[width * y + x];
			pixelData[idx + 1] = tmpBuffer[width * y + x];
			pixelData[idx + 2] = tmpBuffer[width * y + x];
			pixelData[idx + 3] = 255;
		}
	}
	if (compression_level == 0) {
		// Saves uncompressed image using "pixelData" to "filename"
		unsigned err;
		err = lodepng::encode(filename, &pixelData[0], width, height);
		if (!err) {
			return;
		}
		else {
			std::cout << "Error encoding image, code " << err << ": " << lodepng_error_text(err) << std::endl;
		}
	}
	else {
		// TODO: Implement compression. Need to study parameters of LodePNG more. See: https://raw.githubusercontent.com/lvandeve/lodepng/master/examples/example_optimize_png.cpp
	}
	pixelData.clear();
	pixelData.shrink_to_fit();
}

void ImageWriter::WriteTER(const char* filename) {

	// Output data size.
	size_t byte_width = width * sizeof(int16_t);
	size_t total_size = byte_width * height;

	// Buffer a line of data at a time.
	std::vector<uint8_t> lineBuffer;
	lineBuffer.reserve(byte_width);

	// Open output file stream
	std::ofstream out;
	out.clear();

	// Oopen output file in binary mode
	out.open(filename, std::ios::out | std::ios::binary);

	// Build header.
	// height_scale - 0.50f in divisor means 0.25m between sampling points.
	int16_t height_scale = static_cast<int16_t>(floorf(32768.0f / 30.0f));
	// Buffer used for unpacking various types into correct format for writing to stream.
	uint8_t buffer[4];
	out.write("TERRAGENTERRAIN ", 16); // First element of header.
	// Write terrain size.
	out.write("SIZE", 4);
	unpack_16bit(static_cast<int16_t>(std::min(width, height) - 1), buffer);
	out.write(reinterpret_cast<char*>(buffer), 2);
	out.write("\0\0", 2);
	// X dim.
	out.write("XPTS", 4);
	unpack_16bit(static_cast<int16_t>(width), buffer);
	out.write(reinterpret_cast<char*>(buffer), 2);
	out.write("\0\0", 2);
	// Y dim
	out.write("YPTS", 4);
	unpack_16bit(static_cast<int16_t>(height), buffer);
	out.write(reinterpret_cast<char*>(buffer), 2);
	out.write("\0\0", 2);
	// Write scale.
	out.write("SCAL", 4);
	// point-sampling scale is XYZ quantity, write same value three times.
	unpack_float(15.0f, buffer);
	out.write(reinterpret_cast<char*>(buffer), 4);
	out.write(reinterpret_cast<char*>(buffer), 4);
	out.write(reinterpret_cast<char*>(buffer), 4);
	// Write height scale
	out.write("ALTW", 4);
	unpack_16bit(height_scale, buffer);
	out.write(reinterpret_cast<char*>(buffer), 4);
	out.write("\0\0", 2);
	if (out.fail() || out.bad()) {
		throw;
	}

	// Build and write each horizontal line to the file.
	std::vector<int16_t> height_values = convertRawData_Ranged<int16_t>(rawData, 0.0f, 1000.0f);
	for (size_t i = 0; i < height_values.size(); ++i) {
		uint8_t buffer[2];
		unpack_16bit(height_values[i], buffer);
		out.write(reinterpret_cast<char*>(buffer), 2);
	}
	
	out.write("EOF", 4);
	// Close output file.
	out.close();
}

void ImageWriter::SetRawData(const std::vector<float>& raw){
	rawData = raw;
}

std::vector<float> ImageWriter::GetRawData() const{
	return rawData;
}

void ImageWriter::WriteBMP_Header(std::ofstream & output_stream) const{
	// TODO: Implement this. Need to do some more checks on insuring 4-byte alignment and correct sizing, somehow.
}
