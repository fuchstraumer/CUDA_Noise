// LodePNG is used to handle saving PNG images.
#include "lodepng\lodepng.h"
#include "Image.h"
#include <algorithm>

ImageWriter::ImageWriter(int _width, int _height) : width(_width), height(_height) {
	// Setup destination for raw data.
	rawData.resize(width * height);
}

void ImageWriter::WritePNG(const char * filename, int compression_level) const{
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
}

void ImageWriter::ConvertRawData() {
	// raw data has a size that is found by taking width * height in the constructor.
	// Pixel data is four times this: each single location has four seperate entries,
	// one entry per color channel per pixel. So, that's 4 * (size of raw data).
	pixelData.resize(rawData.size() * 4);
	// ^ note: using resize means this can be indexed like a normal C-style array.

	/*

		Indexing into pixel data:

		Getting the position of a pixel at coordinates (x,y) is done like so:
		[4 * height * y + 4 * x] = index into 1D vector to given a 2D position.

		The position retrieved immediately above is the position of the first
		color channel, however. So, this only gets us the red channel. Getting
		all of the channels could be done like this:

		1. Given Point (x,y), get current "offset" into the vector as:
			size_t idx = [4 * height * y + 4 * x]
		2. Now get color channel indices like so:
			size_t r, g, b, a;
			r = pixelData[idx + 0] = (set value of red channel)
			g = pixelData[idx + 1] = (set value of green channel)
			b = pixelData[idx + 2] = (set value of blue channel)
			a = pixelData[idx + 3] = (set value of alpha channel)
	*/

	// Scale raw data, mostly in 0.0f - 1.0f range, into unsigned char range.
	
	std::vector<unsigned char> tmpBuffer;
	tmpBuffer.resize(rawData.size());
	auto min_max = std::minmax_element(rawData.begin(), rawData.end());
	float max = *min_max.first;
	float min = *min_max.second;

	auto scaleRaw = [max, min](float val)->unsigned char {
		val = (val - min) / (max - min);
		// val += 0.50f;
		val *= 255.0f;
		if (val > 255.0f) {
			val = 255.0f;
		}
		if (val < 0.0f) {
			val = 0.0f;
		}
		unsigned char ret = static_cast<unsigned char>(val);
		return ret;
	};
	std::transform(rawData.begin(), rawData.end(), tmpBuffer.begin(), scaleRaw);
	// Copy values over to pixelData, for a grayscale image.
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			size_t idx = 4 * height * y + 4 * x;
			pixelData[idx + 0] = tmpBuffer[height * y + x];
			pixelData[idx + 1] = tmpBuffer[height * y + x];
			pixelData[idx + 2] = tmpBuffer[height * y + x];
			pixelData[idx + 3] = 255;
		}
	}

}

void ImageWriter::SetRawData(const std::vector<float>& raw){
	rawData = raw;
}

void ImageWriter::WriteBMP_Header(std::ofstream & output_stream) const{
	// TODO: Implement this. Need to do some more checks on insuring 4-byte alignment and correct sizing, somehow.
}
