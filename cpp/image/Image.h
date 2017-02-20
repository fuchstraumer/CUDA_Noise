#pragma once
#ifndef IMAGE_H
#define IMAGE_H
#include "common\CommonInclude.h"

/*

	Class - ImageWriter

	Writes an image to a file given data. Can write completely uncompressed
	BMP images, mostly uncompressed PNG images, or highly compressed PNG images.

	Uses LodePNG for PNG stuff. BMP stuff is written into this class.

*/

class ImageWriter {
public:

	// Initializes image and allocates space in vector used to hold raw data.
	ImageWriter(int width, int height);

	// Writes data contained in this image to file with given name.
	void WriteBMP(const char* filename) const;

	// Writes data contained in this image to PNG. Compression level set by
	// second parameter, but is optional. Defaults to uncompressed.
	void WritePNG(const char* filename, int compression_level = 0) const;

	// Attaches a module to this object. Calls the module when WriteImage methods are called.
	// TODO: Implement this.

	// Converts raw image data in "rawData" to pixel data, with a bit depth of 32.
	// This means that each pixel has four 8-byte (unsigned char) elements.
	void ConvertRawData();

	// Sets rawData
	void SetRawData(const std::vector<float>& raw);

	// Gets rawData
	std::vector<unsigned char> GetRawData() const;

private:
	// Holds raw data grabbed from one of the noise modules.
	std::vector<float> rawData;

	// Holds pixel data converted from rawData.
	std::vector<unsigned char> pixelData;

	// Dimensions of this image
	int width, height;

	// Writes BMP header.
	void WriteBMP_Header(std::ofstream& output_stream) const;
};

#endif // !IMAGE_H
