#include "ir_weights.hpp"

#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace cv;

inline std::string get(const std::string& field, const std::string& line) {
  int idx = line.find(field + "=\"") + field.size() + 2;
  return line.substr(idx, line.find('"', idx) - idx);
}

IRWeights::IRWeights(const std::string& xmlPath, const std::string& binPath) {
  // We just reads xml file to extract Const nodes and corresponding shapes, offsets and sizes.
  std::ifstream xml(xmlPath);
  // Binary files is used to extract origin weights and quantization normalizations.
  std::ifstream bin(binPath);

  std::string line;
  while (std::getline(xml, line)) {
    int idx = line.find("offset=");
    if (idx < 0)
      continue;

    if (get("element_type", line) != "f32")
      continue;

    int offset, size;
    std::vector<int> shape;

    std::istringstream(get("offset", line)) >> offset;
    std::istringstream(get("size", line)) >> size;
    std::istringstream iss(get("shape", line));
    std::string token;
    while (std::getline(iss, token, ',')) {
      shape.resize(shape.size() + 1);
      std::istringstream(token) >> shape.back();
    }
    if (shape.empty())
        shape.push_back(1);

    Mat blob(shape, CV_32F);
    bin.seekg(offset, bin.beg);
    bin.read(blob.ptr<char>(), size);
    weights.push(blob);
  }
}

Mat IRWeights::next() {
  Mat w = weights.front();
  weights.pop();
  return w;
}
