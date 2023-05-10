#include "layer.hpp"

using namespace cv;

FakeQuantizeLayer::FakeQuantizeLayer(const Mat& inputLow_, const Mat& inputHigh_,
                                     const Mat& outputLow_, const Mat& outputHigh_,
                                     int levels_) {
  inputLow = inputLow_;
  inputHigh = inputHigh_;
  outputLow = outputLow_;
  outputHigh = outputHigh_;
  levels = levels_;
}

std::shared_ptr<ngraph::Node> FakeQuantizeLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  return std::make_shared<ngraph::op::v0::FakeQuantize>(inputs[0],
                                                        wrapMatToConstant(inputLow),
                                                        wrapMatToConstant(inputHigh),
                                                        wrapMatToConstant(outputLow),
                                                        wrapMatToConstant(outputHigh),
                                                        levels);
}
