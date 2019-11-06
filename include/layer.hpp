#include <opencv2/opencv.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <ngraph/ngraph.hpp>

using namespace cv;

std::shared_ptr<ngraph::op::Constant> wrapMatToConstant(const Mat& m,
                                                        const std::vector<size_t>& shape = {});

class Layer {
public:
  Layer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input) = 0;

protected:
  std::string name;
};

class ConvolutionLayer : public Layer {
public:
  ConvolutionLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input);

private:
  Mat weights, biases;
  std::vector<size_t> strides, dilations;
  std::vector<std::ptrdiff_t> pads_begin, pads_end;
  std::string padMode;  // "VALID", "SAME" or empty
};

class ReLULayer : public Layer {
public:
  ReLULayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input);
};

class LRNLayer : public Layer {
public:
  LRNLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input);

private:
  double alpha, beta, bias;
  bool normBySize;
  size_t size;
  std::string normType;
};

class PoolingLayer : public Layer {
public:
  PoolingLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input);

private:
  std::vector<size_t> kernel_size, strides, pads_begin, pads_end;
  std::string padMode;  // "VALID", "SAME" or empty
  bool ceilMode;  //
};

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input);

private:
  Mat weights, biases;
};

class SoftMaxLayer : public Layer {
public:
  SoftMaxLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::shared_ptr<ngraph::Node> input);
};
