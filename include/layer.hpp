#include <opencv2/opencv.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <ngraph/ngraph.hpp>

using namespace cv;

std::shared_ptr<ngraph::op::Constant> wrapMatToConstant(const Mat& m,
                                                        const std::vector<size_t>& shape = {});

class Layer {
public:
  Layer(const std::string& name = "");

  Layer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) = 0;

protected:
  std::string name;
};

class ConvolutionLayer : public Layer {
public:
  ConvolutionLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  Mat weights, biases;
  std::vector<size_t> strides, dilations;
  std::vector<std::ptrdiff_t> pads_begin, pads_end;
  std::string padMode;  // "VALID", "SAME" or empty
};

class ReLULayer : public Layer {
public:
  ReLULayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);
};

class LRNLayer : public Layer {
public:
  LRNLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  double alpha, beta, bias;
  bool normBySize;
  size_t size;
  std::string normType;
};

class PoolingLayer : public Layer {
public:
  PoolingLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  std::vector<size_t> kernel_size, strides, pads_begin, pads_end;
  std::string padMode;  // "VALID", "SAME" or empty
  bool ceilMode, globalPooling;
  std::string type;
};

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  Mat weights, biases;
};

class SoftMaxLayer : public Layer {
public:
  SoftMaxLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);
};

class ConcatLayer : public Layer {
public:
  ConcatLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  int axis;
};

class FlattenLayer : public Layer {
public:
  FlattenLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);
};

class PermuteLayer : public Layer {
public:
  PermuteLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);
};

class PriorBoxLayer : public Layer {
public:
  PriorBoxLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  int idx;
};

class ReshapeLayer : public Layer {
public:
  ReshapeLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  std::vector<size_t> shape;
};

class DetectionOutputLayer : public Layer {
public:
  DetectionOutputLayer(Ptr<dnn::Layer> cvLayer);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);
};

class FakeQuantizeLayer : public Layer {
public:
  FakeQuantizeLayer(const Mat& inputLow, const Mat& inputHigh,
                    const Mat& outputLow, const Mat& outputHigh,
                    int levels);

  virtual std::shared_ptr<ngraph::Node> initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs);

private:
  Mat inputLow, inputHigh, outputLow, outputHigh;
  int levels;
};
