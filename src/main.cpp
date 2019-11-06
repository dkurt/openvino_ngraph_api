#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

#include "layer.hpp"

using namespace cv;

static void genData(const std::vector<size_t>& dims, Mat& m, InferenceEngine::Blob::Ptr& dataPtr);

void normAssert(Mat ref, Mat test, double l1 = 0.00001, double lInf = 0.0001);

int main(int argc, char** argv) {
  // Read source network from file.
  dnn::Net cvNet = dnn::readNet(utils::fs::join("..", "bvlc_alexnet.caffemodel"),
                                utils::fs::join("..", "bvlc_alexnet.prototxt"));

  // Get input shapes. Note: works for Caffe and IR models.
  std::vector<std::vector<int> > inLayerShapes;
  std::vector<std::vector<int> > outLayerShapes;
  cvNet.getLayerShapes(std::vector<int>(), 0, inLayerShapes, outLayerShapes);
  CV_Assert(!inLayerShapes.empty() && inLayerShapes[0].size() == 4);

  // Create an input nGraph node.
  std::vector<size_t> inpShape(inLayerShapes[0].begin(), inLayerShapes[0].end());
  std::shared_ptr<ngraph::Node> input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape(inpShape));
  std::shared_ptr<ngraph::Node> node = input;

  // Iterate over layers and build nGraph model.
  for (int i = 0, n = cvNet.getLayerNames().size(); i < n; ++i) {
    Ptr<dnn::Layer> cvLayer = cvNet.getLayer(i + 1);

    Ptr<Layer> l;
    if (cvLayer->type == "") {
    } else if (cvLayer->type == "Convolution") {
      l = Ptr<Layer>(new ConvolutionLayer(cvLayer));
    } else if (cvLayer->type == "ReLU") {
      l = Ptr<Layer>(new ReLULayer(cvLayer));
    } else if (cvLayer->type == "LRN") {
      l = Ptr<Layer>(new LRNLayer(cvLayer));
    } else if (cvLayer->type == "Pooling") {
      l = Ptr<Layer>(new PoolingLayer(cvLayer));
    } else if (cvLayer->type == "InnerProduct") {
      l = Ptr<Layer>(new FullyConnectedLayer(cvLayer));
    } else if (cvLayer->type == "Dropout") {
      continue;
    } else if (cvLayer->type == "Softmax") {
      l = Ptr<Layer>(new SoftMaxLayer(cvLayer));
    } else {
      CV_Error(Error::StsNotImplemented, "Unknown layer type: " + cvLayer->type);
    }

    node = l->initNGraph(node);
  }

  // Convert nGraph function to Inference Engine's network.
  auto ngraph_function = std::make_shared<ngraph::Function>(node, ngraph::ParameterVector{std::dynamic_pointer_cast<ngraph::op::Parameter>(input)});
  InferenceEngine::CNNNetwork cnn = InferenceEngine::CNNNetwork(ngraph_function);
  InferenceEngine::Core ie;
  InferenceEngine::ExecutableNetwork netExec = ie.LoadNetwork(cnn, "CPU");
  InferenceEngine::InferRequest infRequest = netExec.CreateInferRequest();

  // Prepare input and output blobs.
  Mat inputMat, ieOutputMat;
  InferenceEngine::BlobMap inputBlobs, outputBlobs;
  CV_Assert(cnn.getInputsInfo().size() == 1);
  CV_Assert(cnn.getOutputsInfo().size() == 1);
  for (auto& it : cnn.getInputsInfo())
  {
      genData(it.second->getTensorDesc().getDims(), inputMat, inputBlobs[it.first]);
  }
  infRequest.SetInput(inputBlobs);

  for (auto& it : cnn.getOutputsInfo())
  {
      genData(it.second->getTensorDesc().getDims(), ieOutputMat, outputBlobs[it.first]);
  }
  infRequest.SetOutput(outputBlobs);

  // Run Inference Engine network.
  infRequest.Infer();

  // Run OpenCV network with the same inputs
  cvNet.setInput(inputMat);
  Mat cvOut = cvNet.forward();

  // Compare outputs from OpenCV and Inference Engine.
  normAssert(ieOutputMat, cvOut);

  return 0;
}

void genData(const std::vector<size_t>& dims, Mat& m, InferenceEngine::Blob::Ptr& dataPtr)
{
    m.create(std::vector<int>(dims.begin(), dims.end()), CV_32F);
    randu(m, -1, 1);
    dataPtr = InferenceEngine::make_shared_blob<float>({
        InferenceEngine::Precision::FP32,
        dims,
        InferenceEngine::Layout::ANY}, (float*)m.data);
}

void normAssert(Mat ref, Mat test, double l1, double lInf)
{
    double normL1 = norm(ref, test, cv::NORM_L1) / ref.total();
    CV_CheckLE(normL1, l1, "l1");

    double normInf = norm(ref, test, cv::NORM_INF);
    CV_CheckLE(normInf, lInf, "lInf");

    std::cout << "l1 diff: " << normL1 << std::endl;
    std::cout << "lInf diff: " << normInf << std::endl;
}
