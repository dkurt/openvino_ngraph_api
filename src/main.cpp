#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

#include "layer.hpp"

using namespace cv;

static void genData(const std::vector<size_t>& dims, Mat& m, InferenceEngine::Blob::Ptr& dataPtr);

void normAssert(Mat ref, Mat test, double l1 = 0.00001, double lInf = 0.0001);

void normAssertDetections(Mat ref, Mat out, double confThreshold = 1e-6,
                          double scores_diff = 1e-5, double boxes_iou_diff = 1e-4);

void normAssertDetections(
        const std::vector<int>& refClassIds,
        const std::vector<float>& refScores,
        const std::vector<cv::Rect2d>& refBoxes,
        const std::vector<int>& testClassIds,
        const std::vector<float>& testScores,
        const std::vector<cv::Rect2d>& testBoxes,
        double confThreshold /*= 0.0*/,
        double scores_diff /*= 1e-5*/, double boxes_iou_diff /*= 1e-4*/);

std::vector<cv::Rect2d> matToBoxes(const cv::Mat& m);

int main(int argc, char** argv) {
  // Read source network from file.
  if (argc < 2) {
    std::cout << "Please specify model to test: alexnet, squeezenet, ssd" << std::endl;
    return 1;
  }
  std::string modelName = argv[1];

  dnn::Net cvNet;
  std::vector<std::vector<int> > inLayerShapes;
  std::vector<std::vector<int> > outLayerShapes;
  if (modelName == "alexnet") {
    cvNet = dnn::readNet(utils::fs::join("..", "bvlc_alexnet.caffemodel"),
                         utils::fs::join("..", "bvlc_alexnet.prototxt"));
    inLayerShapes = std::vector<std::vector<int> >({{1, 3, 227, 227}});
  } else if (modelName == "squeezenet") {
    cvNet = dnn::readNet(utils::fs::join("..", "squeezenet_v1.1.caffemodel"),
                         utils::fs::join("..", "squeezenet_v1.1.prototxt"));
    inLayerShapes = std::vector<std::vector<int> >({{1, 3, 227, 227}});
  } else if (modelName == "ssd") {
    cvNet = dnn::readNet(utils::fs::join("..", "MobileNetSSD_deploy.caffemodel"),
                         utils::fs::join("..", "MobileNetSSD_deploy.prototxt"));
    inLayerShapes = std::vector<std::vector<int> >({{1, 3, 300, 300}});
  }

  // Create an input nGraph node.
  std::vector<size_t> inpShape(inLayerShapes[0].begin(), inLayerShapes[0].end());
  std::shared_ptr<ngraph::Node> input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape(inpShape));
  std::shared_ptr<ngraph::Node> node = input;

  // Map layers indices to nGraph nodes.
  std::map<int, std::shared_ptr<ngraph::Node> > nodes;
  nodes[0] = input;

  // Iterate over layers and build nGraph model.
  for (int i = 0, n = cvNet.getLayerNames().size(); i < n; ++i) {
    Ptr<dnn::Layer> cvLayer = cvNet.getLayer(i + 1);

    std::vector<std::shared_ptr<ngraph::Node> > inputs;
    for (const auto& inp : cvNet.getLayerInputs(i + 1)) {
      int id = cvNet.getLayerId(inp->name);
      CV_Assert(nodes.find(id) != nodes.end());
      inputs.push_back(nodes[id]);
    }
    CV_Assert(!inputs.empty());

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
    } else if (cvLayer->type == "Softmax") {
      l = Ptr<Layer>(new SoftMaxLayer(cvLayer));
    } else if (cvLayer->type == "Concat") {
      l = Ptr<Layer>(new ConcatLayer(cvLayer));
    } else if (cvLayer->type == "Flatten") {
      l = Ptr<Layer>(new FlattenLayer(cvLayer));
    } else if (cvLayer->type == "Permute") {
      l = Ptr<Layer>(new PermuteLayer(cvLayer));
    } else if (cvLayer->type == "PriorBox") {
      l = Ptr<Layer>(new PriorBoxLayer(cvLayer));
    } else if (cvLayer->type == "Reshape") {
      l = Ptr<Layer>(new ReshapeLayer(cvLayer));
    } else if (cvLayer->type == "DetectionOutput") {
      l = Ptr<Layer>(new DetectionOutputLayer(cvLayer));
    } else if (cvLayer->type == "Dropout") {
      nodes[i + 1] = node;
      continue;
    } else {
      CV_Error(Error::StsNotImplemented, "Unknown layer type: " + cvLayer->type);
    }

    node = l->initNGraph(inputs);
    nodes[i + 1] = node;
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

  if (modelName == "ssd") {
    // Tets on real data.
    dnn::blobFromImage(imread("../example.jpg"), inputMat,
                       0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5));
  }

  for (auto& it : cnn.getOutputsInfo())
  {
      genData(it.second->getTensorDesc().getDims(), ieOutputMat, outputBlobs[it.first]);
  }
  infRequest.SetOutput(outputBlobs);

  // Run Inference Engine network.
  infRequest.Infer();

  // // Run OpenCV network with the same inputs
  cvNet.setInput(inputMat);
  cvNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  Mat cvOut = cvNet.forward();

  // Compare outputs from OpenCV and Inference Engine.
  if (modelName == "ssd")
    normAssertDetections(ieOutputMat, cvOut);
  else
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

void normAssertDetections(Mat ref, Mat out, double confThreshold,
                          double scores_diff, double boxes_iou_diff)
{
    CV_Assert(ref.total() % 7 == 0);
    CV_Assert(out.total() % 7 == 0);
    ref = ref.reshape(1, ref.total() / 7);
    out = out.reshape(1, out.total() / 7);

    cv::Mat refClassIds, testClassIds;
    ref.col(1).convertTo(refClassIds, CV_32SC1);
    out.col(1).convertTo(testClassIds, CV_32SC1);
    std::vector<float> refScores(ref.col(2)), testScores(out.col(2));
    std::vector<cv::Rect2d> refBoxes = matToBoxes(ref.colRange(3, 7));
    std::vector<cv::Rect2d> testBoxes = matToBoxes(out.colRange(3, 7));
    normAssertDetections(refClassIds, refScores, refBoxes, testClassIds, testScores,
                         testBoxes, confThreshold, scores_diff, boxes_iou_diff);
}

void normAssertDetections(
        const std::vector<int>& refClassIds,
        const std::vector<float>& refScores,
        const std::vector<cv::Rect2d>& refBoxes,
        const std::vector<int>& testClassIds,
        const std::vector<float>& testScores,
        const std::vector<cv::Rect2d>& testBoxes,
        double confThreshold /*= 0.0*/,
        double scores_diff /*= 1e-5*/, double boxes_iou_diff /*= 1e-4*/)
{
    std::vector<bool> matchedRefBoxes(refBoxes.size(), false);
    for (int i = 0; i < testBoxes.size(); ++i)
    {
        double testScore = testScores[i];
        if (testScore < confThreshold)
            continue;

        int testClassId = testClassIds[i];
        const cv::Rect2d& testBox = testBoxes[i];
        bool matched = false;
        for (int j = 0; j < refBoxes.size() && !matched; ++j)
        {
            if (!matchedRefBoxes[j] && testClassId == refClassIds[j] &&
                std::abs(testScore - refScores[j]) < scores_diff)
            {
                double interArea = (testBox & refBoxes[j]).area();
                double iou = interArea / (testBox.area() + refBoxes[j].area() - interArea);
                if (std::abs(iou - 1.0) < boxes_iou_diff)
                {
                    matched = true;
                    matchedRefBoxes[j] = true;
                }
            }
        }
        if (matched)
            std::cout << format("[ OK ] matched class %d with confidence %.2f", testClassId, testScore) << std::endl;
        else
            std::cout << cv::format("[ FAILED ] Unmatched prediction: class %d score %f box ",
                                    testClassId, testScore) << testBox << std::endl;
    }

    // Check unmatched reference detections.
    for (int i = 0; i < refBoxes.size(); ++i)
    {
        if (!matchedRefBoxes[i] && refScores[i] > confThreshold)
        {
            std::cout << cv::format("[ FAILED ] Unmatched reference: class %d score %f box ",
                                    refClassIds[i], refScores[i]) << refBoxes[i] << std::endl;
            CV_CheckLE(refScores[i], (float)confThreshold, "");
        }
    }
}

std::vector<cv::Rect2d> matToBoxes(const cv::Mat& m)
{
    CV_CheckEQ(m.type(), CV_32FC1, "");
    CV_CheckEQ(m.dims, 2, "");
    CV_CheckEQ(m.cols, 4, "");

    std::vector<cv::Rect2d> boxes(m.rows);
    for (int i = 0; i < m.rows; ++i)
    {
        CV_Assert(m.row(i).isContinuous());
        const float* data = m.ptr<float>(i);
        double l = data[0], t = data[1], r = data[2], b = data[3];
        boxes[i] = cv::Rect2d(l, t, r - l, b - t);
    }
    return boxes;
}
