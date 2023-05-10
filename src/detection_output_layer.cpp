#include "layer.hpp"

#include <ngraph/op/experimental/layers/detection_output.hpp>

using namespace cv;

DetectionOutputLayer::DetectionOutputLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {}

std::shared_ptr<ngraph::Node> DetectionOutputLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  auto& box_logits  = inputs[0];
  auto& class_preds = inputs[1];
  auto& proposals   = inputs[2];

  ngraph::op::DetectionOutputAttrs attrs;
  attrs.num_classes                = 21;
  attrs.background_label_id        = 0;
  attrs.top_k                      = 100;
  attrs.variance_encoded_in_target = false;
  attrs.keep_top_k                 = {100};
  attrs.nms_threshold              = 0.45;
  attrs.confidence_threshold       = 0.25;
  attrs.share_location             = true;
  attrs.clip_before_nms            = false;
  attrs.code_type                  = "caffe.PriorBoxParameter.CENTER_SIZE";
  attrs.normalized                 = true;

  return std::make_shared<ngraph::op::DetectionOutput>(box_logits, class_preds,
             proposals, attrs);
}
