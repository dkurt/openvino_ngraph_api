#include "layer.hpp"

#include <ngraph/op/experimental/layers/prior_box.hpp>

using namespace cv;

PriorBoxLayer::PriorBoxLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  if (cvLayer->name == "conv11_mbox_priorbox") {
    idx = 0;
  } else if (cvLayer->name == "conv13_mbox_priorbox") {
    idx = 1;
  } else if (cvLayer->name == "conv14_2_mbox_priorbox") {
    idx = 2;
  } else if (cvLayer->name == "conv15_2_mbox_priorbox") {
    idx = 3;
  } else if (cvLayer->name == "conv16_2_mbox_priorbox") {
    idx = 4;
  } else if (cvLayer->name == "conv17_2_mbox_priorbox") {
    idx = 5;
  } else {
    CV_Error(Error::StsNotImplemented, "PriorBox with name: " + cvLayer->name);
  }
}

std::shared_ptr<ngraph::Node> PriorBoxLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  auto layer = inputs[0];
  auto image = inputs[1];
  auto layer_shape = std::make_shared<ngraph::op::ShapeOf>(layer);
  auto image_shape = std::make_shared<ngraph::op::ShapeOf>(image);

  auto lower_bounds = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{2});
  auto upper_bounds = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{4});
  auto strides      = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{1});

  auto slice_layer = std::make_shared<ngraph::op::v1::StridedSlice>(layer_shape,
                                      lower_bounds, upper_bounds, strides, std::vector<int64_t>{}, std::vector<int64_t>{});
  auto slice_image = std::make_shared<ngraph::op::v1::StridedSlice>(image_shape,
                                      lower_bounds, upper_bounds, strides, std::vector<int64_t>{}, std::vector<int64_t>{});

  static std::vector<float> minSizes{60, 105, 150, 195, 240, 285};
  static std::vector<float> maxSizes{150, 195, 240, 285, 300};

  ngraph::op::PriorBoxAttrs attrs;
  attrs.min_size = std::vector<float>{minSizes[idx]};
  attrs.max_size = idx > 0 ? std::vector<float>{maxSizes[idx - 1]} : std::vector<float>();
  attrs.aspect_ratio = idx > 0 ? std::vector<float>{2.0, 3.0} : std::vector<float>{2.0};
  attrs.clip = false;
  attrs.flip = true;
  attrs.variance = {0.1, 0.1, 0.2, 0.2};
  attrs.offset = 0.5;
  attrs.step = static_cast<float>(inputs[1]->get_shape()[2]) / inputs[0]->get_shape()[2];
  attrs.scale_all_sizes = true;

  auto priorBox = std::make_shared<ngraph::op::PriorBox>(slice_layer, slice_image, attrs);
  auto axis = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
  auto unsqueeze = std::make_shared<ngraph::op::Unsqueeze>(priorBox, axis);
  return unsqueeze;
}
