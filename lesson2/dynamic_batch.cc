#include "builder/trt_builder.hpp"
#include "common/ilogger.hpp"
#include "infer/trt_infer.hpp"

void lesson2() {
  int batch_size = 5;
  // model , max batch size, onnx file, trt mode
  TRT::compile(TRT::Mode::FP32, batch_size, "../lesson2.onnx",
               "lesson2.fp32.trtmodel");

  // get infer
  auto infer = TRT::load_infer("lesson2.fp32.trtmodel");

  infer->input(0)->resize_single_dim(0, 2);
  // set value
  infer->input(0)->set_to(1.0f);
  infer->forward();

  // get output
  auto out = infer->output(0);
  INFO("Get output shape = %s", out->shape_string());
  for (int index = 0; index < out->channel(); ++index) {
    INFO("%f", out->at<float>(0, index));
  }

  for (int index = 0; index < out->channel(); ++index) {
    INFO("%f", out->at<float>(1, index));
  }
}

int main(int argc, char** argv) { lesson2(); }