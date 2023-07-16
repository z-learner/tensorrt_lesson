#include "builder/trt_builder.hpp"
#include "common/ilogger.hpp"
#include "infer/trt_infer.hpp"

void lesson1() {
  // model , max batch size, onnx file, trt mode
  TRT::compile(TRT::Mode::FP32, 1, "../lesson1.onnx", "lesson1.fp32.trtmodel");

  // get infer
  auto infer = TRT::load_infer("lesson1.fp32.trtmodel");
  // set value
  infer->input(0)->set_to(1.0f);
  infer->forward();

  // get output
  auto out = infer->output(0);
  INFO("Get output shape = %s", out->shape_string());
  for (int index = 0; index < out->channel(); ++index) {
    INFO("%f", out->at<float>(0, index));
  }
}

int main(int argc, char** argv) { lesson1(); }