#include "builder/trt_builder.hpp"
#include "common/ilogger.hpp"
#include "infer/trt_infer.hpp"

void lesson3() {
  // shape : 5x5
  TRT::set_layer_hook_reshape(
      [](const std::string& name,
         const std::vector<int64_t>& shape) -> std::vector<int64_t> {
        INFO("name:%s, shape:%s", name.c_str(),
             iLogger::join_dims(shape).c_str());
        return {-1, 25};
      });

  // model , max batch size, onnx file, trt mode
  TRT::compile(TRT::Mode::FP32, 1, "../lesson2.onnx", "lesson2.fp32.trtmodel",
               {{1, 1, 5, 5}});

  // get infer
  auto infer = TRT::load_infer("lesson2.fp32.trtmodel");
  // // set value
  infer->input(0)->set_to(1.0f);
  infer->forward();

  // get output
  auto out = infer->output(0);
  INFO("Get output shape = %s", out->shape_string());
}

int main(int argc, char** argv) { lesson3(); }