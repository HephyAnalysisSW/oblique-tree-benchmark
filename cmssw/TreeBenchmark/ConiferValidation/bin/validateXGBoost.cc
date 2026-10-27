// Exercise the actual CMSSW conifer external, with balanced score aggregation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "conifer.h"
#include "ap_fixed.h"

struct Arguments { std::string model, input, output, trace; };

nlohmann::json validate_contract(const std::string& filename) {
  std::ifstream input(filename); if (!input) throw std::runtime_error("Cannot open model");
  nlohmann::json spec; input >> spec;
  if (spec.at("n_classes").get<int>() != 2 || spec.at("n_features").get<int>() != 16 ||
      spec.at("splitting_convention").get<std::string>() != "<")
    throw std::runtime_error("Expected binary 16-feature XGBoost export with strict-left cuts");
  for (const auto& group: spec.at("trees")) {
    const auto& tree=group.at(0);
    for (size_t node=0;node<tree.at("feature").size();++node) {
      int feature=tree.at("feature").at(node).get<int>();
      if (feature != -2 && (feature < 0 || feature >= 16)) throw std::runtime_error("Invalid feature index");
      const auto& weights=tree.at("weight").at(node);
      if(weights.size()!=16) throw std::runtime_error("Invalid weight vector");
      for(int j=0;j<16;++j) if(weights.at(j).get<double>() != double(j==feature))
        throw std::runtime_error("CMSSW conifer 1.7 cannot emulate oblique weights: use a compatible modern backend");
    }
  }
  return spec;
}

template<class Input, class Score>
int evaluate(const Arguments& args) {
  auto contract=validate_contract(args.model);
  conifer::BDT<Input, Score, true> model(args.model);
  std::ifstream input(args.input, std::ios::binary);
  std::ofstream output(args.output, std::ios::binary);
  if (!input || !output) throw std::runtime_error("Cannot open input/output");
  char magic[8]; uint64_t events; uint32_t features;
  input.read(magic,8); input.read(reinterpret_cast<char*>(&events),8);
  input.read(reinterpret_cast<char*>(&features),4);
  if (std::string(magic,8) != std::string("TBXGB01\0",8) || features != 16)
    throw std::runtime_error("Bad input header");
  std::vector<double> row(features);
  std::vector<Input> x(features);
  struct TraceTree {
    std::vector<int> feature, left, right;
    std::vector<Input> threshold;
  };
  std::vector<TraceTree> trace_trees;
  std::ofstream trace;
  if (!args.trace.empty()) {
    const auto& spec=contract;
    for (const auto& group: spec.at("trees")) {
      const auto& tree=group.at(0); TraceTree t;
      t.feature=tree.at("feature").get<std::vector<int>>();
      t.left=tree.at("children_left").get<std::vector<int>>();
      t.right=tree.at("children_right").get<std::vector<int>>();
      for (const auto& v:tree.at("threshold")) t.threshold.push_back(v.get<double>());
      trace_trees.push_back(std::move(t));
    }
    trace.open(args.trace, std::ios::binary);
    if (!trace) throw std::runtime_error("Cannot open leaf trace");
  }
  auto begin = std::chrono::steady_clock::now();
  for (uint64_t i=0;i<events;++i) {
    input.read(reinterpret_cast<char*>(row.data()), features*sizeof(double));
    if (!input) throw std::runtime_error("Truncated input");
    for (uint32_t f=0;f<features;++f) x[f]=row[f];
    auto scores=model.decision_function(x);
    double score=static_cast<double>(scores.at(0));
    output.write(reinterpret_cast<const char*>(&score),sizeof(score));
    if (!args.trace.empty()) {
      // Independent exported-tree walker, to check conversion routing against
      // all native XGBoost leaf indices. This is not the score emulator.
      for (const auto& tree:trace_trees) {
        unsigned int node=0;
        while (tree.feature[node] != -2) {
          auto feature=tree.feature[node];
          node=x[feature]<tree.threshold[node] ? tree.left[node] : tree.right[node];
        }
        if (node>65535) throw std::runtime_error("Trace node exceeds uint16");
        uint16_t leaf=node; trace.write(reinterpret_cast<const char*>(&leaf),2);
      }
    }
  }
  if (input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Trailing input data");
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  std::cerr << "events=" << events << " wall_seconds=" << seconds
            << " (includes binary I/O and optional independent trace)\n";
  return 0;
}

int main(int argc,char** argv) {
  try {
    if(argc!=5 && argc!=6) throw std::runtime_error("Usage: profile model.json input.bin scores.bin [leaves.bin]");
    std::string name=argv[1]; Arguments args{argv[2],argv[3],argv[4],argc==6?argv[5]:""};
    // Populated from the tracked scan configuration by the validation driver.
    #include "PrecisionProfiles.inc"
    throw std::runtime_error("Unknown profile: "+name);
  } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
