// CPU emulator and matched sparse traversal benchmark; no synthesis claims.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "ap_fixed.h"
#include "conifer.h" // unmodified conifer 1.11 Python-package C++ header
using json=nlohmann::json;
json load(const std::string& p){std::ifstream f(p);if(!f)throw std::runtime_error("Cannot read "+p);json j;f>>j;return j;}

template<class Input,class Weight,class Score> struct Sparse {
 struct Affine {std::vector<std::pair<int,Weight>> terms; Score leaf_bias=0; Input threshold=0;};
 struct Tree {std::vector<Affine> splits,leaves;};
 std::vector<Tree> trees; int features,internal; bool linear;
 explicit Sparse(const json& spec):features(spec.at("feature_count")),internal(spec.at("internal_count")),linear(spec.at("leaf_mode")=="linear"){
  for(const auto& t:spec.at("trees")){
   Tree tree;
   for(int n=0;n<internal;++n){Affine a;a.threshold=-t.at("split_bias").at(n).get<double>();
    for(int f=0;f<features;++f){double v=t.at("split_weights").at(n).at(f);if(v!=0)a.terms.emplace_back(f,Weight(v));}tree.splits.push_back(a);}
   for(const auto& p:t.at("leaf_parameters")){Affine a;a.leaf_bias=p.at(linear?features:0).get<double>();
    if(linear)for(int f=0;f<features;++f){double v=p.at(f);if(v!=0)a.terms.emplace_back(f,Weight(v));}tree.leaves.push_back(a);}
   trees.push_back(tree);
  }
 }
 Score predict(const Input* x,int* routes=nullptr)const{
  Score total=0;
  for(size_t t=0;t<trees.size();++t){const auto& tree=trees[t];int n=0;
   while(n<internal){const auto& a=tree.splits[n];Input sum=0;for(const auto& v:a.terms)sum+=x[v.first]*v.second;
    n=2*n+1+int(sum>a.threshold);}
   if(routes)routes[t]=n-internal;
   const auto& leaf=tree.leaves[n-internal];Score value=0;
   for(const auto& v:leaf.terms)value+=x[v.first]*v.second;
   value+=leaf.leaf_bias; total-=value;
  }
  return total;
 }
};

template<class Input,class Weight,class Score>struct Config:conifer::ConiferConfiguration{
 using threshold_t=Input;using input_t=Input;using weight_t=Weight;using score_t=Score;static constexpr bool useAddTree=true;
};

template<class Input,class Weight,class Score>
int evaluate(const std::string& portable,const std::string& stock,const std::string& input,const std::string& output,bool benchmark){
 auto spec=load(portable);Sparse<Input,Weight,Score> model(spec);
 std::ifstream f(input,std::ios::binary);char magic[8];uint64_t events;uint32_t features;
 f.read(magic,8);f.read(reinterpret_cast<char*>(&events),8);f.read(reinterpret_cast<char*>(&features),4);
 if(std::string(magic,8)!=std::string("TBXGB01\0",8)||features!=unsigned(model.features))throw std::runtime_error("Bad inputs");
 std::vector<double> raw(events*features);f.read(reinterpret_cast<char*>(raw.data()),raw.size()*8);
 if(!f||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Bad input length");
 std::vector<Input> x(raw.begin(),raw.end());std::vector<double> scores(events);std::vector<int> routes(events*model.trees.size());
 for(size_t i=0;i<events;++i)scores[i]=double(model.predict(&x[i*features],&routes[i*model.trees.size()]));
 std::ofstream out(output,std::ios::binary);out.write(reinterpret_cast<const char*>(scores.data()),scores.size()*8);out.close();
 std::ofstream route(output+".routes",std::ios::binary);route.write(reinterpret_cast<const char*>(routes.data()),routes.size()*sizeof(int));route.close();
 json report;report["events"]=events;report["timing_note"]="Single-thread in-memory sparse path traversal, input conversion and I/O excluded; not FPGA latency";
 if(benchmark){std::vector<double> elapsed;volatile double checksum=0;
  for(int repeat=-1;repeat<9;++repeat){double sum=0;auto begin=std::chrono::steady_clock::now();
   for(size_t i=0;i<events;++i)sum+=double(model.predict(&x[i*features]));
   auto end=std::chrono::steady_clock::now();checksum=sum;if(repeat>=0)elapsed.push_back(std::chrono::duration<double>(end-begin).count());}
  std::sort(elapsed.begin(),elapsed.end());report["median_microseconds_per_event"]=elapsed[4]*1e6/events;report["repeats"]=9;report["checksum"]=double(checksum);
 }
 if(stock!="-"){
  conifer::BDT<Config<Input,Weight,Score>> reference(stock);std::vector<Input> row(features);size_t differences=0;double max_error=0;
  std::ofstream stock_out(output+".stock",std::ios::binary);
  for(size_t i=0;i<events;++i){std::copy_n(&x[i*features],features,row.begin());double value=double(reference.decision_function(row).at(0));
   max_error=std::max(max_error,std::abs(value-scores[i]));differences+=(value!=scores[i]);stock_out.write(reinterpret_cast<const char*>(&value),8);}
  report["stock_conifer_max_error"]=max_error;report["stock_conifer_exact_mismatches"]=differences;
 }
 std::cout<<report.dump()<<std::endl;return 0;
}
int main(int argc,char** argv){try{
 if(argc!=6)throw std::runtime_error("Usage: emulator profile normalized-portable stock-or-dash inputs.bin output.bin");
 std::string profile=argv[1];
 if(profile=="double")return evaluate<double,double,double>(argv[2],argv[3],argv[4],argv[5],true);
 #define FIXED(F) if(profile=="f" #F)return evaluate<ap_fixed<10+F,10,AP_RND_CONV,AP_SAT>,ap_fixed<6+F,6,AP_RND_CONV,AP_SAT>,ap_fixed<16+F,16,AP_RND_CONV,AP_SAT>>(argv[2],argv[3],argv[4],argv[5],false)
 FIXED(18);FIXED(12);FIXED(8);
 throw std::runtime_error("Unknown profile");
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
