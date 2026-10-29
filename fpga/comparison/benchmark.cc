// One binary64 sparse traversal kernel for the ten-model comparison.
// Axis comparisons elide fixed +/-1 factors; parsing, input conversion and I/O
// are outside the timed region. XGBoost retains binary32 input semantics.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "nlohmann/json.hpp"
using json=nlohmann::json;
struct Affine {
 std::vector<std::pair<int,double>> products;
 std::vector<std::pair<int,int>> units;
 double bias=0;
 double value(const double* x)const {
  // Retain feature order to match the canonical reference's floating arithmetic.
  size_t p=0,u=0;double result=0;
  while(p<products.size()||u<units.size()){
   if(u<units.size()&&(p==products.size()||units[u].first<products[p].first)){
    result+=units[u].second>0?x[units[u].first]:-x[units[u].first];++u;
   }else{result+=products[p].second*x[products[p].first];++p;}
  }
  return result+bias;
 }
 void add(int feature,double weight){
  if(weight==1||weight==-1)units.emplace_back(feature,weight>0?1:-1);
  else if(weight!=0)products.emplace_back(feature,weight);
 }
};
struct Node {Affine affine;int left=-1,right=-1,axis=-1,sign=1;double threshold=0;bool leaf=false,strict=false;};
struct Model {
 std::vector<std::vector<Node>> trees;bool xgb=false;double init=0;int features;
 explicit Model(const json& s){
  xgb=s.contains("n_trees");features=xgb?s.at("n_features").get<int>():s.at("feature_count").get<int>();
  if(xgb){
   init=s.at("init_predict").at(0);if(s.at("norm").get<double>()!=1)throw std::runtime_error("Unexpected normalization");
   for(const auto& group:s.at("trees")){const auto& t=group.at(0);std::vector<Node> tree;
    for(size_t n=0;n<t.at("feature").size();++n){Node node;int feature=t.at("feature").at(n);
     node.leaf=feature<0;node.strict=true;
     if(node.leaf)node.affine.bias=t.at("value").at(n).get<double>();
     else{node.axis=feature;node.threshold=t.at("threshold").at(n);node.left=t.at("children_left").at(n);node.right=t.at("children_right").at(n);}
     tree.push_back(node);
    }trees.push_back(tree);
   }
  }else{
   int internal=s.at("internal_count"),leaves=s.at("leaf_count");bool linear=s.at("leaf_mode")=="linear";
   for(const auto& t:s.at("trees")){std::vector<Node> tree;
    for(int n=0;n<internal;++n){Node node;int nonzero=0,feature=-1;double weight=0;
     for(int f=0;f<features;++f){double w=t.at("split_weights").at(n).at(f);node.affine.add(f,w);if(w!=0){++nonzero;feature=f;weight=w;}}
     node.threshold=-t.at("split_bias").at(n).get<double>();
     if(nonzero==1&&(weight==1||weight==-1)){node.axis=feature;node.sign=weight>0?1:-1;}
     node.left=2*n+1;node.right=2*n+2;tree.push_back(node);
    }
    for(int n=0;n<leaves;++n){Node node;node.leaf=true;const auto& p=t.at("leaf_parameters").at(n);
     if(linear)for(int f=0;f<features;++f)node.affine.add(f,p.at(f).get<double>());
     node.affine.bias=p.at(linear?features:0).get<double>();tree.push_back(node);
    }trees.push_back(tree);
   }
  }
 }
 double predict(const double* x,int* routes=nullptr)const{
  double total=init;
  for(size_t t=0;t<trees.size();++t){const auto& tree=trees[t];int n=0;
   while(!tree[n].leaf){const auto& node=tree[n];double v=node.axis>=0?(node.sign>0?x[node.axis]:-x[node.axis]):node.affine.value(x);
    bool left=node.strict?v<node.threshold:v<=node.threshold;n=left?node.left:node.right;}
   if(routes)routes[t]=n;
   double value=tree[n].affine.value(x);total+=xgb?value:-value;
  }return total;
 }
};
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("Usage: benchmark model.json input.bin output.bin");
 std::ifstream model_file(argv[1]);json spec;model_file>>spec;Model model(spec);
 std::ifstream input(argv[2],std::ios::binary);char magic[8];uint64_t events;uint32_t features;
 input.read(magic,8);input.read(reinterpret_cast<char*>(&events),8);input.read(reinterpret_cast<char*>(&features),4);
 if(std::string(magic,8)!=std::string("TBXGB01\0",8)||features!=unsigned(model.features))throw std::runtime_error("Bad input header");
 std::vector<double> x(events*features);input.read(reinterpret_cast<char*>(x.data()),x.size()*8);
 if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Bad input length");
 if(model.xgb)for(auto& v:x)v=static_cast<float>(v);
 std::vector<double> scores(events);std::vector<int> routes(events*model.trees.size());
 for(size_t i=0;i<events;++i)scores[i]=model.predict(&x[i*features],&routes[i*model.trees.size()]);
 std::ofstream output(argv[3],std::ios::binary);output.write(reinterpret_cast<const char*>(scores.data()),scores.size()*8);output.close();
 std::ofstream trace(std::string(argv[3])+".routes",std::ios::binary);trace.write(reinterpret_cast<const char*>(routes.data()),routes.size()*sizeof(int));trace.close();
 std::vector<double> elapsed;volatile double checksum=0;
 for(int r=-1;r<9;++r){double sum=0;auto start=std::chrono::steady_clock::now();
  for(size_t i=0;i<events;++i)sum+=model.predict(&x[i*features]);
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();checksum=sum;if(r>=0)elapsed.push_back(seconds);
 }
 std::sort(elapsed.begin(),elapsed.end());json report={{"events",events},{"median_microseconds_per_event",elapsed[4]*1e6/events},{"repeats",9},{"checksum",double(checksum)}};
 std::cout<<report.dump()<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
