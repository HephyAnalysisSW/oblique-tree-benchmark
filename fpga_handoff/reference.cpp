// Floating-point reference only; FPGA scheduling/precision are intentionally explicit future choices.
#include "model_coefficients.h"
#include <array>
#include <iomanip>
#include <iostream>
using namespace oblique_model;
int main() {
    std::array<double,N_FEATURES> x{};
    std::cout << std::setprecision(17);
    while (std::cin >> x[0]) {
        for(int feature=1;feature<N_FEATURES;++feature) if(!(std::cin>>x[feature])) return 2;
        double score=0;
        std::array<int,N_TREES> routes{};
        for(int tree=0;tree<N_TREES;++tree) {
            int node=0;
            for(int level=0;level<DEPTH;++level) {
                double margin=0;
                for(int feature=0;feature<N_FEATURES;++feature) margin+=SPLIT_W[tree][node][feature]*x[feature];
                margin+=SPLIT_B[tree][node];
                node=2*node+1+(margin>0?1:0);
            }
            const int leaf=node-N_INTERNAL;
            routes[tree]=leaf;
            double prediction=0;
            if constexpr(LEAF_WIDTH>1)
                for(int feature=0;feature<N_FEATURES;++feature) prediction+=LEAF[tree][leaf][feature]*x[feature];
            prediction+=LEAF[tree][leaf][LEAF_WIDTH-1];
            score+=prediction;
        }
        std::cout << -score;
        for(int leaf:routes) std::cout << ' ' << leaf;
        std::cout << '\n';
    }
    return 0;
}
