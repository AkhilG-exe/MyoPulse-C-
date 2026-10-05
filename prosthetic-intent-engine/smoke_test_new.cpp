#include "intent_engine/intent_engine.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    using namespace intent_engine;
    EngineConfig cfg;
    cfg.numChannels = 4;
    cfg.sampleRateHz = 1000.0;
    IntentEngine engine(cfg);
    std::vector<double> samples(4, 0.0);
    for (int i = 0; i < 100; ++i) {
        for (int ch = 0; ch < 4; ++ch) {
            samples[ch] = std::sin(2.0 * 3.14159 * 10.0 * i / 1000.0 + ch);
        }
        engine.processSample(samples, 0.001);
    }
    std::cout << "SMOKE_OK" << std::endl;
    return 0;
}
