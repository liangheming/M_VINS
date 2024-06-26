#pragma once
#include "commons.h"

struct SFMFeature
{
    bool state;
    int id;
    std::vector<std::pair<int, Vec2d>> observation;
    double position[3];
    double depth;
};