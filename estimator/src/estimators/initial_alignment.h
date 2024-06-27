#pragma once
#include "commons.h"
#include "integration.h"

class ImageFrame
{
public:
    ImageFrame() = default;
    ImageFrame(const Feats &_feats, double _t) : timestamp(_t), is_keyframe(false) { feats = _feats; }

    Feats feats;
    double timestamp;
    Mat3d r;
    Vec3d t;
    std::shared_ptr<Integration> integration;
    bool is_keyframe;
};