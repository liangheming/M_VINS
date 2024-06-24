#pragma once
#include "commons.h"

class SWConfig
{
};

class SlideWindowEstimator
{
public:
    SlideWindowEstimator(SWConfig &config);

    void processImu(const double &dt, const Vec3d &acc, const Vec3d &gyro);

    void processFeature(const Feats &feats, double timestamp);

private:
    SWConfig m_config;
};