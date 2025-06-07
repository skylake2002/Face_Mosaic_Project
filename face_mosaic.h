#pragma once
#include <opencv2/opencv.hpp>

// 타원형 모자이크 함수 (강도 scale은 기본 0.1)
void applyMosaic(cv::Mat& image, const cv::Rect& face, double scale = 0.1);
