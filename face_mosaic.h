#pragma once
#include <opencv2/opencv.hpp>

// Ÿ���� ������ũ �Լ� (���� scale�� �⺻ 0.1)
void applyMosaic(cv::Mat& image, const cv::Rect& face, double scale = 0.1);
