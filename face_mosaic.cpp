#include "face_mosaic.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void applyMosaic(cv::Mat& image, const cv::Rect& face, double scale) {
    // ��ȿ ���� üũ
    if (face.x < 0 || face.y < 0 || face.x + face.width > image.cols || face.y + face.height > image.rows)
        return;

    // �� ROI ����
    cv::Mat faceROI = image(face).clone();
    cv::Mat mosaicFace;

    // ������ũ ó��
    cv::resize(faceROI, mosaicFace, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::resize(mosaicFace, mosaicFace, faceROI.size(), 0, 0, cv::INTER_NEAREST);

    // Ÿ���� ������ũ ����
    cv::Mat mask = cv::Mat::zeros(faceROI.size(), CV_8UC1);
    cv::ellipse(mask,
        cv::Point(faceROI.cols / 2, faceROI.rows / 2),
        cv::Size(faceROI.cols / 2, faceROI.rows / 2),
        0, 0, 360, cv::Scalar(255), -1);

    // ���� �̹����� �ش� ������ Ÿ���� ����ŷ�� ������ũ ����
    mosaicFace.copyTo(image(face), mask);
}
