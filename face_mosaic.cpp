#include "face_mosaic.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void applyMosaic(cv::Mat& image, const cv::Rect& face, double scale) {
    // 유효 범위 체크
    if (face.x < 0 || face.y < 0 || face.x + face.width > image.cols || face.y + face.height > image.rows)
        return;

    // 얼굴 ROI 추출
    cv::Mat faceROI = image(face).clone();
    cv::Mat mosaicFace;

    // 모자이크 처리
    cv::resize(faceROI, mosaicFace, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::resize(mosaicFace, mosaicFace, faceROI.size(), 0, 0, cv::INTER_NEAREST);

    // 타원형 모자이크 생성
    cv::Mat mask = cv::Mat::zeros(faceROI.size(), CV_8UC1);
    cv::ellipse(mask,
        cv::Point(faceROI.cols / 2, faceROI.rows / 2),
        cv::Size(faceROI.cols / 2, faceROI.rows / 2),
        0, 0, 360, cv::Scalar(255), -1);

    // 원본 이미지의 해당 영역에 타원형 마스킹된 모자이크 적용
    mosaicFace.copyTo(image(face), mask);
}
