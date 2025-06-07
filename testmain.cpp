#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "face_mosaic.h"

using namespace cv;
using namespace cv::face;
using namespace std;

// ����� �� ��ǥ �ҷ�����
vector<Rect> loadUserFaceRects(const string& filename) {
    vector<Rect> rects;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        istringstream ss(line);
        string token;
        vector<int> values;

        while (getline(ss, token, ',')) {
            values.push_back(stoi(token));
        }

        if (values.size() == 4) {
            rects.emplace_back(values[0], values[1], values[2], values[3]);
        }
    }

    return rects;
}

// IoU ������� ��ǥ ��
bool isUserFaceArea(const Rect& face, const vector<Rect>& userRects) {
    for (const Rect& user : userRects) {
        Rect intersect = face & user;
        float iou = (float)intersect.area() / (face.area() + user.area() - intersect.area());
        if (iou > 0.3f) return true;

        cout << "�� �� " << face <<  user << ", IoU=" << iou << endl;
    }

    return false;
}

// �� ������ũ Ŭ���� ����
class FaceMosaic {
public:
    FaceMosaic(const string& cascadeFile) {
        if (!faceCascade.load(cascadeFile)) {
            cerr << "Error loading cascade classifier!" << endl;
            exit(-1);
        }
    }

    Mat processImage(const string& imageFile, const vector<Rect>& userRects) {
        Mat img = imread(imageFile);
        if (img.empty()) {
            cerr << "�̹��� �ε� ����!" << endl;
            exit(-1);
        }

        vector<Rect> faces;
        faceCascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(30, 30));
      

        for (const Rect& face : faces) {
            cout << "Ž���� ��: " << face << endl;
            
            if (!isUserFaceArea(face, userRects)) {
                applyMosaic(img, face, 0.1);  // scale������ ���� ���� ���� // �ܺ� face_mosaic �Լ� ȣ��
            }
        }

        return img;
    }

    void saveImage(const Mat& img, const string& outputFile) {
        imwrite(outputFile, img);
    }

private:
    CascadeClassifier faceCascade;
};

int main() {
    string cascadeFile = "C:/Users/LJM/Desktop/FocuSSU_Project/haarcascade_frontalface_default.xml";
    string imageFile = "C:/Users/LJM/Desktop/FocuSSU_Project/compare_face.jpg";
    string outputFile = "C:/Users/LJM/Desktop/FocuSSU_Project/result.jpg";
    string userFaceBoxFile = "C:/Users/LJM/Desktop/FocuSSU_Project/user_faces.txt";

    FaceMosaic faceMosaic(cascadeFile);
    vector<Rect> userFaceRects = loadUserFaceRects(userFaceBoxFile);

    Mat processedImage = faceMosaic.processImage(imageFile, userFaceRects);

    // ��� �̹��� ���
    int targetWidth = 800;
    float scale = static_cast<float>(targetWidth) / processedImage.cols;
    Size newSize(targetWidth, static_cast<int>(processedImage.rows * scale));
    Mat resizedImage;
    resize(processedImage, resizedImage, newSize);

    namedWindow("Mosaic Result", WINDOW_AUTOSIZE);
    imshow("Mosaic Result", resizedImage);
    waitKey(0);

    return 0;
}
