// pch.cpp: файл исходного кода, соответствующий предварительно скомпилированному заголовочному файлу

#include "Dll1.h"
#include <windows.h>
#include <k4a/k4a.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <deque>
#include <string>
#include <vector>
#include <iostream>
#include <windows.h>
#include <mutex>
#include <condition_variable>
#include <opencv2/flann/matrix.h>
#include <ctime>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
using namespace std;
using namespace cv;
Point transform_cord(vector<Point> camera_cord, Point target_cord, Point touch) {
    // переводим координаты точки из системы камеры в систему экрана
    // scr - разрешение камеры
    int scr_x = abs(camera_cord[0].x - camera_cord[1].x), scr_y = abs(camera_cord[0].y - camera_cord[1].y);
    // новые координаты считаем по формуле: старые_координаты / разрешение_камеры * разрешение_целевого_экрана
    // по у зеркалим т.к. система координат камеры перевернута
    Point new_touch = Point((float)touch.x / scr_x * target_cord.x, target_cord.y - (float)touch.y / scr_y * target_cord.y);
    return new_touch;
}



vector<Point> load_vector(String filename) {
    // загружаем вектор настроек из файла
    // используем для этого функции opencv
    // если интересно: https://docs.opencv.org/4.x/da/d56/classcv_1_1FileStorage.html
    // 180с - учебник
    FileStorage fs(filename, FileStorage::READ);
    vector<Point> ans;
    fs["points"] >> ans;
    return ans;
}

vector<int> load_vector_int(String filename) {
    // загружаем вектор настроек из файла
    // используем для этого функции opencv
    // если интересно: https://docs.opencv.org/4.x/da/d56/classcv_1_1FileStorage.html
    // 180с - учебник
    FileStorage fs(filename, FileStorage::READ);
    vector<int> ans;
    fs["points"] >> ans;
    return ans;
}

void save_vector(const vector<int>& points, String filename) {
    // сохраняем вектор
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "points" << points;
    fs.release();
}