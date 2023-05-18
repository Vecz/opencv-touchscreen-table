// dllmain.cpp : Определяет точку входа для приложения DLL.
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
k4a_image_t upscaleIR16Image(const k4a_image_t& ir16Image)
{
    // Get the dimensions of the input IR16 image
    int width = k4a_image_get_width_pixels(ir16Image);
    int height = k4a_image_get_height_pixels(ir16Image);

    // Create a new image with the target resolution (720p) and custom16 format
    k4a_image_t upscaledImage = nullptr;
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM16, 1280, 720, 1280 * static_cast<int>(sizeof(uint16_t)), &upscaledImage);

    // Get the buffer containing the IR16 image data
    uint16_t* ir16Data = reinterpret_cast<uint16_t*>(k4a_image_get_buffer(ir16Image));
    uint16_t* upscaledData = reinterpret_cast<uint16_t*>(k4a_image_get_buffer(upscaledImage));

    // Copy and upscale the IR16 image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint16_t ir16Value = ir16Data[y * width + x];
            upscaledData[y * 1280 + x] = ir16Value;
        }
    }

    return upscaledImage;
}

k4a_image_t convertIR16ToCustom16(const k4a_image_t& ir16Image)
{
    // Get the dimensions of the input IR16 image
    int width = k4a_image_get_width_pixels(ir16Image);
    int height = k4a_image_get_height_pixels(ir16Image);

    // Create a new image with the same resolution and custom16 format
    k4a_image_t custom16Image = nullptr;
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM16, width, height, width * static_cast<int>(sizeof(uint16_t)), &custom16Image);

    // Get the buffer containing the IR16 image data
    uint16_t* ir16Data = reinterpret_cast<uint16_t*>(k4a_image_get_buffer(ir16Image));
    uint16_t* custom16Data = reinterpret_cast<uint16_t*>(k4a_image_get_buffer(custom16Image));

    // Copy the data from IR16 image to custom16 image
    std::memcpy(custom16Data, ir16Data, width * height * sizeof(uint16_t));

    return custom16Image;
}


k4a_image_t IR_to_color(k4a_transformation_t& transformation, k4a_image_t& depth_image, k4a_image_t& IR_image_raw, k4a_image_t& color_image)
{
    // Get attributes of the color image
    uint32_t color_height = k4a_image_get_height_pixels(color_image);
    uint32_t color_width = k4a_image_get_width_pixels(color_image);
    uint32_t color_stride = k4a_image_get_stride_bytes(color_image);

    // Upscale the IR image to match the color image resolution
    k4a_image_t IR_image = convertIR16ToCustom16(IR_image_raw);

    // Create blank image containers for transformed images
    k4a_image_t transformed_IR_image = nullptr;
    k4a_image_t transformed_depth_image = nullptr;
    if (k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM16, color_width, color_height, color_stride, &transformed_IR_image) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to create blank IR image (IR_to_color)\n");
    }
    if (k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, color_width, color_height, color_stride, &transformed_depth_image) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to create blank depth image (IR_to_color)\n");
    }

    // Apply the transformation
    if (k4a_transformation_depth_image_to_color_camera_custom(transformation, depth_image, IR_image, transformed_depth_image,
        transformed_IR_image, K4A_TRANSFORMATION_INTERPOLATION_TYPE_NEAREST, 0) != K4A_RESULT_SUCCEEDED)
    {
        printf("IR/depth transformation failed\n");
    }
    //cout << "Вроде успех\n";
    // Release the intermediate IR image
    k4a_image_release(IR_image);
    //k4a_image_release(transformed_depth_image);
    return transformed_IR_image;
}


Mat computeOpticalFlow(Mat previous_frame, Mat current_frame) {
    // вычисляем параметры оптического потока
    // задаем параметры
    // подробнее: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    // учебник - 498с.
    double pyrScale = 0.5;
    int levels = 1;
    int winsize = 13;
    int iterations = 2;
    int polyN = 7;
    double polySigma = 1.5;
    int flags = 0;
    Mat flow;
    // Вычисляем оптический поток методом Фарнебека
    calcOpticalFlowFarneback(previous_frame, current_frame, flow, pyrScale, levels, winsize, iterations, polyN, polySigma, flags);
    // переводим картинку из серых тонов, в пространство ргб
    cvtColor(previous_frame, previous_frame, COLOR_GRAY2RGB);
    // делим поток на оси х и у
    Mat flow_xy[2], mag, ang;
    split(flow, flow_xy);
    Mat flow_magnitude, flow_angle;
    // переводим координаты в полярные
    cv::cartToPolar(flow_xy[0], flow_xy[1], flow_magnitude, flow_angle);

    // задаем параметры фильтра мусора
    double threshold = 5.0;
    cv::Mat flow_magnitude_thresholded;
    // https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    cv::threshold(flow_magnitude, flow_magnitude_thresholded, threshold, 1.0, cv::THRESH_BINARY);

    // переводим в формат CV_8UC1
    cv::Mat flow_magnitude_thresholded_8u;
    flow_magnitude_thresholded.convertTo(flow_magnitude_thresholded_8u, CV_8UC1, 255.0);

    // Преобразуем фильтры в карту цветов
    // https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html#gadf478a5e5ff49d8aa24e726ea6f65d15
    cv::Mat flow_magnitude_colormap;
    cv::applyColorMap(flow_magnitude_thresholded_8u, flow_magnitude_colormap, cv::COLORMAP_JET);
    //imshow("Example4", flow_magnitude_colormap);
    return flow_magnitude_colormap;
}



Point get_point_in_cam(Mat frame, vector<int>& hsv_set) {
    // получаем точку касания путем усреднения всех областей попадающих в диапазон фильтра
    Mat res, dist;
    Point ans = Point(0, 0);
    frame.copyTo(res);
    // очень важно подобрать цветовые пороги, чтобы выделять только точку касания
    // переводим из пространства цветов брг в hsv 
    inRange(res, Scalar(hsv_set[0], hsv_set[1], hsv_set[2]), Scalar(hsv_set[3], hsv_set[4], hsv_set[5]), dist);
    // отрисовываем полученное изображение
    //imshow("Example5", dist);
    // считаем моменты 
    // https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html
    // учебник - 366с.
    auto mom = moments(dist, 1);
    auto dm01 = mom.m01;
    auto dm10 = mom.m10;
    auto dArea = mom.m00;
    // размер искомой области больше 2-х пикселей
    if (dArea > 10) {
        ans.x = dm10 / dArea;
        ans.y = dm01 / dArea;
    }
    return ans;
}
void procesing_frame(vector<int> hsv_set, Mat previous_frame, Mat current_frame, Point& output, mutex& control) {
    //cout << "Enter\n";
    // вычисляем оптический поток, он в флоу сохранится
    Mat result = computeOpticalFlow(previous_frame, current_frame);
    // получаем координату точки касания
    auto ans = get_point_in_cam(result, hsv_set);
    unique_lock<mutex> ul(control);
    output = ans;
    //cout << "Procesing end\n";
    ul.unlock();
}

Point split_frame(vector<int>& hsv_set, Mat& previous_frame, Mat& current_frame, Point& previous, int numRows, int numCols) {
    int height = previous_frame.rows;
    int width = previous_frame.cols;
    int subImgHeight = (double)height / numRows;
    int subImgWidth = (double)width / numCols;
    ////cout << "h: " << height << " w:" << width << endl;
    ////cout << "sh: " << subImgHeight << " sw:" << subImgWidth << endl;
    vector<vector<Point>> ans(numRows, vector<Point>(numCols));
    mutex control;
    vector<thread> t;
    for (int i = 0; i < numRows; i++) {
        int y = i * subImgHeight;
        for (int j = 0; j < numCols; j++) {
            int x = j * subImgWidth;

            auto roi = Rect(x, y, subImgWidth, subImgHeight);
            // Extract the current sub-image
            ////cout << i << " " << j<<endl;
            ////cout << x << " " << y << " cnt: " << previous_frame.isContinuous() << " " << endl;
            ////cout << roi.width << " " << roi.height << endl;
            //cout << (roi.x + roi.width <= previous_frame.cols) << " " << (roi.y + roi.height <= previous_frame.rows) << endl;
            try {
                Mat subImage_prev = previous_frame(roi).clone();
                ////cout << "First\n";
                Mat subImage_current = current_frame(roi).clone();
                ////cout << subImage_prev.cols << " " << subImage_prev.rows << endl;
                ////cout << "endl\n";
                t.push_back(thread(procesing_frame, hsv_set, subImage_prev, subImage_current, ref(ans[i][j]), ref(control)));
            }
            catch (cv::Exception& e) {
                cerr << "Reason: " << e.msg << endl;
                // nothing more we can do
                exit(1);
            }

            // Create a thread to process the sub-image

            // Detach the thread so that it runs independently
        }
    }
    for (auto& i : t) {
        //unique_lock<mutex> ul(control);
        i.join();
        //ul.unlock();
    }
    //cout << "Threads exit" << endl;
    Point answer = ans[0][0];
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int x = j * subImgWidth;
            int y = i * subImgHeight;
            if (ans[i][j] != Point(0, 0)) {
                ans[i][j].x += x;
                ans[i][j].y += y;
                answer = ans[i][j];
            }
        }
    }
    auto temp2 = answer - previous;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            if (ans[i][j] == Point(0, 0))continue;
            auto temp1 = ans[i][j] - previous;
            if (temp1.x * temp1.x + temp1.y * temp1.y < temp2.x * temp2.x + temp2.y * temp2.y) {
                answer = ans[i][j];
                temp2 = answer - previous;
            }
        }
    }
    //cout << answer.x << " " << answer.y << endl;
    //cout << "OK" << endl;
    return answer;
}

// типа главный
void loading(int * array_c, int * array_p, int * cords, int *& ans) {
    //cout << "First" << endl;
    int numRows = 8, numCols = 2;
    // загружаем координаты угловых точек, которые определяем через 2 программу
    Point previous = Point(cords[0], cords[1]), next = Point(cords[2], cords[3]);
    Mat curr = cv::Mat::zeros(576, 640, CV_16U);
    Mat previous_frame = cv::Mat::zeros(576, 640, CV_16U);

    // Copy data from array_c to curr matrix
    for (int i = 0; i < curr.rows; i++) {
        for (int j = 0; j < curr.cols; j++) {
            curr.at<ushort>(i, j) = static_cast<ushort>(array_c[i * curr.cols + j]);
        }
    }

    // Copy data from array_p to previous_frame matrix
    for (int i = 0; i < previous_frame.rows; i++) {
        for (int j = 0; j < previous_frame.cols; j++) {
            previous_frame.at<ushort>(i, j) = static_cast<ushort>(array_p[i * previous_frame.cols + j]);
        }
    }

    // Set ROI coordinates
    int x1 = cords[0];
    int y1 = cords[1];
    int x2 = cords[2];
    int y2 = cords[3];

    // Check if ROI coordinates are valid
    if (x1 < 0 || y1 < 0 || x2 > curr.cols || y2 > curr.rows) {
        // Handle invalid coordinates (e.g., throw an error, return, or set default values)
        return;
    }

    // Create ROI
    Rect roi(x1, y1, x2 - x1, y2 - y1);

    // Crop the frames using ROI
    Mat current_frame = curr(roi).clone();
    previous_frame = previous_frame(roi).clone();
    //cout << "Forth" << endl;
    // параметры фильтров для hsv
    vector<int> hsv_set = { 0,0,90,255,255,220 };
    //auto temp = load_vector_int(filename_hsv);
    //if (temp.size() == 6) hsv_set = temp;
    //cout << "Fiveth" << endl;
    //cout << "5.5\n";
    /*
    if (!current_frame.isContinuous()) {
        //cout << current_frame.type() << " - type\n";
        current_frame = current_frame.reshape(1, current_frame.rows * current_frame.cols).clone();
    }
    if (!previous_frame.isContinuous()) {
        //cout << current_frame.type() << " - type\n";
        previous_frame = previous_frame.reshape(1, previous_frame.rows * previous_frame.cols).clone();
    }
    */
    //cout << "Six\n";
    next = split_frame(hsv_set, previous_frame, current_frame, previous, numRows, numCols);
    //split_frame(hsv_set, previous_frame, current_frame, previous, numRows, numCols);
    //cout << "Seven \n";
    if (next == Point(0, 0)) next = previous;
    ans = new int[2];
    ans[0] = next.x;
    ans[1] = next.y;
    //imwrite("megaframe.jpg", current_frame);
} 