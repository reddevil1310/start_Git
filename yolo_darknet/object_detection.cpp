#include "object_detection.h"
//#include<QtCore>
//#include<QString>

//QString qt_exe_path = QCoreApplication::applicationDirPath();
//std::string exe_path = qt_exe_path.toUtf8().data();

object_detection::object_detection(): Detector("learning/yolov3.cfg", "learning/yolov3.weights")
{
    names_file = "learning/coco.names";
    cfg_file = "learning/yolov3.cfg";
    weights_file = "learning/yolov3.weights";
    obj_names = objects_names_from_file("learning/coco.names");
}

object_detection::object_detection(std::string  namesFile, std::string  cfgFile, std::string  weightsFile) : Detector(cfgFile, weightsFile)
{
    names_file = namesFile;
    cfg_file = cfgFile;
    weights_file = weightsFile;
    obj_names = objects_names_from_file(namesFile);
}

void object_detection::draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,int thicknessRec,double fontScaleText,int thicknessText)//, unsigned int wait_msec
{
    for (auto &i : result_vec) {
        cv::Scalar color(60, 160, 260);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, thicknessRec);
//        if(obj_names.size() > i.obj_id)
//            putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
//        if(i.track_id > 0)
//            putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x+5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        if (obj_names.size() > i.obj_id)
                    putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, fontScaleText, color, thicknessText);
        if (i.track_id > 0)
                    putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x + 5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, fontScaleText, color, thicknessText);
    }
    //cv::imshow("window name", mat_img);
    //cv::waitKey(wait_msec);
}

void object_detection::show_result(const std::vector<bbox_t> result_vec, const std::vector<std::string> obj_names)
{
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y << ", w = " << i.w << ", h = " << i.h << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> object_detection::objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; std::getline(file, line);) file_lines.push_back(line);
//    std::string line;
//    while (std::getline(file, line))
//    {
//        file_lines.push_back(line);
//    }
    std::cout << "object names loaded \n";
    return file_lines;
}

int object_detection::MergeOverlappingRectangles(bbox_t input1, bbox_t input2)
{
    cv::Point2d A, B, C, D, O, A1, B1, C1, D1, O1, OO1;
    A.x = input1.x; A.y = input1.y;
    B.x = input1.x + input1.w; B.y = input1.y;
    C.x = input1.x + input1.w; C.y = input1.y + input1.h;
    D.x = input1.x; D.y = input1.y + input1.h;
    O.x = 0.5*(A.x + C.x); O.y = 0.5*(A.y + C.y);

    A1.x = input2.x; A1.y = input2.y;
    B1.x = input2.x + input2.w; B1.y = input2.y;
    C1.x = input2.x + input2.w; C1.y = input2.y + input2.h;
    D1.x = input2.x; D1.y = input2.y + input2.h;
    O1.x = 0.5*(A1.x + C1.x); O1.y = 0.5*(A1.y + C1.y);
    OO1.x = O.x - O1.x; OO1.y = O.y - O1.y;

    double distance_OO1_x = abs(OO1.x);
    double distance_OO1_y = abs(OO1.y);
    double xmax = C.x, xmin = A.x, ymax = C.y, ymin = A.y;
    if (A.x > A1.x) xmin = A1.x;
    if (C.x < C1.x) xmax = C1.x;
    if (A.y > A1.y) ymin = A1.y;
    if (C.y < C1.y) ymax = C1.y;
    bbox_t out = input1;
    outputM.clear();
    if (input1.obj_id != input2.obj_id || distance_OO1_x >= 0.5*(input1.w + input2.w) || distance_OO1_y >= 0.5*(input1.h + input2.h))
    {
        outputM.push_back(input1);
        outputM.push_back(input2);
        return 0;
    }
    else
    {
        out.x = xmin; out.y = ymin;
        out.w = xmax - xmin; out.h = ymax - ymin;
        out.frames_counter = input1.frames_counter + input2.frames_counter;
        out.prob = 0.5*(input1.prob + input2.prob);
        outputM.push_back(out);
        return 1;
    }

}


std::vector<bbox_t> object_detection::detection(cv::Mat mat_img, cv::Mat &out_img, float thresh, bool use_mean)
{
    std::vector<bbox_t> result_vec = detect(mat_img, thresh, use_mean);
    out_img = mat_img.clone();
    if(out_img.channels() == 1) cv::cvtColor(out_img, out_img, cv::COLOR_GRAY2BGR);
    draw_boxes(out_img, result_vec, obj_names);
    //show_result(result_vec, obj_names);
    return result_vec;
}
std::vector<bbox_t> object_detection::detection(cv::Mat mat_img, float thresh, bool use_mean)
{
    std::vector<bbox_t> result_vec = detect(mat_img, thresh, use_mean);
    return result_vec;
}
