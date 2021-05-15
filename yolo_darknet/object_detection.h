#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector> // for vector
#include <algorithm> // for copy() and assign()
#include <iterator> // for back_inserter
#include <iomanip>
#include <string>
#include <fstream>
#include <thread>
#define OPENCV
#include "yolo_v2_class.hpp"

class object_detection : public Detector
{
public:
    object_detection();
    object_detection(std::string  namesFile, std::string  cfgFile, std::string  weightsFile);

    void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,int thicknessRec = 3,double fontScaleText = 1,int thicknessText = 1);//, unsigned int wait_msec = 0
    void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names);
    std::vector<std::string> objects_names_from_file(std::string const filename);
    std::vector<bbox_t> detection(cv::Mat mat_img, cv::Mat &out_img, float thresh = 0.2f, bool use_mean = false);
     std::vector<bbox_t> detection(cv::Mat mat_img, float thresh = 0.2f, bool use_mean = false);

    void setNamesFile(std::string filename){ names_file = filename; }
    std::string getNamesFile(){ return names_file;}

    void setCfgFile(std::string filename){ cfg_file = filename; }
    std::string getCfgFile(){ return cfg_file;}

    void setWeightsFile(std::string filename){ weights_file = filename; }
    std::string getWeightsFile(){ return weights_file;}

    void setObjNames(std::vector<std::string> objNames){
        obj_names.reserve(objNames.size());
        std::copy(objNames.begin(),objNames.end(),std::back_inserter(obj_names));
    }
    std::vector<std::string> getObjNames(){ return obj_names;}

    int MergeOverlappingRectangles(bbox_t input1, bbox_t input2);
    std::vector<bbox_t> getOutputM(){ return outputM;}

private:
    std::string  names_file;
    std::string  cfg_file;
    std::string  weights_file;
    std::vector<std::string> obj_names;
    std::vector<bbox_t> outputM;
};

#endif // OBJECT_DETECTION_H
