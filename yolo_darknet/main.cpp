#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <object_detection.h>
#include <QTime>

int main(int argc, char *argv[])
{
    std::string name_file = "/mnt/Ubuntu/Model/labels.txt";
    std::string cfg_file = "/mnt/Ubuntu/Model/yolov3-tiny.cfg";
    std::string weights_file = "/mnt/Ubuntu/Model/yolov3-tiny.weights";
    object_detection* obj = new object_detection(name_file,cfg_file,weights_file);
    cv::Mat input,output;
    float thresh = 0.5;
    cv::VideoCapture cap("/opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_720p.mp4");
    input = cv::imread("/home/huy/Pictures/people.png");
    QTime t0 = QTime::currentTime();
    obj->detection(input,output,thresh);
    cv::imshow("output",output);
    QTime t1 = QTime::currentTime();
    std::cout<<"Time: "<<(t1.hour() - t0.hour())*1000*60*60 + (t1.minute()-t0.minute())*60*1000 + (t1.second()-t0.second())*1000 + t1.msec() - t0.msec()<<std::endl;
    while(true)
    {
        t0 = QTime::currentTime();
        cap.read(input);
        //cv::resize(input,input,cv::Size(1000,1000));
        obj->detection(input,output,thresh);
        cv::imshow("output",output);
         t1 = QTime::currentTime();
          std::cout<<"Time: "<<(t1.hour() - t0.hour())*1000*60*60 + (t1.minute()-t0.minute())*60*1000 + (t1.second()-t0.second())*1000 + t1.msec() - t0.msec()<<std::endl;
        cv::waitKey(10);

    }
    cv::waitKey();


    return 0;
}
