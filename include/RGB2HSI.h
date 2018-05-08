#ifndef _RGB2HSI_H
#define _RGB2HSI_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <string>
class RGB2HSI{
    public:
        void rgb2HSI(cv::Mat rbgImage,cv::Mat hsiImage,cv::Mat Hvalue,cv::Mat Svalue,cv::Mat Ivalue,int num);
        cv::Mat getHsiImage(){return hsiImage;};
        cv::Mat setHsiImage(cv::Mat hsiImage){this->hsiImage = hsiImage;};
        cv::Mat getHvalue(){return Hvalue;};
        cv::Mat setHvalue(cv::Mat Hvalue){this->Hvalue = Hvalue;};
        cv::Mat getSvalue(){return Svalue;};
        cv::Mat setSvalue(cv::Mat Svalue){this->Svalue = Svalue;};
        cv::Mat getIvalue(){return Ivalue;};
        cv::Mat setIvalue(cv::Mat Ivalue){this->Ivalue = Ivalue;};
        std::vector<cv::Mat> getHistmat(){return outputHist;}
        void drawHist(cv::Mat sourcePic,std::string i,int num);
    private:
    cv::Mat Hvalue;
    cv::Mat Svalue;
    cv::Mat Ivalue;
    cv::Mat hsiImage;
    std::vector<cv::Mat> outputHist;
    
};
#endif 