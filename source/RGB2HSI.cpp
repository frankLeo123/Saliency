#include <opencv2/opencv.hpp>
#include "RGB2HSI.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void RGB2HSI::rgb2HSI(cv::Mat rbgImage,cv::Mat hsiImage,cv::Mat Hvalue,cv::Mat Svalue,cv::Mat Ivalue,int num){
        hsiImage = Mat(Size(rbgImage.cols, rbgImage.rows), CV_8UC3);
        vector <Mat> channels;
        split(hsiImage, channels);
        Hvalue = channels.at(0);
        Svalue = channels.at(1);
        Ivalue = channels.at(2);

    for (int i = 0; i < rbgImage.rows; ++i)
        for (int j = 0; j < rbgImage.cols; ++j)
        {
            double H, S, I;
            int Bvalue = rbgImage.at<Vec3b>(i, j)(0);
            int Gvalue = rbgImage.at<Vec3b>(i, j)(1);
            int Rvalue = rbgImage.at<Vec3b>(i, j)(2);

            //求Theta =acos((((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2) / sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*(Gvalue - Bvalue)));
            double numerator = ((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2;
            double denominator = sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*(Gvalue - Bvalue));
            if (denominator == 0) H = 0;
            else{
                double Theta = acos(numerator / denominator) * 180 / 3.14;
                if (Bvalue <= Gvalue)
                     H = Theta;
                else  H = 360 - Theta;
            }       
            Hvalue.at<uchar>(i, j) = (int)(H * 255 / 360); //为了显示将[0~360]映射到[0~255]

            //求S = 1-3*min(Bvalue,Gvalue,Rvalue)/(Rvalue+Gvalue+Bvalue);
            int minvalue = Bvalue;
            if (minvalue > Gvalue) minvalue = Gvalue;
            if (minvalue > Rvalue) minvalue = Rvalue;
            numerator = 3 * minvalue;
            denominator = Rvalue + Gvalue + Bvalue;
            if (denominator == 0)  S = 0;
            else{
                S = 1 - numerator / denominator;
            }
            Svalue.at<uchar>(i, j) = (int)(S * 255);//为了显示将[0~1]映射到[0~255]

            I= (Rvalue + Gvalue + Bvalue) / 3;          
            Ivalue.at<uchar>(i, j) = (int)(I);
        }
	// cout<<"before imageI"<<Ivalue.size()<<endl;
	merge(channels, hsiImage);
	drawHist(Hvalue,std::to_string(num)+"_H_channel",0);
    // outputHist.clear();
	drawHist(Svalue,std::to_string(num)+"_S_channel",1);
    // outputHist.clear();
	drawHist(Ivalue,std::to_string(num)+"_I_channel",2);
    // outputHist.clear();
	// ostringstream os;
	// os << num;
	// imwrite("../result/pic_H_"+std::to_string(num)+".jpg",Hvalue);
	// imwrite("../result/pic_S_"+os.str()+".jpg",Svalue);
	// imwrite("../result/pic_I_"+os.str()+".jpg",Ivalue);
	imwrite("../result/pic/HSI_"+std::to_string(num)+".jpg",hsiImage);
	// imshow("demo",Ivalue);
    setHsiImage(hsiImage);
    setHvalue(Hvalue);
    setIvalue(Ivalue);
    setSvalue(Svalue);
    
    // waitKey(6000);
    }

void RGB2HSI::drawHist(Mat sourcePic,string i,int num){
	// cv::Mat sourcePic = cv::imread("../debug/rgb.jpg", cv::IMREAD_GRAYSCALE);
    // cv::imshow("Source Picture", sourcePic);

    //定义函数需要的一些变量
    //图片数量nimages
    int nimages = 1;
    //通道数量,我们总是习惯用数组来表示，后面会讲原因
    int channels[1] = { 0 };
    //输出直方图
    cv::Mat outputHist_;
    //维数
    int dims = 1;
    //存放每个维度直方图尺寸（bin数量）的数组histSize
    int histSize[1] = { 256 };
    //每一维数值的取值范围ranges
    float hranges[2] = { 0, 255 };
    //值范围的指针
    const float *ranges[1] = { hranges };
    //是否均匀
    bool uni = true;
    //是否累积
    bool accum = false;

    //计算图像的直方图
    cv::calcHist(&sourcePic, nimages, channels, cv::Mat(), outputHist_, dims, histSize, ranges, uni, accum);

    //遍历每个箱子(bin)检验，这里的是在控制台输出的。
    // for (int i = 0; i < 256; i++)
    //     std::cout << "bin/value:" << i << "=" << outputHist.at<float>(i) << std::endl;
    // std::cout<<outputHist<<"++"<<std::endl;

    //画出直方图
    int scale = 1;
    //直方图的图片
    cv::Mat histPic(histSize[0] * scale, histSize[0], CV_8U, cv::Scalar(255));
    //找到最大值和最小值
    double maxValue = 0;
    double minValue = 0;
    cv::minMaxLoc(outputHist_, &minValue, &maxValue, NULL, NULL);
    //测试
    // std::cout << minValue << std::endl;
    // std::cout << maxValue << std::endl;

    //纵坐标缩放比例
    double rate = (histSize[0] / maxValue)*0.9;

    for (int i = 0; i < histSize[0]; i++)
    {
        //得到每个i和箱子的值
        float value = outputHist_.at<float>(i);
        //画直线	
        cv::line(histPic, cv::Point(i*scale, histSize[0]), cv::Point(i*scale, histSize[0] - value*rate), cv::Scalar(0));
    }
	// cout<<""<<endl;
    cv::imwrite("../result/pic/"+i+".jpg", sourcePic);
    cv::imwrite("../result/hist/histgram_"+i+".jpg", histPic);
    outputHist.push_back(outputHist_);
    // cv::waitKey(0); 
    // return outputHist;
}

// cv::Mat RGB2HSI::getHsiImage(){
    
// }