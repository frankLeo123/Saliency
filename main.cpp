#include <iostream>  
#include <sys/time.h>
#include <vector>
#include "opencv2/opencv.hpp" 
#include "SLIC.h"
#include "math.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <sstream>
#include "RGB2HSI.h"
using namespace std;
using namespace cv;
//整理思路，首先代码分为三个部分，
//1.分解HSI分量；
//2.对于HSI画直方图，求出max与min，通过求取波峰波谷/nums作为算子，筛选出合适的通道；
//3.单通道求SLIC效果图，再贴到原图中；
int imgOpenCV2SLIC(Mat img, int &height, int &width, int &dim, unsigned int * &image);
int imgSLIC2openCV(unsigned int *image, int height, int width, int dim, Mat &imgSLIC);

int main()
{

	vector<Mat> imageRGB(4);
	vector<Mat> imageHSI(4);
	vector<Mat> imageH(4);
	vector<Mat> imageS(4);
	vector<Mat> imageI(4);


	time_t tStart,tEnd;
	double exeT;

	unsigned int *image; 
	unsigned int *image_r; 

	int height; 
	int width; 
	int dim;
	long imgSize;
	struct timeval{  
    long int tv_sec; // 秒数  
    long int tv_usec;};
	int MAX_KERNEL_LENGTH = 70;//31;
	
	int numlabels(0);
	SLIC slic;
	int m_spcount= 500 ;
	int m_compactness=10;
	imgSize = height* width;
	int* labels = new int[imgSize];



	imageRGB[0]= imread("/home/frank/Documents/lab/img/bird.jpg");
	imageRGB[1]= imread("/home/frank/Documents/lab/img/clock.jpg");
	imageRGB[2]= imread("/home/frank/Documents/lab/img/leaf.jpg");
	imageRGB[3]= imread("/home/frank/Documents/lab/img/plane.jpg");
	if (imageRGB.empty() == true){
		cout<<"can not open rgb image!"<<endl;
	}
//获取hsi
	Mat temp_h,temp_s,temp_i;
	vector<cv::Mat> hist;
	//梯度
	// Mat ;
	// for(int i=0;i<imageRGB.size();i++){
	// 	cout<<"image num:"<<i<<endl;
	// 	RGB2HSI rgb2HSI;
	// 	rgb2HSI.rgb2HSI(imageRGB[i],imageHSI[i],imageH[i],imageS[i],imageI[i],i);
	// 	temp_h=rgb2HSI.getHvalue().clone();
	// 	temp_s=rgb2HSI.getSvalue().clone();
	// 	temp_i=rgb2HSI.getIvalue().clone();
	// 	hist.push_back(rgb2HSI.getHistmat());
	// 	// cout<<hist[0].size()<<std::endl;
	// 	// waitKey(0);
	// }
	//Gaussian blur 
	std::vector<cv::Mat> gaussian_dst(4);
	
	for ( int j = 0; j < imageRGB.size(); j++){
		for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        { 
			cv::Mat src = imageRGB[j].clone();
			
			GaussianBlur( src, gaussian_dst[j], Size( i, i ), 0, 0 );
			// cout<<"ok"<<endl;
		}
	}
	
	RGB2HSI rgb2HSI;
	int num_i = 3;
	// for(int num_i; num_i < imageRGB.size();num_i++){
	rgb2HSI.rgb2HSI(gaussian_dst[num_i],imageHSI[num_i],imageH[num_i],imageS[num_i],imageI[num_i],num_i);
	
	hist = rgb2HSI.getHistmat();
	// hist.push_back(rgb2HSI.getSistmat());
	// hist.push_back(rgb2HSI.getIistmat());
	#if 1
	int tempNum[3] = {0,0,0};
	int tempNumhigh[3] = {0,0,0};
	const int INI = 25;//45
	int flag = 0;
	int flagSmall = 0;
	int numDouble = 0;
	int dist = 0;
	int idx = 0;
	// cout<<hist.size()<<endl;
	for(int j = 0; j < hist.size(); j++){
		numDouble = 0;
		dist = 0;
		for(int i=1; i < 256; i++){	
			int big = ((int)(hist[j].at<float>(i)-hist[j].at<float>(i+1)));
			int small = (int)(hist[j].at<float>(i)-hist[j].at<float>(i-1));
			if((small < -INI) && (big < -INI)){
				++tempNum[j];
				flag++;
				idx = i;
			}			
			else if((small > INI) && (big > INI)){
				++tempNumhigh[j];
				flagSmall++;
			}
			if((flag > 0) && (flagSmall > 0)){
				flag = 0;
				flagSmall = 0;
				++numDouble;
				dist += (int)(hist[j].at<float>(i)-hist[j].at<float>(idx));
				cout<<"win :"<<big<<"small"<<small<<endl;
				cout<<"win :"<<hist[j].at<float>(i)<<"small"<<hist[j].at<float>(idx)<<endl;
			}	
		}
		std::cout<< "image["<<num_i<<" :"<<"j:=="<<j<<"num high === "<<tempNumhigh[j]<<endl;
		std::cout<< "image["<<num_i<<" :"<<"j:=="<<j<<"num min ==="<<tempNum[j]<<endl;
		std::cout<< "image["<<num_i<<" :"<<"j:=="<<j<<"num  ==="<<numDouble<<endl;
		int result = dist/numDouble;
		cout<<" dist :"<<dist<<endl;
		cout<<"num: "<<numDouble<<endl;
		cout<< "result:"<<result << endl;
	
	}
	#endif
	// }
	
	
	
	// cout << hist[0] << endl;

// 超像素分割
#if 0
	imgOpenCV2SLIC(temp_h,  height,  width,  dim, image_r);
	imgOpenCV2SLIC(imageRGB[i],  height,  width,  dim, image);
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(image_r, width, height, labels, numlabels, m_spcount, m_compactness);
	Mat imgtemp;
	slic.DrawContoursAroundSegments(image, labels, width, height, 0);
	imgSLIC2openCV(image, height,width,dim,imgtemp);//SLIC结果：SLIC图像数据转换成OpenCV 图像数据
	imwrite("../debug/test/test_1.jpg",imgtemp);
	cout<<rgb2HSI.getHsiImage().size()<<endl;
#endif

	#if 0
	cv::Mat demo_hsv;
	cv::cvtColor(imageRGB[0],demo_hsv,CV_RGB2HSV);
	
	//验证hsv与hsi效果
	imwrite("../debug/hsv.jpg",demo_hsv);
	cv::Mat channel1;
	cv::Mat channel2;
	cv::Mat channel3;
	vector<cv::Mat>channels;
	split(demo_hsv,channels);
	channel1=channels.at(0);
	channel2=channels.at(1);
	channel3=channels.at(2);
	imwrite("../debug/h.jpg",channel1);
	imwrite("../debug/s.jpg",channel2);
	imwrite("../debug/v.jpg",channel3);
	#endif
	
	// imgOpenCV2SLIC(imageRGB[0],  height,  width,  dim, image);//OpenCV 图像数据转换成SLIC图像数据
	
	
	tStart=clock();

	// imgOpenCV2SLIC(rgb[0],  height,  width,  dim, image_r);
	// // //SLIC超像素分割，代码下载网站：http://ivrl.epfl.ch/research/superpixels#SLICO
	// slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(image_r, width, height, labels, numlabels, m_spcount, m_compactness);
	// // // Mat imgtemp;
	// imgSLIC2openCV(image, height,width,dim,imgtemp);//SLIC结果：SLIC图像数据转换成OpenCV 图像数据
	
	
	// slic.DrawContoursAroundSegments(image_r, labels, width, height, 0);

	// tEnd=clock();
	// exeT=(tEnd-tStart)/(double)CLOCKS_PER_SEC;

	// Mat imgSLIC;

	// imgSLIC2openCV(image_r, height,width,dim,imgSLIC);//SLIC结果：SLIC图像数据转换成OpenCV 图像数据


	// //结果显示
	// cout<<"SLIC执行时间exeT："<<exeT<<"秒"<<endl;
	// // imshow("imgRGB",imgRGB);
	// // imshow("imgSLIC1",imgSLIC);
	// imwrite("../debug/rgb.jpg",imgRGB);
	// imwrite("../debug/test_plane.jpg",imgSLIC);
	// waitKey();
	return 0;

}


//OpenCV Mat图像数据转换为SLIC图像数据
//输入：Mat img, int &height, int &width, int &dim,
//输出：unsigned int * &image，同时返回转换是否成功的标志：成功为0，识别为1
int imgOpenCV2SLIC(Mat img, int &height, int &width, int &dim, unsigned int * &image)
{
	int error=0;
	if( img.empty() ) //请一定检查是否成功读图 
	{ 
		error =1;
	} 

	dim=img.channels();//图像通道数目
	height=img.rows;
	width=img.cols;

	int imgSize=width*height;

	unsigned char *pImage  = new unsigned char [imgSize*4];
	if(dim==1){
		for(int j = 0; j < height; j++){
			uchar * ptr = img.ptr<uchar>(j);
			for(int i = 0; i < width; i++) {
				pImage[j * width*4 + 4*i+3] = 0;
				pImage[j * width*4 + 4*i+2] = ptr[0];
				pImage[j * width*4 + 4*i+1] = ptr[0];
				pImage[j * width*4 + 4*i] = ptr[0];		
				ptr ++;
			}
		}
	}
	else{
		if(dim==3){
			for(int j = 0; j < height; j++){
				Vec3b * ptr = img.ptr<Vec3b>(j);
				for(int i = 0; i < width; i++) {
					pImage[j * width*4 + 4*i+3] = 0;
					pImage[j * width*4 + 4*i+2] = ptr[0][2];//R
					pImage[j * width*4 + 4*i+1] = ptr[0][1];//G
					pImage[j * width*4 + 4*i]   = ptr[0][0];//B		
					ptr ++;
				}
			}
		}
		else  error=1;

	}

	image = new unsigned int[imgSize];
	memcpy( image, (unsigned int*)pImage, imgSize*sizeof(unsigned int) );
	delete pImage;

	return error;

}


//SLIC图像数据转换为OpenCV Mat图像数据
//输入：unsigned int *image, int height, int width, int dim
//输出：Mat &imgSLIC ，同时返回转换是否成功的标志：成功为0，识别为1
int imgSLIC2openCV(unsigned int *image, int height, int width, int dim, Mat &imgSLIC)
{
	int error=0;//转换是否成功的标志：成功为0，识别为1

	if(dim==1){
		imgSLIC.create(height, width, CV_8UC1);
		//遍历所有像素，并设置像素值 
		for( int j = 0; j< height; ++j) 
		{ 
			//获取第 i行首像素指针 
			uchar * p = imgSLIC.ptr<uchar>(j); 
			//对第 i行的每个像素(byte)操作 
			for( int i = 0; i < width; ++i ) 
				p[i] =(unsigned char)(image[j*width+i]& 0xFF)  ; 
		} 
	}
	else{
		if(dim==3){
			imgSLIC.create(height, width, CV_8UC3);
			//遍历所有像素，并设置像素值 
			for( int j = 0; j < height; ++j) 
			{ 
				//获取第 i行首像素指针 
				Vec3b * p = imgSLIC.ptr<Vec3b>(j); 
				for( int i = 0; i < width; ++i ) 
				{ 
					p[i][0] = (unsigned char)(image[j*width+i]          & 0xFF ); //Blue 
					p[i][1] = (unsigned char)((image[j*width+i] >>  8 ) & 0xFF ); //Green 
					p[i][2] = (unsigned char)((image[j*width+i] >>  16) & 0xFF ) ; //Red 
				} 
			}
		}
		else  error= 1 ;

	}

	return error;
}

// void imshow_pic()
