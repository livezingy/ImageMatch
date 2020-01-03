//#include "stdafx.h"
#include <iostream>
#include  <io.h>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "util.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int LOOP_NUM = 1;
const int GOOD_PTS_MAX = 20;
const float GOOD_PORTION = 0.15f;
const float DEGREE = 57.29578049;
const float PI = 3.1415926;

const string refColorPath = "emblems/ref/color";
const string refChromeImgsPath = "emblems/ref/chrome/images";
const string refChromeFeaPath = "emblems/ref/chrome/features";

const string testPath = "emblems/test";



static int load_features_from_file(const string& file_name, Mat& features)
{
	FILE* fp = fopen(file_name.c_str(), "r");
	if (fp == NULL)
	{
		printf("fail to open %s\n", file_name.c_str());
		return -1;
	}
	//printf("loading file %s\n", file_name.c_str());

	vector<float> inData;
	while (!feof(fp))
	{
		float tmp;
		fscanf(fp, "%f", &tmp);
		inData.push_back(tmp);
	}

	//vector to Mat
	int mat_cols = 128;
	int mat_rows = inData.size() / 128;
	features = Mat::zeros(mat_rows, mat_cols, CV_32FC1);
	int count = 0;
	for (int i = 0; i < mat_rows; i++)
	{
		for (int j = 0; j < mat_cols; j++)
		{
			features.at<float>(i, j) = inData[count++];
		}
	}

	return 0;
}

static int WriteFeatures2File(const string& file_name, const Mat& features)
{
	FILE* fp = fopen(file_name.c_str(), "a+");
	if (fp == NULL)
	{
		printf("fail to open %s\n", file_name.c_str());
		return -1;
	}

	for (int i = 0; i < features.rows; i++)
	{
		for (int j = 0; j < features.cols; j++)
		{
			int data = features.at<float>(i, j);
			fprintf(fp, "%d\t", data);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	return 0;
}

//需要调试
static int load_keypoints_from_file(const string& file_name, vector<KeyPoint> &keyPts)
{
	FILE* fp = fopen(file_name.c_str(), "r");
	if (fp == NULL)
	{
		printf("fail to open %s\n", file_name.c_str());
		return -1;
	}
	//printf("loading file %s\n", file_name.c_str());

	//vector<float> inData;
	while (!feof(fp))
	{
		KeyPoint tmpPts;
		float tmp;
		fscanf(fp, "%f", &tmp);
		tmpPts.pt.x = tmp;

		fscanf(fp, "%f", &tmp);
		tmpPts.pt.y = tmp;

		keyPts.push_back(tmpPts);
	}

	return 0;
}

static int WriteKeypoints2File(const string& file_name, const vector<KeyPoint> &keyPts)
{

	FILE* fp = fopen(file_name.c_str(), "a+");
	if (fp == NULL)
	{
		newUtils::mkdir(file_name.c_str());
		fp = fopen(file_name.c_str(), "a+");
	}


	for (int i = 0; i < keyPts.size(); i++)
	{
		float x = keyPts[i].pt.x;
		fprintf(fp, "%f\t", x);

		float y = keyPts[i].pt.y;
		fprintf(fp, "%f\t", y);

		fprintf(fp, "\n");
	}

	fclose(fp);

	return 0;
}

struct SURFDetector
{
	Ptr<Feature2D> surf;
	SURFDetector()
	{
		surf = SURF::create(100, 4, 4, true, false);
	}
	template<class T>
	void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	{
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

struct SIFTDetector
{
	Ptr<Feature2D> sift;
	SIFTDetector()
	{
		sift = SIFT::create();//SIFT::create(0, 3, 1, 15, 1.6);
	}
	template<class T>
	void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	{
		sift->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

template<class KPMatcher>
struct SURFMatcher
{
	KPMatcher matcher;
	template<class T>
	void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
	{
		matcher.match(in1, in2, matches);
	}
};


static void findGoodMatchePoints(Mat imageDesc1, Mat imageDesc2, vector<DMatch> &GoodMatchePoints)
{
	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matchePoints;

	vector<Mat> train_desc(1, imageDesc1);
	matcher.add(train_desc);
	matcher.train();

	matcher.knnMatch(imageDesc2, matchePoints, 2);
	cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm,获取优秀匹配点
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i][0].distance < 0.6 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}
}

static void findGoodMatcheImage(Mat imageDesc1, vector<Mat>& refImgDescs, int &matchIndex,vector<DMatch> &GoodMatchePoints)
{
	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matchePoints;

	//vector<Mat> train_desc(1, imageDesc1);
	matcher.add(refImgDescs);
	matcher.train();

	matcher.knnMatch(imageDesc1, matchePoints, 2);
	//cout << "total match points: " << matchePoints.size() << endl;

	//matchIndex = matcher.

	// Lowe's algorithm,获取优秀匹配点
	for (int i = 0; i < matchePoints.size(); i++)
	{
		//matchIndex = matchePoints[i].imgIdx;
		if (matchePoints[i][0].distance < 0.6 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
			matchIndex = matchePoints[i][0].imgIdx;
		}
	}
}



//get sitar from sift feature 
static float calAngle(Point2f p1_I1, Point2f p2_I1, Point2f p1_I2, Point2f p2_I2)
{
	float a = sqrt((p1_I1.x - p2_I1.x) * (p1_I1.x - p2_I1.x) + (p1_I1.y - p2_I1.y) * (p1_I1.y - p2_I1.y));
	float b = sqrt((p1_I2.x - p2_I2.x) * (p1_I2.x - p2_I2.x) + (p1_I2.y - p2_I2.y) * (p1_I2.y - p2_I2.y));
	float c = (p1_I1.x - p2_I1.x) * (p1_I2.x - p2_I2.x) + (p1_I1.y - p2_I1.y) * (p1_I2.y - p2_I2.y);

	//待测图片与参考图片的夹角(不含方向)
	float sitar = acos(c / (a * b)) * DEGREE;

	//判断待测图像相对于参考图片的旋转方向
	//参考图像与X轴正向的夹角
	float sitarRef = asin((p1_I1.y - p2_I1.y) / a);
	if (p1_I1.x > p2_I1.x)
	{
		sitarRef = PI - sitarRef;
	}
	//待测图像与X轴正向的夹角
	float sitarTest = asin((p1_I2.y - p2_I2.y) / b);
	if (p1_I1.x > p2_I1.x)
	{
		sitarTest = PI - sitarTest;
	}

	//若待测图夹角大于参考图，则待测图需要逆时针旋转sitar角度到参考图的方向，故角度为负值
	if (sitarTest < sitarRef)
	{
		return -sitar;//逆时针
	}
	else
	{
		return sitar;//顺时针
	}
}

//在较好的匹配点中找到单张图中距离最大的两个点，用于计算待处理图片的旋转角度
static float getRotatedAngle(Mat &refImg, Mat &testImg,std::vector<DMatch>& matches, const std::vector<KeyPoint>& keypointsR, const std::vector<KeyPoint>& keypointsT)
{
	Mat refImage = refImg.clone();
	Mat testImage = testImg.clone();
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());
	std::vector< DMatch > good_matches;
	//double minDist = matches.front().distance;
	//double maxDist = matches.back().distance;

	const int ptsPairs = GOOD_PTS_MAX;//std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	//std::cout << "\nMax distance: " << maxDist << std::endl;
	//std::cout << "Min distance: " << minDist << std::endl;

	//std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

	//计算标准图中滤除后的匹配点之间的直线距离，取出距离最长的两个点	
	float maxLen = 0;
	int index1 = 0;
	int index2 = 0;

	for (int m = 0; m < ptsPairs; m++)
	{
		Point2f pt1 = keypointsR[good_matches[m].queryIdx].pt;

		for (int n = ptsPairs - 1; n > m; n--)
		{
			Point2f pt2 = keypointsR[good_matches[n].queryIdx].pt;

			float tmpLen = (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y);

			if (tmpLen > maxLen)
			{
				maxLen = tmpLen;
				index1 = m;
				index2 = n;
			}
		}
	}


	//根据参考图中的距离最长的两个点，计算测试图相对于参考图的旋转角度
	Point2f pt1_Ref = keypointsR[good_matches[index1].trainIdx].pt;
	Point2f pt2_Ref = keypointsR[good_matches[index2].trainIdx].pt;

	Point2f pt1_Test = keypointsT[good_matches[index1].queryIdx].pt;
	Point2f pt2_Test = keypointsT[good_matches[index2].queryIdx].pt;

	line(refImage, pt1_Ref, pt2_Ref,Scalar(0, 255, 0), 2, LINE_AA);

	line(testImage, pt1_Test, pt2_Test, Scalar(0, 255, 255), 2, LINE_AA);



	//waitKey(0);

	return (calAngle(pt1_Ref, pt2_Ref, pt1_Test, pt2_Test));
}


static RotatedRect getMinRect(Mat& sourceImage)
{
	Mat threshImg;
	threshold(sourceImage, threshImg, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(threshImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//默认取面积最大的轮廓作为目标轮廓
	int maxCount = 0;
	int maxIndex = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > maxCount)
		{
			maxCount = contours[i].size();

			maxIndex = i;
		}
	}
	
	return minAreaRect(contours[maxIndex]);
}

static void rotate_bound(Mat &testImage, float angle, Mat &rotatedImg)
{
	int h = testImage.rows;
	int w = testImage.cols;

	float cX = (float)w / 2;
	float cY = (float)h / 2;

	Mat M = getRotationMatrix2D(Point2f(cX,cY), -angle,1.0);
	/*
	float cos = abs(M.at<int>(0, 0));
	float sin = abs(M.at<int>(0, 1));

	int nW = int((h * sin) + (w * cos));
	int nH = int((h * cos) + (w * sin));

	M.at<int>(0, 2) += (nW / 2) - cX;
	M.at<int>(1, 2) += (nH / 2) - cY;
	*/

	warpAffine(testImage, rotatedImg,M, Size(w,h));
}


////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
	//SURFDetector sift;
	SIFTDetector sift;
	vector<Mat> featuressRef;
	vector<vector<KeyPoint>> keypointsRef;

	TickMeter tm;


	char buffer[60] = { 0 };
	std::sprintf(buffer, refChromeImgsPath.c_str());
	vector<string> imgRefPaths = newUtils::getFiles(buffer);
	vector<string> refName;
	vector<Mat> refImg;

	tm.start();
	//获取参考图库中图片的特征
	for (auto f : imgRefPaths)
	{
		Mat tmpImg = imread(f);
		refImg.push_back(tmpImg);
		UMat image;
		imread(f, IMREAD_GRAYSCALE).copyTo(image);
		if (image.empty())
		{
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.c_str());
			continue;
		}
		else
		{
			string fileName = Utils::getFileName(f);
			string desFileName = refChromeFeaPath + "/descriptor/" + fileName + ".txt";
			string keyFileName = refChromeFeaPath + "/keypoints/" + fileName + ".txt";

			//若特征点或描述文件有任何一个不存在，都将重新训练参考图库
			if ((_access(desFileName.c_str(), 0) == -1) || (_access(keyFileName.c_str(), 0) == -1))
			{
				std::vector<KeyPoint> keypoints1;

				UMat _descriptors1;
				Mat descriptors1 = _descriptors1.getMat(ACCESS_RW);
				sift(image.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);

				WriteFeatures2File(desFileName, descriptors1);

				WriteKeypoints2File(keyFileName, keypoints1);

				featuressRef.push_back(descriptors1);
				keypointsRef.push_back(keypoints1);
				refName.push_back(fileName);
			}
			else
			{
				Mat descriptor;
				load_features_from_file(desFileName, descriptor);
				featuressRef.push_back(descriptor);

				std::vector<KeyPoint> keypoints;
				load_keypoints_from_file(keyFileName, keypoints);
				keypointsRef.push_back(keypoints);

				refName.push_back(fileName);
			}
		}
	}
	tm.stop();
	double getRefTime = tm.getTimeMilli();
	cout << "get reference sift feature time: " << getRefTime << " ms; " << endl;
	cout << ">" << endl;

	//std::cout << "Couldn't load " << leftName << std::endl;

	int refNum = featuressRef.size();
	//获取测试图片的特征
	std::sprintf(buffer, testPath.c_str());
	vector<string> imgTestPaths = newUtils::getFiles(buffer);

	for (auto f : imgTestPaths)
	{
		tm.reset();
		tm.start();
		Mat testImg = imread(f);
		UMat image;
		imread(f, IMREAD_GRAYSCALE).copyTo(image);
		string testFileName = Utils::getFileName(f);

		if (image.empty())
		{
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.c_str());
			continue;
		}
		else
		{
			//resize(image, image, Size(image.cols * 0.5, image.rows * 0.5), 0, 0, INTER_LINEAR);

			std::vector<KeyPoint> keypoints1;
			vector<DMatch> GoodMatchePoints;
			//UMat _descriptors1;
			Mat descriptors1;// = _descriptors1.getMat(ACCESS_RW);

			Mat sourceImg = image.getMat(ACCESS_READ);
			sift(sourceImg, Mat(), keypoints1, descriptors1);
			//int maxValue = 0;
			//int maxIndex = 0;
			int matchIdx = 0;
			vector<DMatch> GoodPoints;
			findGoodMatcheImage(descriptors1, featuressRef, matchIdx, GoodMatchePoints);
			

			tm.stop();
		
			double singleTestTime = tm.getTimeMilli();
			/*
			for (int i = 0; i < refNum; i++)
			{
				vector<DMatch> tmpGoodPoints;
				//GoodMatchePoints.clear();
				findGoodMatchePoints(featuressRef[i], descriptors1, tmpGoodPoints);

				int tmpVal = tmpGoodPoints.size();
				//匹配最多的认为是相对应的图像
				if (tmpVal > maxValue)
				{
					GoodMatchePoints.clear();
					maxValue = tmpVal;
					maxIndex = i;
					GoodMatchePoints.assign(tmpGoodPoints.begin(), tmpGoodPoints.end());
				}
			}
			*/
			//计算当前测试图片与匹配参考图片之间的旋转角度
			float rotatedAngle = getRotatedAngle(refImg[matchIdx], testImg,GoodMatchePoints, keypointsRef[matchIdx], keypoints1);

			//获取当前测试图片最小包围矩形
			RotatedRect rrect = getMinRect(sourceImg);
			

			Mat roImg;
			rotate_bound(testImg, rotatedAngle, roImg);
			
			Point2f rect_points[4];
			rrect.points(rect_points);
			line(testImg, rect_points[0], rect_points[1], Scalar(0, 0, 255), 2, LINE_AA);
			line(testImg, rect_points[1], rect_points[2], Scalar(0, 0, 255), 2, LINE_AA);
			line(testImg, rect_points[2], rect_points[3], Scalar(0, 0, 255), 2, LINE_AA);
			line(testImg, rect_points[3], rect_points[0], Scalar(0, 0, 255), 2, LINE_AA);

			Point2f center0 = Point2f((rect_points[0].x + rect_points[1].x) / 2, (rect_points[0].y + rect_points[1].y) / 2);
			Point2f center1 = Point2f((rect_points[1].x + rect_points[2].x)/2,(rect_points[1].y + rect_points[2].y)/2);
			Point2f center2 = Point2f((rect_points[2].x + rect_points[3].x) / 2, (rect_points[2].y + rect_points[3].y) / 2);
			Point2f center3 = Point2f((rect_points[0].x + rect_points[3].x) / 2, (rect_points[0].y + rect_points[3].y) / 2);
			
			
			line(testImg, center0, center2, Scalar(0, 0, 255), 1, LINE_AA);

			line(testImg, center1, center3, Scalar(0, 0, 255), 1, LINE_AA);

			String strAngle = format("%.2f", rotatedAngle);
			std::string text = "The slope angle is " + strAngle + " degree."; 

			cv::putText(testImg, text, cv::Point(10, 20),0, 0.5, (255, 0, 255), 1);

			cout << "single image test time: " << singleTestTime << " ms; " << endl;
			cout << ">" << endl;
			cout << "The type of the test image is " << refName[matchIdx] << endl;
			cout << ">" << endl;
			cout << "The slope angle of the image is " << strAngle << endl;
			cout << ">" << endl;
			//imshow("rotateImage", roImg);

			//imshow("refImage", refImg[matchIdx]);
			//string fileName = 
			imshow(testFileName, testImg);
			waitKey(0);
			//Mat corrImg = 
		}

		
	}
	destroyAllWindows();

	return EXIT_SUCCESS;
}

