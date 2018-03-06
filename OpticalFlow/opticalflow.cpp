/* License:
   July 20, 2011
   Standard BSD
   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130
   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <string>

using std::string;
using namespace cv;
using namespace std;

void lkk(string image1, string image2, int k);

//The number of images in the inputs folder
#define NUM_INPUTS 102


int main() {
	int k;
	struct timeval start,end;
	double elapsedTime;

	gettimeofday(&start,NULL);

  //Run the algorithm on every two consecutive images
	for(int i=0;i<NUM_INPUTS-1;i++){
		string im1="inputs/",im2="inputs/";
		k=i+1;
		im1 += to_string(k);
		im1 += ".jpg";
		im2 += to_string(k+1);
		im2 += ".jpg";
		lkk(im1,im2,k);
		printf("[%d] Completed algorithm run on images %d.jpg and %d.jpg\n",k,k,k+1);
	}

	gettimeofday(&end,NULL);;

	elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	printf("Elapsed: %.3f seconds\n", elapsedTime / 1000);
	return 0;
}

void lkk(string image1,string image2,int k)
	{

		string outputImage = "outputs/result_";
		outputImage += to_string(k);
		outputImage += ".jpg";

		Mat imgA = imread(image1,CV_LOAD_IMAGE_GRAYSCALE);
		Mat imgB = imread(image2,CV_LOAD_IMAGE_GRAYSCALE);

		if(!imgA.data || !imgB.data){
			printf("Either or both of the images %s and %s are empty\n", image1.c_str(),image2.c_str());
			return ;
		}

		Size img_sz = imgA.size();
		int win_size = 10;
		Mat imgC = imread(image1,CV_LOAD_IMAGE_UNCHANGED);

		// The first thing we need to do is get the features
		// we want to track.
		vector<Point2f> cornersA, cornersB;
		const int MAX_CORNERS = 500;
		goodFeaturesToTrack(imgA, cornersA, MAX_CORNERS, 0.01, 5, noArray(), 3, false, 0.04);

		cornerSubPix(imgA, cornersA, Size(win_size, win_size), Size(-1,-1),
		TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));

		vector<uchar> features_found;

		// Call the Lucas Kanade algorithm
		calcOpticalFlowPyrLK(imgA, imgB, cornersA, cornersB, features_found, noArray(),
		Size(win_size*4+1,win_size*4+1), 5,
		TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ));

		// Now make some image of what we are looking at:
		for( int i = 0; i < (int)cornersA.size(); i++ ) {
			if( !features_found[i] )
			continue;
			arrowedLine(imgC, cornersA[i], cornersB[i], Scalar(0,0,255),1, CV_AA);
		}
		imwrite( outputImage, imgC );
	}
