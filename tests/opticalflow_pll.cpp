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
#include <time.h>
#include <omp.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	// Initialize, load two images from the file system, and
	// allocate the images and other structures we will need for
	// results.

	if(argc!=3){
		printf("Insufficient / Extra arguements provided\n");
		return -1;
	}

	char * image1,*image2;
	image1 = argv[1];
	image2 = argv[2];

	Mat imgA = imread(image1,CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgB = imread(image2,CV_LOAD_IMAGE_GRAYSCALE);

	if(!imgA.data || !imgB.data){
		printf("Either or both of the images are empty\n");
		return -1;
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

	clock_t begin = clock();

	// Call the Lucas Kanade algorithm
	setNumThreads(1);
	calcOpticalFlowPyrLK(imgA, imgB, cornersA, cornersB, features_found, noArray(),
	Size(win_size*4+1,win_size*4+1), 5,
	TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ));

	// Now make some image of what we are looking at:
	setNumThreads(4);
	#pragma omp parallel for
	for( int i = 0; i < (int)cornersA.size(); i++ ) {
		if( !features_found[i] )
		continue;
		arrowedLine(imgC, cornersA[i], cornersB[i], Scalar(0,0,255),1, CV_AA);
	}

	clock_t end = clock();
	printf("%f", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
