#include "kyheader.h"
#include "Objectness.h"
#include "CmShow.h"
#include <math.h>
//#include "../matplotlibcpp.h"
#include<iostream>
#include<vector>
#include<algorithm>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#include"Matplot.h"
//void plot_recall_MABO(vecD &recalls,vecD &avgScore) ;

void print_null(const char *s) {}
const int CN = 21; // Color Number
const char* COLORs[CN] = {"'k'", "'b'", "'g'", "'r'", "'c'", "'m'", "'y'",
	"':k'", "':b'", "':g'", "':r'", "':c'", "':m'", "':y'",
	"'--k'", "'--b'", "'--g'", "'--r'", "'--c'", "'--m'", "'--y'"
};


// base for window size quantization, R orientation channels, and feature window size (_W, _W)
Objectness::Objectness(DataSetVOC &voc,int NSS)
	: _voc(voc)
	, _NSS(NSS)
	, _logBase(log(2.0))
	, _minT(cvCeil(log(10.)/_logBase))
	, _maxT(cvCeil(log(500.)/_logBase))
	, _numT(_maxT - _minT + 1)
{
	setmodelpath();
}

Objectness::~Objectness(void)
{
}

void Objectness::setmodelpath()
{
	
	_modelName = _voc.resDir + format("OPM");
	_trainDirSI = _voc.localDir + format("OPM");
	_bbResDir = _voc.resDir + format("OPM");
}


int Objectness::loadTrainedModel(string modelName) 
{
	
	Mat filters1f, reW1f, idx1i, show3u;
	
	CStr s11 = "./modle_OPM/OPM.wS1", s12 = "./modle_OPM/OPM.wS2", s1I = "./modle_OPM/OPM.idx";
	//Mat filters1f, reW1f, idx1i, show3u;
	if (!matRead(s11, filters1f) || !matRead(s1I, idx1i)){
		printf("Can't load model: %s or %s\n", _S(s11), _S(s1I));
		return 0;
	}

	normalize(filters1f, show3u, 1, 255, NORM_MINMAX, CV_8U);
	//CmShow::showTinyMat(_voc.resDir + "Filter.png", show3u);
	_synbihl.update(filters1f);
	_synbihl.reconstruct(filters1f);

	_svmSzIdxsbihl = idx1i;
	CV_Assert(_svmSzIdxsbihl.size() > 1 && filters1f.size() == Size(8, 8) && filters1f.type() == CV_32F);
	_svmFilterbihl = filters1f;

	if (!matRead(s12, _svmReW1fbihl) || _svmReW1fbihl.size() != Size(2, _svmSzIdxsbihl.size())){
		_svmReW1fbihl = Mat();
		return -1;
	}
	
	return 1;
}

void Objectness::predictBBoxSI(CMat &img3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ, bool fast)
{
	const int numSz = _svmSzIdxsbihl.size(); //number of class
        
	const int imgW = img3u.cols, imgH = img3u.rows;

        Mat nmsMax1 = Mat::ones(imgW, imgH, CV_8U),nmsMax2 = Mat::ones(imgW, imgH, CV_8U);

	valBoxes.reserve(10000);
	sz.clear(); sz.reserve(10000);

	for (int ir = numSz -1; ir >= 0; ir--){

		int r = _svmSzIdxsbihl[ir]; 

		int height = cvRound(pow(2.0, r/_numT + _minT)), width = cvRound(pow(2.0, r%_numT + _minT));
		//if (height > imgH * 2.0 || width > imgW * 2.0||height >=imgH || width >= imgW||(height >=256 && width >= 256)||(height ==128 && width == 16)||(height ==256 && width == 32))
		//	continue;

		height = min(height, imgH), width = min(width, imgW);
		Mat im3u, matchCost1f, mag1uA, mag1uB, mag1uC, mag1u,m_roi1;
                
		float valuesvm=0;
		resize(img3u, im3u, Size(cvRound(imgW*8.0/width)*2, cvRound(imgH*8.0/height)*2));

		frequencyHL(im3u, mag1uA);

		const float* svmIIwbihl = _svmReW1fbihl.ptr<float>(ir);
		matchCost1f = _synbihl.matchTemplate(mag1uA);

		ValStructVec<float, Point> matchCostbihl;
		vector<Vec4f> _boxesmerge;

                 nonMaxSup(matchCost1f, matchCostbihl,_boxesmerge, _NSS,NUM_WIN_PSZ,0);
		// Find true locations and match values
		double ratioX = width/8, ratioY = height/8;
		int iMax = min(matchCostbihl.size(), NUM_WIN_PSZ)*0.91;
                // iMax =iMax>900?cvRound(iMax*0.90):cvRound(iMax*0.9);
               
		int widths,heights;
#pragma omp parallel for
		for (int i = 0; i < iMax; i++){
			float mVal = matchCostbihl(i);
                        mVal =  mVal*svmIIwbihl[0] + svmIIwbihl[1]; 
                        if (mVal>-0.9997){//0.9997
			Point pnt = matchCostbihl[i];
			Vec4i boxbihl(cvRound(pnt.x * ratioX), cvRound(pnt.y*ratioY));
                        _synbihl.Boundary_limit(boxbihl,_boxesmerge[i],height,width,imgH,imgW);
			
			valBoxes.pushBack(mVal, boxbihl); 
			sz.push_back(ir);


                       }
                 }

	}
}

void Objectness::predictBBoxSII(ValStructVec<float, Vec4i> &valBoxes, const vecI &sz)
{
	int numI = valBoxes.size();
	for (int i = 0; i < numI; i++){
		const float* svmIIw = _svmReW1f.ptr<float>(sz[i]);
		valBoxes(i) = valBoxes(i) * svmIIw[0] + svmIIw[1]; 
	}
	valBoxes.sort();
}


void Objectness::getObjBndBoxes(CMat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize)
{
	
	CV_Assert_(filtersLoadedbihl() , ("SVM bihl filters should be initialized before getting object proposals\n"));
	//ValStructVec<float, Vec4i> &valBoxesA,&valBoxesB,&valBoxesC;
	vecI sz;
	predictBBoxSI(img3u, valBoxes, sz, numDetPerSize, false);

	return;
}


double T_background;
void Objectness::nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, vector<Vec4f> &_boxesmerge, int NSS, int maxPoint, bool fast) //NSS=2,maxPoint=130
{
	//vector<Vec4i> &_boxesTests;
	//_boxesmerge.resize(matchCost.size());
	//for (int j = 0; j < matchCost.size(); j++)
	//_boxesmerge[j] = matchCost[j];
	const int _h = matchCost1f.rows, _w = matchCost1f.cols;
	Mat isMax1u = Mat::ones(_h, _w, CV_8U),isMax1u2 = Mat::ones(_h, _w, CV_8U), costSmooth1f,isNub1u = Mat::zeros(_h, _w, CV_16U)+65500,seqImage = Mat::zeros(_h, _w, CV_16U)+65500;
	Mat resultimage = cv::Mat::ones(_h, _w, CV_8UC1);
	ValStructVec<float, Point> valPnt;
	ValStructVec<float, Point> valPnt2;
	matchCost.reserve(_h * _w);
	_boxesmerge.resize(_h * _w);
	valPnt.reserve(_h * _w);
	valPnt2.reserve(_h * _w);
	int push_r=0;
	if (fast){
		medianBlur(matchCost1f, costSmooth1f, 3);
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			const float* ds = costSmooth1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				if (d[c] >= ds[c]){
					valPnt.pushBack(d[c], Point(c, r));


				}
		}

	}
	else{ 
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			for (int c = 0; c < _w; c++){
                               if(d[c]>=T_background)
				valPnt.pushBack(d[c], Point(c, r));}
		}
	}

	valPnt.sort();
	int push_i=0;
	//vector<int> *grow_add;
	//int *grow_add;


        for (int is = 0; is <valPnt.size(); is++){
		Point &pnt6 = valPnt[is];
                float mval=valPnt(is);
		if (isMax1u2.at<byte>(pnt6)){
			valPnt2.pushBack(mval, pnt6);
                        int NSS2=NSS;
			for (int dy = -NSS2; dy <= NSS2; dy++) for (int dx = -NSS2; dx <= NSS2; dx++){
				Point neighbor6 = pnt6 + Point(dx, dy);
				if (!CHK_IND(neighbor6))
					continue;
				isMax1u2.at<byte>(neighbor6) = 0;
			}
		}
                if (valPnt2.size() >= 0.4*valPnt.size())
			break;
		
	}


        //cout<<" valPnt.size()= "<<valPnt.size()<<"   valPnt2.size()="<<valPnt2.size()<<endl;

	//for (int i = 0; i < valPnt.size(); i++)
	//	seqImage.at<ushort>(valPnt[i])=i;

	//#pragma omp parallel for
	for (int i = 0; i < (valPnt2.size()>1100?1100:valPnt2.size()); i++){
		Point &pnt = valPnt2[i];
		_boxesmerge[i][0]=0;
		_boxesmerge[i][1]=0;
		_boxesmerge[i][2]=0;
		_boxesmerge[i][3]=0;
		//push_r=i;



		if (isMax1u.at<byte>(pnt)){
                         
			matchCost.pushBack(valPnt2(i), pnt);

			for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
				Point neighbor = pnt + Point(dx, dy);

				if (!CHK_IND(neighbor)) //#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)
					continue;
				if(isMax1u.at<byte>(neighbor) != 0){
					isMax1u.at<byte>(neighbor) = 0;
					isNub1u.at<ushort>(neighbor) = push_i;
				}
				else{

					if(i-isNub1u.at<ushort>(neighbor)<T_S1){
						Point &pnt2 = matchCost[isNub1u.at<ushort>(neighbor)];
						// alpha_1=6.0;
						if(neighbor.x-pnt2.x<=-1){
							_boxesmerge[isNub1u.at<ushort>(neighbor)][0]=max(alpha_1*(pnt2.x-pnt.x),_boxesmerge[isNub1u.at<ushort>(neighbor)][0]);
                                                        //_boxesmerge[isNub1u.at<ushort>(neighbor)][2]=min(alpha_1*(pnt2.x-pnt.x),_boxesmerge[isNub1u.at<ushort>(neighbor)][0]);

                                                }
						if(neighbor.y-pnt2.y<=-1){
							_boxesmerge[isNub1u.at<ushort>(neighbor)][1] =max(alpha_1*(pnt2.y-pnt.y),_boxesmerge[isNub1u.at<ushort>(neighbor)][1]);

                                                }
						if(neighbor.x-pnt2.x>=0){
							_boxesmerge[isNub1u.at<ushort>(neighbor)][2] =max(alpha_1*(pnt.x-pnt2.x),_boxesmerge[isNub1u.at<ushort>(neighbor)][2]);

                                                 }
						if(neighbor.x-pnt2.x>=0){
							_boxesmerge[isNub1u.at<ushort>(neighbor)][3] =max(alpha_1*(pnt.x-pnt2.x),_boxesmerge[isNub1u.at<ushort>(neighbor)][3]);
                                                }

                                          }

				}   


			}
			push_i++;
		}

		if (isMax1u.at<byte>(pnt)==0&&i-isNub1u.at<ushort>(pnt)<T_S2){

			Point &pnt2 = matchCost[isNub1u.at<ushort>(pnt)];

			if(abs(pnt.x-pnt2.x)<2&&abs(pnt.y-pnt2.y)<2&&i-isNub1u.at<ushort>(pnt)<100){
				for (int dy = -6; dy <= 6; dy++) for (int dx = -6; dx <= 6; dx++){

					Point neighbor2 = pnt + Point(dx, dy);
					if (!CHK_IND(neighbor2)) //#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)

						continue;
					//isMax1u.at<byte>(neighbor2) = 0;
					isNub1u.at<ushort>(neighbor2) = isNub1u.at<ushort>(pnt);
				}
			}

		}



		if (matchCost.size() >= maxPoint)
			return;
		
	}
}



void Objectness::frequencyHL(CMat &bgr3u, Mat &mag1u)
{     

	const int Height = bgr3u.cols;
	const int Width = bgr3u.rows;
	Mat img3;
	Mat img;
	cvtColor(bgr3u, img,CV_RGB2GRAY);
	int row=Width;
	int col=Height;


	Mat wavelet;
	Mat tmp;
	wavelet=cv::Mat::zeros(row, col, CV_32S);
	tmp=cv::Mat::zeros(row, col, CV_32S);

//#pragma omp parallel for
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col/ 2; j++){
			
			tmp.at<int>(i, j + col / 2) = (img.at<byte>(i, 2 * j) - img.at<byte>(i, 2 * j + 1)) / 2;

		}
	}
//#pragma omp parallel for
	for (int i = 0; i < row / 2; i++){
		for (int j = 0; j < col; j++){
			wavelet.at<int>(i, j) = abs(tmp.at<int>(2 * i, j) + tmp.at<int>(2 * i + 1, j)) / 2;
			
		}
	}

	mag1u=_synbihl.Feature_enhance(wavelet,row,col);

}


void Objectness::trainObjectness(int numDetPerSize)
{
	
        time_t start, finish;
	//* Learning stage I    
	generateTrianData();
	time(&start);
	trainStageI();
	time(&finish);	
        double duration1 = difftime(finish, start);
	printf("Learning stage I takes %g seconds... \n", duration1); //*/

	//* Learning stage II
	time(&start);
	trainStateII(numDetPerSize);
	time(&finish);	
        double duration2 = difftime(finish, start);
	printf("Learning stage II takes %g seconds... \n", duration2); //*/
	return;
}

void Objectness::generateTrianData()
{
	const int NUM_TRAIN = _voc.trainNum;
	const int FILTER_SZ = 8*8;
	vector<vector<Mat>> xTrainP(NUM_TRAIN), xTrainN(NUM_TRAIN);
	vector<vecI> szTrainP(NUM_TRAIN); // Corresponding size index. 
	const int NUM_NEG_BOX = 100; // Number of negative windows sampled from each image

#pragma omp parallel for
	for (int i = 0; i < NUM_TRAIN; i++)	{
		const int NUM_GT_BOX = (int)_voc.gtTrainBoxes[i].size();
		vector<Mat> &xP = xTrainP[i], &xN = xTrainN[i];
		vecI &szP = szTrainP[i];
		xP.reserve(NUM_GT_BOX*4), szP.reserve(NUM_GT_BOX*4), xN.reserve(NUM_NEG_BOX);
		Mat im3u = imread(format(_S(_voc.imgPathW), _S(_voc.trainSet[i])));

		// Get positive training data
		for (int k = 0; k < NUM_GT_BOX; k++){

			const Vec4i& bbgt =  _voc.gtTrainBoxes[i][k];
			
			vector<Vec4i> bbs; // bounding boxes;
			vecI bbR; // Bounding box ratios
			int nS = gtBndBoxSampling(bbgt, bbs, bbR);
			for (int j = 0; j < nS; j++){
				bbs[j][2] = min(bbs[j][2], im3u.cols);
				bbs[j][3] = min(bbs[j][3], im3u.rows);
				Mat mag1f = getFeature(im3u, bbs[j]), magF1f;
				flip(mag1f, magF1f, CV_FLIP_HORIZONTAL);
				xP.push_back(mag1f);
				xP.push_back(magF1f);
				szP.push_back(bbR[j]);
				szP.push_back(bbR[j]);
			}
			//}			
		}
		// Get negative training data
		for (int k = 0; k < NUM_NEG_BOX; k++){
			
			int x1 = rand() % im3u.cols + 1, x2 = rand() % im3u.cols + 1;
			int y1 = rand() % im3u.rows + 1, y2 = rand() % im3u.rows + 1;
			Vec4i bb(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2));
			if (maxIntUnion(bb, _voc.gtTrainBoxes[i]) < 0.5)
				xN.push_back(getFeature(im3u, bb));
		}
		//}
	}

	const int NUM_R = _numT * _numT + 1;
	vecI szCount(NUM_R); // Object counts of each size (combination of scale and aspect ratio) 
	int numP = 0, numN = 0, iP = 0, iN = 0;
	for (int i = 0; i < NUM_TRAIN; i++){
		numP += xTrainP[i].size();
		numN += xTrainN[i].size();
		const vecI &rP = szTrainP[i];
		for (size_t j = 0; j < rP.size(); j++)
			szCount[rP[j]]++;
	}
	vecI szActive; // Indexes of active size
	for (int r = 1; r < NUM_R; r++){
		if (szCount[r] > 50) // If only 50- positive samples at this size, ignore it.
			szActive.push_back(r-1);			
	}
	matWrite(_modelName + ".idx", Mat(szActive));

	Mat xP1f(numP, FILTER_SZ, CV_32F), xN1f(numN, FILTER_SZ, CV_32F);
	for (int i = 0; i < NUM_TRAIN; i++)	{
		vector<Mat> &xP = xTrainP[i], &xN = xTrainN[i];
		for (size_t j = 0; j < xP.size(); j++)
			memcpy(xP1f.ptr(iP++), xP[j].data, FILTER_SZ*sizeof(float));
		for (size_t j = 0; j < xN.size(); j++)
			memcpy(xN1f.ptr(iN++), xN[j].data, FILTER_SZ*sizeof(float));
	}
	CV_Assert(numP == iP && numN == iN);
	matWrite(_modelName + ".xP", xP1f);
	matWrite(_modelName + ".xN", xN1f);
}

Mat Objectness::getFeature(CMat &img3u, const Vec4i &bb)
{
	int x = bb[0] - 1, y = bb[1] - 1;
	Rect reg(x, y, bb[2] -  x, bb[3] - y);
	Mat subImg3u, mag1f, mag1u;
	resize(img3u(reg), subImg3u, Size(8*2, 8*2)); 
	
	frequencyHL(subImg3u, mag1u);
	mag1u.convertTo(mag1f, CV_32F);
	return mag1f;
}

int Objectness::gtBndBoxSampling(const Vec4i &bbgt, vector<Vec4i> &samples, vecI &bbR)
{
	double wVal = bbgt[2] - bbgt[0] + 1, hVal = (bbgt[3] - bbgt[1]) + 1;
	wVal = log(wVal)/_logBase, hVal = log(hVal)/_logBase;
	int wMin = max((int)(wVal - 0.5), _minT), wMax = min((int)(wVal + 1.5), _maxT);
	int hMin = max((int)(hVal - 0.5), _minT), hMax = min((int)(hVal + 1.5), _maxT);
	for (int h = hMin; h <= hMax; h++) for (int w = wMin; w <= wMax; w++){
		int wT = tLen(w) - 1, hT = tLen(h) - 1;
		Vec4i bb(bbgt[0], bbgt[1], bbgt[0] + wT, bbgt[1] + hT);
		if (DataSetVOC::interUnio(bb, bbgt) >= 0.5){
			samples.push_back(bb);
			bbR.push_back(sz2idx(w, h));
		}
	}
	return samples.size();
}

void Objectness::trainStateII(int numPerSz)
{
	loadTrainedModel();
	const int NUM_TRAIN = _voc.trainNum;
	vector<vecI> SZ(NUM_TRAIN), Y(NUM_TRAIN);
	vector<vecF> VAL(NUM_TRAIN);

#pragma omp parallel for
	for (int i = 0; i < _voc.trainNum; i++)	{
		const vector<Vec4i> &bbgts = _voc.gtTrainBoxes[i];
		ValStructVec<float, Vec4i> valBoxes;
		vecI &sz = SZ[i], &y = Y[i];
		vecF &val = VAL[i];
		CStr imgPath = format(_S(_voc.imgPathW), _S(_voc.trainSet[i]));
		predictBBoxSI(imread(imgPath), valBoxes, sz, numPerSz, false);                        
		const int num = valBoxes.size();
		//cout<<"num="<<num<<endl;
		CV_Assert(sz.size() == num);
		y.resize(num), val.resize(num);
		for (int j = 0; j < num; j++){
			Vec4i bb = valBoxes[j];
			val[j] = valBoxes(j);
			y[j] = maxIntUnion(bb, bbgts) >= 0.5 ? 1 : -1; 
		}
	}

	const int NUM_SZ = _svmSzIdxs.size();
	const int maxTrainNum = 100000;
	vector<vecM> rXP(NUM_SZ), rXN(NUM_SZ);
	for (int r = 0; r < NUM_SZ; r++){
		rXP[r].reserve(maxTrainNum);
		rXN[r].reserve(1000000);
	}
	for (int i = 0; i < NUM_TRAIN; i++){
		const vecI &sz = SZ[i], &y = Y[i];
		vecF &val = VAL[i];
		int num = sz.size();
		for (int j = 0; j < num; j++){
			int r = sz[j];
			CV_Assert(r >= 0 && r < NUM_SZ);
			if (y[j] == 1)
				rXP[r].push_back(Mat(1, 1, CV_32F, &val[j]));
			else 
				rXN[r].push_back(Mat(1, 1, CV_32F, &val[j]));
		}
	}

	Mat wMat(NUM_SZ, 2, CV_32F);
	for (int i = 0; i < NUM_SZ; i++){
		const vecM &xP = rXP[i], &xN = rXN[i];
		if (xP.size() < 10 || xN.size() < 10)
			printf("Warning %s:%d not enough training sample for r[%d] = %d. P = %d, N = %d\n", __FILE__, __LINE__, i, _svmSzIdxs[i], xP.size(), xN.size());	
		for (size_t k = 0; k < xP.size(); k++)
			CV_Assert(xP[k].size() == Size(1, 1) && xP[k].type() == CV_32F);

		Mat wr = trainSVM(xP, xN, L1R_L2LOSS_SVC, 100, 1);
		CV_Assert(wr.size() == Size(2, 1));
		wr.copyTo(wMat.row(i));
	}
	matWrite(_modelName + ".wS2", wMat);
	_svmReW1f = wMat;
}

void Objectness::meanStdDev(CMat &data1f, Mat &mean1f, Mat &stdDev1f)
{
	const int DIM = data1f.cols, NUM = data1f.rows;
	mean1f = Mat::zeros(1, DIM, CV_32F), stdDev1f = Mat::zeros(1, DIM, CV_32F);
	for (int i = 0; i < NUM; i++)
		mean1f += data1f.row(i);
	mean1f /= NUM;
	for (int i = 0; i < NUM; i++){
		Mat tmp;
		pow(data1f.row(i) - mean1f, 2, tmp);
		stdDev1f += tmp;
	}
	pow(stdDev1f/NUM, 0.5, stdDev1f);
}

vecD Objectness::getVector(const Mat &_t1f)
{
	Mat t1f;
	_t1f.convertTo(t1f, CV_64F);
	return (vecD)(t1f.reshape(1, 1));
}

void Objectness::illustrate()  
{
	Mat xP1f, xN1f;
	CV_Assert(matRead(_modelName + ".xP", xP1f) && matRead(_modelName + ".xN", xN1f));
	CV_Assert(xP1f.cols == xN1f.cols && xP1f.cols == 8*8 && xP1f.type() == CV_32F && xN1f.type() == CV_32F);
	Mat meanP,  meanN, stdDevP, stdDevN;
	meanStdDev(xP1f, meanP, stdDevP);
	meanStdDev(xN1f, meanN, stdDevN);
	Mat meanV(8, 8*2, CV_32F), stdDev(8, 8*2, CV_32F);
	meanP.reshape(1, 8).copyTo(meanV.colRange(0, 8));
	meanN.reshape(1, 8).copyTo(meanV.colRange(8, 8*2));
	stdDevP.reshape(1, 8).copyTo(stdDev.colRange(0, 8));
	stdDevN.reshape(1, 8).copyTo(stdDev.colRange(8, 8*2));
	normalize(meanV, meanV, 0, 255, NORM_MINMAX, CV_8U);
	CmShow::showTinyMat(_voc.resDir + "PosNeg.png", meanV);

	FILE* f = fopen(_S(_voc.resDir + "PosNeg.m"), "w"); 
	CV_Assert(f != NULL);
	fprintf(f, "figure(1);\n\n");
	PrintVector(f, getVector(meanP), "MeanP");
	PrintVector(f, getVector(meanN), "MeanN");
	PrintVector(f, getVector(stdDevP), "StdDevP");
	PrintVector(f, getVector(stdDevN), "StdDevN");
	PrintVector(f, getVector(_svmFilter), "Filter");
	fprintf(f, "hold on;\nerrorbar(MeanP, StdDevP, 'r');\nerrorbar(MeanN, StdDevN, 'g');\nhold off;");
	fclose(f);	
}

void Objectness::trainStageI()
{
	vecM pX, nX;
	pX.reserve(200000), nX.reserve(200000);
	Mat xP1f, xN1f;
	CV_Assert(matRead(_modelName + ".xP", xP1f) && matRead(_modelName + ".xN", xN1f));
	for (int r = 0; r < xP1f.rows; r++)
		pX.push_back(xP1f.row(r));
	for (int r = 0; r < xN1f.rows; r++)
		nX.push_back(xN1f.row(r));
	Mat crntW = trainSVM(pX, nX, L1R_L2LOSS_SVC, 10, 1);
	crntW = crntW.colRange(0, crntW.cols - 1).reshape(1, 8);
	CV_Assert(crntW.size() == Size(8, 8));
	matWrite(_modelName + ".wS1", crntW);
}


Mat Objectness::trainSVM(CMat &X1f, const vecI &Y, int sT, double C, double bias, double eps)
{

	parameter param; {
		param.solver_type = sT; 
		param.C = C;
		param.eps = eps; 
		param.p = 0.1;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		set_print_string_function(print_null);
		CV_Assert(X1f.rows == Y.size() && X1f.type() == CV_32F);
	}

	feature_node *x_space = NULL;
	problem prob;{
		prob.l = X1f.rows;
		prob.bias = bias;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(feature_node*, prob.l);
		const int DIM_FEA = X1f.cols;
		prob.n = DIM_FEA + (bias >= 0 ? 1 : 0);
		x_space = Malloc(feature_node, (prob.n + 1) * prob.l);
		int j = 0;
		for (int i = 0; i < prob.l; i++){
			prob.y[i] = Y[i];
			prob.x[i] = &x_space[j];
			const float* xData = X1f.ptr<float>(i);
			for (int k = 0; k < DIM_FEA; k++){
				x_space[j].index = k + 1;
				x_space[j++].value = xData[k];
			}
			if (bias >= 0){
				x_space[j].index = prob.n;
				x_space[j++].value = bias;
			}
			x_space[j++].index = -1;
		}
		CV_Assert(j == (prob.n + 1) * prob.l);
	}


	const char*  error_msg = check_parameter(&prob, &param);
	if(error_msg){
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	model *svmModel = train(&prob, &param);
	Mat wMat(1, prob.n, CV_64F, svmModel->w);
	wMat.convertTo(wMat, CV_32F);
	free_and_destroy_model(&svmModel);
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	return wMat;
}


Mat Objectness::trainSVM(const vector<Mat> &pX1f, const vector<Mat> &nX1f, int sT, double C, double bias, double eps, int maxTrainNum)
{
	vecI ind(nX1f.size());
	for (size_t i = 0; i < ind.size(); i++)
		ind[i] = i;
	int numP = pX1f.size(), feaDim = pX1f[0].cols;
	int totalSample = numP + nX1f.size();
	if (totalSample > maxTrainNum)
		random_shuffle(ind.begin(), ind.end());
	totalSample = min(totalSample, maxTrainNum);
	Mat X1f(totalSample, feaDim, CV_32F);
	vecI Y(totalSample);
	for(int i = 0; i < numP; i++){
		pX1f[i].copyTo(X1f.row(i));
		Y[i] = 1;
	}
	for (int i = numP; i < totalSample; i++){
		nX1f[ind[i - numP]].copyTo(X1f.row(i));
		Y[i] = -1;
	}
	return trainSVM(X1f, Y, sT, C, bias, eps);
}



void Objectness::softNms(std::vector<BboxWithScore>& bboxes,const int& method,const float& sigma,const float& iou_thre,const float& threshold) {
        if (bboxes.empty())
{
    return;
}

int N = bboxes.size();

float max_score,max_pos,cur_pos,weight;
BboxWithScore tmp_bbox,index_bbox;
for (int i = 0; i < N; ++i)
{
    max_score = bboxes[i].score;

    max_pos = i;
    tmp_bbox = bboxes[i];
    cur_pos = i + 1;


    while (cur_pos < (i+1000< N?i+1000:N))
    {
        if (max_score < bboxes[cur_pos].score)
        {
            max_score = bboxes[cur_pos].score;
            max_pos = cur_pos;
        }
        cur_pos ++;
    }


    bboxes[i] = bboxes[max_pos];

    
    bboxes[max_pos] = tmp_bbox;
    tmp_bbox = bboxes[i];

    cur_pos = i + 1;

    while (cur_pos < (i+1000< N?i+1000:N))
    {
        index_bbox = bboxes[cur_pos];

        
        float index_bbox_area = (float)( (index_bbox.tx-index_bbox.bx + 1) *(index_bbox.ty-index_bbox.by + 1));
        float tmp_bbox_area = (float)( (tmp_bbox.tx-tmp_bbox.bx + 1) *(tmp_bbox.ty-tmp_bbox.by + 1));
        float area = max(tmp_bbox_area,index_bbox_area); 
        float iou = calIOU_softNms(tmp_bbox,index_bbox); 
        if (iou <= 0)
        {
            cur_pos++;
            continue;
        }

         iou /= area;
        if (method == 1) 
        {
            if (iou > iou_thre) 
            {
        weight = 1 - iou;
            } else
            {
                weight = 1;
            }
        }else if (method == 2) 
        {
            weight = exp(-(iou * iou) / sigma);
        }else // original NMS
        {

            if (iou > iou_thre)
            {

        weight = 0;
            }else
            {
                weight = 1;
            }
        }
        bboxes[cur_pos].score *= weight;

        if (bboxes[cur_pos].score <= threshold) 
        {

            bboxes[cur_pos] = bboxes[N - 1];
            N --;
            cur_pos = cur_pos - 1;
        }
        cur_pos++;
    }
        }
bboxes.resize(N);
    }


float Objectness::calIOU_softNms(const BboxWithScore& bbox1,const BboxWithScore& bbox2)
{
int x1max = max(bbox1.bx,bbox2.bx);  
int x2min = min(bbox1.tx,bbox2.tx);    
int y1max = max(bbox1.by,bbox2.by);      
int y2min = min(bbox1.ty,bbox2.ty);    
float overlapWidth = x2min - x1max + 1;           
float overlapHeight = y2min - y1max + 1;  
if (overlapWidth < 0){
    return 0.;
}
if (overlapHeight < 0)
{
    return 0.;
}

return overlapWidth * overlapHeight;
 
    }






bool Traditinal_cmpScore(const BboxWithScore &lsh, const BboxWithScore &rsh) {
if (lsh.score < rsh.score)
	return true;
else
	return false;
}

void Objectness::Traditinal_NMS(std::vector<BboxWithScore>& boundingBox_, const float overlap_threshold)
{

	if (boundingBox_.empty()){
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), Traditinal_cmpScore);

	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	vector<int> vPick;
	int nPick = 0;
	multimap<float, int> vScores; 
	const int num_boxes = boundingBox_.size();

	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i){
		vScores.insert(pair<float, int>(boundingBox_[i].score, i));
	}

	while (vScores.size() > 0&&nPick<num_boxes){
		int last = vScores.rbegin()->second; 
		vPick[nPick] = last;
		nPick += 1;

		for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
			int it_idx = it->second;
			maxX = max(boundingBox_.at(it_idx).tx, boundingBox_.at(last).tx);
			maxY = max(boundingBox_.at(it_idx).ty, boundingBox_.at(last).ty);
			minX = min(boundingBox_.at(it_idx).bx, boundingBox_.at(last).bx);
			minY = min(boundingBox_.at(it_idx).by, boundingBox_.at(last).by);
				
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
				

			IOU = (maxX * maxY) / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);

			if (IOU > overlap_threshold){
				it = vScores.erase(it);  
			}
			else{
				it++;
			}
		}
	}

	vPick.resize(nPick);
	vector<BboxWithScore> tmp_;
	tmp_.resize(nPick);
	for (int i = 0; i < nPick; i++){
		tmp_[i] = boundingBox_[vPick[i]];
	}
	boundingBox_ = tmp_;

}

// Get potential bounding boxes for all test images

void Objectness::getObjBndBoxesForTests(vector<vector<Vec4i>> &_boxesTests, int numDetPerSize)
{
	const int TestNum = _voc.testSet.size();
	vecM imgs3u(TestNum);
	vector<ValStructVec<float, Vec4i>> boxesTests;
	boxesTests.resize(TestNum);
	//omp_set_nested(true);
#pragma omp parallel for
	for (int i = 0; i < TestNum; i++){
		imgs3u[i] = imread(format(_S(_voc.imgPathW), _S(_voc.testSet[i])));
		//blur(imgs3u[i], imgs3u[i], Size(3, 3));
		boxesTests[i].reserve(10000);
	}
        int average_boxsize=0;
	int scales[4] = {1, 3, 5,7};
	setmodelpath();
	//trainObjectness(numDetPerSize);
	loadTrainedModel();
       
        time_t start, finish;
        
        time(&start);

#pragma omp parallel for
	for (int i = 0; i < TestNum; i++){
		getObjBndBoxes(imgs3u[i], boxesTests[i], numDetPerSize);
                average_boxsize=average_boxsize+boxesTests[i].size();
	}

        time(&finish);	
        double duration = difftime(finish, start);
	printf("Average time for predicting an image is %gs\n",  duration/TestNum);
        cout<<"average_boxsize= "<<average_boxsize/TestNum<<endl;
	_boxesTests.resize(TestNum);
	CmFile::MkDir(_bbResDir);
	int numBoxesOut=0;

#pragma omp parallel for
	for (int i = 0; i <TestNum; i++){

		_boxesTests[i].resize(boxesTests[i].size());

		int boxesminsize=boxesTests[i].size();
		//cout<<"boxesminsize= "<<boxesminsize<<endl;
                //valBoxesout1[i].reserve(10000);
		//nonMaximumSuppression2(boxesTests[i].size(), boxesTests[i],0.9,numBoxesOut, valBoxesout1[i]);
                //valBoxesout1[i].sort(true);
                //for (int j = 0; j < valBoxesout1[i].size(); j++){

                       // _boxesTests[i][j] =  valBoxesout1[i][j];
                       //cout<<"valBoxesout1[i][j]= "<<valBoxesout1[i][j]<<endl;
		       //_boxesTests[i][j] = boxesTests[i][j];

		//}
		
                for (int j = 0; j < boxesminsize; j++){

                //        _boxesTests[i][j] =  valBoxesout1[i][j];
			_boxesTests[i][j] = boxesTests[i][j];
//
		}
		
	}

	evaluatePerImgRecall(_boxesTests, "PerImgAllNS.m", 10000);

#pragma omp parallel for
	for (int i = 0; i < TestNum; i++){
		//boxesTests1[i]= _boxesTests[i];
		CStr fName = _bbResDir + _voc.testSet[i];

		FILE *f = fopen(_S(fName + ".txt"), "w");
		fprintf(f, "%d\n", boxesTests[i].size());
		for (size_t k = 0; k < boxesTests[i].size(); k++)//{
			fprintf(f, "%g, %s\n", boxesTests[i](k), _S(strVec4i(boxesTests[i][k])));
		boxesTests[i].sort(true);
		for (int j = 0; j < boxesTests[i].size(); j++)
			_boxesTests[i][j] = boxesTests[i][j];
		fclose(f);


	}

	evaluatePerImgRecall(_boxesTests, "PerImgAllS.m", 10000);
}


// Get potential bounding boxes for all test images

void Objectness::getRandomBoxes(vector<vector<Vec4i>> &boxesTests, int num)
{
	const int TestNum = _voc.testSet.size();
	boxesTests.resize(TestNum);
#pragma omp parallel for
	for (int i = 0; i < TestNum; i++){
		Mat imgs3u = imread(format(_S(_voc.imgPathW), _S(_voc.testSet[i])));
		int H = imgs3u.cols, W = imgs3u.rows;
		boxesTests[i].reserve(num);
		for (int k = 0; k < num; k++){
			int x1 = rand()%W + 1, x2 = rand()%W + 1;
			int y1 = rand()%H + 1, y2 = rand()%H + 1;
			boxesTests[i].push_back(Vec4i(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)));
		}
	}
	evaluatePerImgRecall(boxesTests, "PerImgAll.m", num);
}
float T_IOU;
void Objectness::evaluatePerImgRecall(const vector<vector<Vec4i>> &boxesTests, CStr &saveName, const int NUM_WIN)
{
	vecD recalls(NUM_WIN);
	vecD recalls2(_voc.testSet.size());
	vecD avgScore(NUM_WIN);
	CStr fName = _bbResDir;
	
	const int TEST_NUM = _voc.testSet.size();
	for (int i = 0; i < TEST_NUM; i++){
		const vector<Vec4i> &boxesGT = _voc.gtTestBoxes[i];
		const vector<Vec4i> &boxes = boxesTests[i];  
		const int gtNumCrnt = boxesGT.size();
		vecI detected(gtNumCrnt);
		vecD score(gtNumCrnt);
		double sumDetected = 0, abo = 0;
		for (int j = 0; j < NUM_WIN; j++){
			if (j >= (int)boxes.size()){
				recalls[j] += sumDetected/gtNumCrnt;
				
				avgScore[j] += abo/gtNumCrnt;
				continue;
			}

			for (int k = 0; k < gtNumCrnt; k++)	{
				double s = DataSetVOC::interUnio(boxes[j], boxesGT[k]);
				score[k] = max(score[k], s);
				detected[k] = score[k] >= T_IOU ? 1 : 0;    
			}
			sumDetected = 0, abo = 0;
			for (int k = 0; k < gtNumCrnt; k++)	
				sumDetected += detected[k], abo += score[k];
			recalls[j] += sumDetected/gtNumCrnt;
			avgScore[j] += abo/gtNumCrnt;
			
		}
		
	}
	

        int objectproposals_i=0;
        for (int i = 0; i < TEST_NUM; i++){
		const vector<Vec4i> &boxesGT = _voc.gtTestBoxes[i];
		const vector<Vec4i> &boxes = boxesTests[i];  
		const int gtNumCrnt = boxesGT.size();

                for (int k = 0; k < gtNumCrnt; k++)	{
		        int detecteds=0;
		        double scores=0;
			for (int j = 0; j < (int)boxes.size(); j++){
				double s = DataSetVOC::interUnio(boxes[j], boxesGT[k]);
				scores = max(scores, s);
				detecteds = scores >= T_IOU ? 1 : 0;    
			}
		}                     		
	}


	for (int i = 0; i < NUM_WIN; i++){
		recalls[i] /=  TEST_NUM;
		avgScore[i] /= TEST_NUM;
	}
	
	int idx[12] = {1, 10, 100, 1000, 2000, 3000, 4000, 5000,6000,7000,8000,10000};
	for (int i = 0; i < 12; i++){
		if (idx[i] > NUM_WIN)
			continue;
		printf("%d:%.3g,%.3g\t", idx[i], recalls[idx[i] - 1], avgScore[idx[i] - 1]);
		
	}
	printf("\n");

        //plot_recall_MABO( recalls, avgScore) ;
	FILE* f = fopen(_S(_voc.resDir + saveName), "w"); 
	CV_Assert(f != NULL);
	fprintf(f, "figure(1);\n\n");
	PrintVector(f, recalls, "DR");
	PrintVector(f, avgScore, "MABO");
	fprintf(f, "semilogx(1:%d, DR(1:%d));\nhold on;\nsemilogx(1:%d, DR(1:%d));\naxis([1, 10000, 0, 1]);\nhold off;\n", NUM_WIN, NUM_WIN, NUM_WIN, NUM_WIN);
	fclose(f);	
}
//namespace plt = matplotlibcpp;
 
/*void plot_recall_MABO(vecD &recalls,vecD &avgScore) 
{
    // Prepare data.
    int n = recalls.size();
 
    vector<double> x(n);
    for(int j=0; j<n; ++j) {
        x.at(j) = j;
    }

    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 600);

  //keywords["alpha"] = "0.4";
  //keywords["color"] = "grey";
  //keywords["hatch"] = "-";
    plt::plot(x, recalls,"--k");
    plt::plot(x, avgScore,"--b");


    // Plot a red dashed line from given x and y data.
    //plt::plot(x, Ac_radar_rd_plot,"b");

    // Plot a line whose name will show up as "log(x)" in the legend.
    plt::named_plot("Recalls", x, recalls,"k");
    plt::named_plot("MABO", x,  avgScore,"b");


    // Set x-axis to interval [0,1000000]
    plt::xlim(0, n);

    // Add graph title
    plt::title("Recall and MABO");

    // Enable legend.
    plt::legend();
    
    //plt::show();

}*/
void Objectness::illuTestReults(const vector<vector<Vec4i>> &boxesTests)
{
	CStr resDir = _voc.localDir + "ResIlu/";
	CmFile::MkDir(resDir);
	const int TEST_NUM = _voc.testSet.size();
	for (int i = 0; i < TEST_NUM; i++){
		const vector<Vec4i> &boxesGT = _voc.gtTestBoxes[i];
		const vector<Vec4i> &boxes = boxesTests[i];
		const int gtNumCrnt = boxesGT.size();
		CStr imgPath = format(_S(_voc.imgPathW), _S(_voc.testSet[i]));
		CStr resNameNE = CmFile::GetNameNE(imgPath);
		Mat img = imread(imgPath);
		Mat bboxMatchImg = Mat::zeros(img.size(), CV_32F);

		vecD score(gtNumCrnt);
		vector<Vec4i> bboxMatch(gtNumCrnt);
		for (int j = 0; j < boxes.size(); j++){
			const Vec4i &bb = boxes[j];
			for (int k = 0; k < gtNumCrnt; k++)	{
				double mVal = DataSetVOC::interUnio(boxes[j], boxesGT[k]);
				if (mVal < score[k])
					continue;
				score[k] = mVal;
				bboxMatch[k] = boxes[j];
			}
		}

		for (int k = 0; k < gtNumCrnt; k++){
			const Vec4i &bb = bboxMatch[k];
			rectangle(img, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(0), 3);
			rectangle(img, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(255, 255, 255), 2);
			rectangle(img, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(0, 0, 255), 1);
		}

		imwrite(resDir + resNameNE + "_Match.jpg", img);
	}
}

void Objectness::evaluatePerClassRecall(vector<vector<Vec4i>> &boxesTests, CStr &saveName, const int WIN_NUM) 
{
	const int TEST_NUM = _voc.testSet.size(), CLS_NUM = _voc.classNames.size();
	if (boxesTests.size() != TEST_NUM){
		boxesTests.resize(TEST_NUM);
		for (int i = 0; i < TEST_NUM; i++){
			Mat boxes;
			matRead(_voc.localDir + _voc.testSet[i] + ".dat", boxes);
			Vec4i* d = (Vec4i*)boxes.data;
			boxesTests[i].resize(boxes.rows, WIN_NUM);
			memcpy(&boxesTests[i][0], boxes.data, sizeof(Vec4i)*boxes.rows);
		}
	}

	for (int i = 0; i < TEST_NUM; i++)
		if ((int)boxesTests[i].size() < WIN_NUM){
			//printf("%s.dat: %d, %d\n", _S(_voc.testSet[i]), boxesTests[i].size(), WIN_NUM);
			boxesTests[i].resize(WIN_NUM);
		}


		// #class by #win matrix for saving correct detection number and gt number
		Mat crNum1i = Mat::zeros(CLS_NUM, WIN_NUM, CV_32S);
                Mat crNum2i = Mat::zeros(CLS_NUM, WIN_NUM, CV_64F);
                vector<vecD> everyclassUnio;
                everyclassUnio.resize(CLS_NUM);
		vecD gtNums(CLS_NUM); {
			for (int i = 0; i < TEST_NUM; i++){
				const vector<Vec4i> &boxes = boxesTests[i];
				const vector<Vec4i> &boxesGT = _voc.gtTestBoxes[i];
				const vecI &clsGT = _voc.gtTestClsIdx[i];
				CV_Assert((int)boxes.size() >= WIN_NUM);
				const int gtNumCrnt = boxesGT.size();
				for (int j = 0; j < gtNumCrnt; j++){
					gtNums[clsGT[j]]++;
					double maxIntUni = 0;
					int* crNum = crNum1i.ptr<int>(clsGT[j]);
                                        double* scores = crNum2i.ptr<double>(clsGT[j]);
					for (int k = 0; k < WIN_NUM; k++) {
						double val = DataSetVOC::interUnio(boxes[k], boxesGT[j]);
						maxIntUni = max(maxIntUni, val);
						crNum[k] += maxIntUni >= 0.5 ? 1 : 0;
                                                scores[k] += maxIntUni;
                                                	
					}
                                        everyclassUnio[clsGT[j]].push_back(maxIntUni);
				}
			}
		}

		        FILE* f0 = fopen(_S(_voc.resDir + "ClassAboRecall.txt"), "w");
			CV_Assert(f0 != NULL);
			
			vecD val0(WIN_NUM),aboclass0(WIN_NUM), recallObjs0(WIN_NUM), recallClss0(WIN_NUM);
			for (int i = 0; i < WIN_NUM; i++)
				val0[i] = i;
			
			fprintf(f0, "\n");
			
			double sumObjs0 = 0;
			for (int c = 0; c < CLS_NUM; c++){
				sumObjs0 += gtNums[c];
				memset(&val0[0], 0, sizeof(double)*WIN_NUM);
				int* crNum0 = crNum1i.ptr<int>(c);
                                double* scores0 = crNum2i.ptr<double>(c);
				for (int i = 0; i < WIN_NUM; i++){
					val0[i] = crNum0[i]/(gtNums[c] + 1e-200);
					recallClss0[i] += val0[i];
					recallObjs0[i] += crNum0[i];
                                        aboclass0[i]= scores0[i]/(gtNums[c] + 1e-200);
				}
				CStr className0 = _voc.classNames[c];
				PrintVector(f0, val0, "");
                                PrintVector(f0, aboclass0, "");
                                PrintVector(f0, everyclassUnio[c], "");
                         }
                       fclose(f0);

		        FILE* f = fopen(_S(_voc.resDir + saveName), "w"); {
			CV_Assert(f != NULL);
			fprintf(f, "figure(1);\nhold on;\n\n\n");
			vecD val(WIN_NUM),aboclass(WIN_NUM), recallObjs(WIN_NUM), recallClss(WIN_NUM);
			for (int i = 0; i < WIN_NUM; i++)
				val[i] = i;
			PrintVector(f, gtNums, "GtNum");
			PrintVector(f, val, "WinNum");
			fprintf(f, "\n");
			string leglendStr("legend(");
			double sumObjs = 0;
			for (int c = 0; c < CLS_NUM; c++){
				sumObjs += gtNums[c];
				memset(&val[0], 0, sizeof(double)*WIN_NUM);
				int* crNum = crNum1i.ptr<int>(c);
                                double* scores = crNum2i.ptr<double>(c);
				for (int i = 0; i < WIN_NUM; i++){
					val[i] = crNum[i]/(gtNums[c] + 1e-200);
					recallClss[i] += val[i];
					recallObjs[i] += crNum[i];
                                        aboclass[i]= scores[i]/(gtNums[c] + 1e-200);
				}
				CStr className = _voc.classNames[c];
				PrintVector(f, val, className);
                                //PrintVector(f, aboclass, className);
				fprintf(f, "plot(WinNum, %s, %s, 'linewidth', 2);\n", _S(className), COLORs[c % CN]);
				leglendStr += format("'%s', ", _S(className));
			}
			for (int i = 0; i < WIN_NUM; i++){
				recallClss[i] /= CLS_NUM;
				recallObjs[i] /= sumObjs;
			}
			PrintVector(f, recallClss, "class");
			fprintf(f, "plot(WinNum, %s, %s, 'linewidth', 2);\n", "class", COLORs[CLS_NUM % CN]);
			leglendStr += format("'%s', ", "class");
			PrintVector(f, recallObjs, "objects");
			fprintf(f, "plot(WinNum, %s, %s, 'linewidth', 2);\n", "objects", COLORs[(CLS_NUM+1) % CN]);
			leglendStr += format("'%s', ", "objects");
			leglendStr.resize(leglendStr.size() - 2);
			leglendStr += ");";		
			fprintf(f, "%s\nhold off;\nxlabel('#WIN');\nylabel('Recall');\ngrid on;\naxis([0 %d 0 1]);\n", _S(leglendStr), WIN_NUM);
			fprintf(f, "[class([1,10,100,1000]);objects([1,10,100,1000])]\ntitle('%s')\n", _S(saveName));
			fclose(f);
			printf("%-70s\r", "");
		}
		evaluatePerImgRecall(boxesTests, CmFile::GetNameNE(saveName) + "_PerI.m", WIN_NUM);
}

void Objectness::PrintVector(FILE *f, const vecD &v, CStr &name)
{
	fprintf(f, "%s = [", name.c_str());
	for (size_t i = 0; i < v.size(); i++)
		fprintf(f, "%g ", v[i]);
	fprintf(f, "];\n");
}

// Write matrix to binary file
bool Objectness::matWrite(CStr& filename, CMat& _M){
	Mat M;
	_M.copyTo(M);
	FILE* file = fopen(_S(filename), "wb");
	if (file == NULL || M.empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, file);
	int headData[3] = {M.cols, M.rows, M.type()};
	fwrite(headData, sizeof(int), 3, file);
	fwrite(M.data, sizeof(char), M.step * M.rows, file);
	fclose(file);
	return true;
}

// Read matrix from binary file
bool Objectness::matRead(const string& filename, Mat& _M){
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file %s\n", _S(filename));
		return false;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	Mat M(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	fclose(f);
	M.copyTo(_M);
	return true;
}


float distG(float d, float delta) {return exp(-d*d/(2*delta*delta));}

Mat Objectness::aFilter(float delta, int sz)
{
	float dis = float(sz-1)/2.f;
	Mat mat(sz, sz, CV_32F);
	for (int r = 0; r < sz; r++)
		for (int c = 0; c < sz; c++)
			mat.at<float>(r, c) = distG(sqrt(sqr(r-dis)+sqr(c-dis)) - dis, delta);
	return mat;
}




