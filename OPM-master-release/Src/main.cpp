#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
//#include "CmShow.h"

int T_S1 ;
int T_S2 ;
int box_num;
float alpha_1;
float alpha_2;
void RunObjectness(CStr &resName, int NSS, int numPerSz,char* ad1,char* ad2);

void illutrateLoG()
{
    for (float delta = 0.5f; delta < 1.1f; delta+=0.1f){
        Mat f = Objectness::aFilter(delta, 8);
        normalize(f, f, 0, 1, NORM_MINMAX);
        //CmShow::showTinyMat(format("D=%g", delta), f);
    }
    waitKey(0);
}



int main(int argc, char* argv[])
{

    RunObjectness("WinRecall.m", 1,800,argv[1],argv[2]);


    return 0;
}

void RunObjectness(CStr &resName, int NSS, int numPerSz,char* ad1,char* ad2)
{
    srand((unsigned int)time(NULL));
    DataSetVOC voc2007("/home/jiangchao/DataSet/VOC2007/");
    voc2007.loadAnnotations();
    //voc2007.loadDataGenericOverCls();

    printf("Dataset:`%s' with %d training and %d testing\n", _S(voc2007.wkDir), voc2007.trainNum, voc2007.testNum);
    printf("%s NSS = %d, perSz = %d\n", _S(resName),NSS, numPerSz);

    Objectness objNess(voc2007,NSS);




    vector<vector<Vec4i>> boxesTests;  //modle_haarHH g   number in 61 and 63 is mostly close to 0.994
    int Algorithm_variant;
    extern float T_IOU;
    cout<<"Please Select the algorithm variant that you want to compute"<<endl;
    cout<<"OPMBQ : 1    /OPMQ : 2    /OPML : 3"<<endl;
    cin>>Algorithm_variant;
    cout<<"Please enter the overlap threshold (0.5/0.55/0.6)"<<endl;
    cin>>T_IOU;
    //sprintf(Algorithm_variant,"%d",ad1);
    //Mat query = imread(ad1);
    extern double T_background;

if (Algorithm_variant==1){//OPMBQ
    T_S1=16;//18;
    T_S2=690;//190;
    box_num=110;
    alpha_1=3.6;//5.2;
    alpha_2=2.76;
    T_background=0;
}
if (Algorithm_variant==2){//OPMQ
    T_S1=957;
    T_S2=960;
    box_num=110;
    alpha_1=5.59;
    alpha_2=3.6;
    T_background=0;
}
if (Algorithm_variant==3){//OPMQ
    T_S1=17;
    T_S2=170;
    box_num=110;
    alpha_1=5.2;
    alpha_2=2.81;
    T_background=0.0012;
}

    cout<<"box_num :"<<box_num<<" --alpha_2 :"<<alpha_2<<" --alpha_1 :"<<alpha_1<<"  --T_S1 :"<<T_S1<<"  --T_S2 :"<<T_S2<<endl;

    objNess.getObjBndBoxesForTests(boxesTests,box_num*10);

/*

     cout<<ad2<<ad1<<endl;
     char filename[50];
     sprintf(filename,"%s",ad2);
     Mat query = imread(ad1);
     objNess.loadTrainedModel2();
     ValStructVec<float, Vec4i> bboxes;
     bboxes.reserve(10000);
#pragma omp parallel for
     for (int is = 0; is < 1; is++)
     objNess.getObjBndBoxes2(query, bboxes, i*10);

     bboxes.sort(true);   
     cout<<"bboxes.size()= "<<bboxes.size()<<endl;
     FILE *f = fopen(filename, "w");
     fprintf(f, "%d\n", bboxes.size());
     for (size_t k = 0; k < bboxes.size(); k++)
         fprintf(f, "%f, %s\n", bboxes(k), _S(objNess.strVec4i(bboxes[k])));

      fclose(f);
*/
 


}
