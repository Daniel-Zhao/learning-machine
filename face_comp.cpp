/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: minglu@openailab.comm
 *
 */
#include <string>
#include <sys/time.h>
#include "mtcnn.hpp"
#include "lightcnn.hpp"
#include "mtcnn_utils.hpp"


/* calculate cosine distance of two vectors *///计算相似度
float cosine_dist(float* vectorA, float* vectorB, int size)
{
    float Numerator=0;
    float Denominator1=0;
    float Denominator2=0;
    float Similarity;
    for (int i = 0 ; i < size ; i++)
    {
        Numerator += (vectorA[i] * vectorB[i]);
        Denominator1 += (vectorA[i] * vectorA[i]);
        Denominator2 += (vectorB[i] * vectorB[i]);
    }

    Similarity = Numerator/sqrt(Denominator1)/sqrt(Denominator2);

    return Similarity;
}

/* get face image with face detection results */
void get_face_image(fcv::Mat& img,std::vector<face_box>& face_list, fcv::Mat& faceImg)
{
    float boxSize=0;
    float maxWidth=0;
    int maxI = 0;
    float faceWidth = 0;
    fcv::Size faceSize(FACEWIDTH,FACEHEIGHT);


    /* Select face with largest size */
    for ( unsigned int i = 0; i < face_list.size(); i++)
    {
        face_box &box = face_list[i];
        boxSize = box.x1 - box.x0;
        if (boxSize>maxWidth)
        {
            maxWidth = boxSize;
            maxI = i;
        }
    }
    face_box &box = face_list[maxI];

    /* calculate face width in oringal image */
    faceWidth = (box.landmark.x[1] - box.landmark.x[0])/0.53194925;

    /* calculate face roi for image cropping */
    fcv::Rect roi;
    roi.x = box.landmark.x[0]-(0.224152*faceWidth);
    roi.y = (box.landmark.y[0]+box.landmark.y[1])/2-(0.2119465*faceWidth);
    roi.width = faceWidth;
    roi.height = faceWidth;

    std::cout << "FaceWidth=" << faceWidth << endl;

    /* crop face image */
    fcv::Mat cropA = img(roi);

    /* resize image for lcnn input */
    fcv::resize(cropA,faceImg,faceSize);

}




int main(int argc,char** argv)
{

    /*default image pair for comparing*/
    std::string imgA_name = "./images/George_W_Bush_0009.jpg";
    std::string imgB_name = "./images/George_W_Bush_0031.jpg";

    std::string saveA_name = "./faceA.jpg";
    std::string saveB_name = "./faceB.jpg";

    /* Variable definition */
    fcv::Mat faceImage;

    float *featureA;
    float *featureB;
    int ret;

    std::string model_dir = "./models/";
    std::vector<face_box> face_info;

    if(argc<=2)
    {
        std::cout<<"[usage]: "<<argv[0]<<" <imageA.jpg> <imageB.jpg> <model_dir>\n";
    }
    if(argc >=3 )
    {
        imgA_name=argv[1];
        imgB_name=argv[2];
    }
    if(argc >=4 ) model_dir=argv[2];


    fcv::Mat imageA = fcv::imread(imgA_name);
    if (imageA.empty())
    {
        std::cerr<<"fcv::imread "<<imgA_name<<" failed\n";
        return -1;
    }

    fcv::Mat imageB = fcv::imread(imgB_name);
    if (imageA.empty())
    {
        std::cerr<<"fcv::imread "<<imgB_name<<" failed\n";
        return -1;
    }

    /* Tengine -- initialization *///修改
    init_tengine_library();
    if(request_tengine_version("0.1")<0)
    {
        release_tengine_library();
        return -2;
    }
    printf("Tengine version: %s\n", get_tengine_version());

    /* MTCNN -- default value *///修改
    int min_size=60;
    float conf_p=0.6;
    float conf_r=0.7;
    float conf_o=0.8;
    float nms_p=0.5;
    float nms_r=0.7;
    float nms_o=0.7;

    /* MTCNN -- initialization *///修改
    mtcnn* det = new mtcnn(min_size,conf_p,conf_r,conf_o,nms_p,nms_r,nms_o);   
    /* MTCNN -- load models *///修改
     ret = detector.load_3model(model_dir);
	 if(ret != 0)
	 {
		 std::cout << "can not load mtcnn models." << endl;
		 release_tengine_library();
		 return -1;
	 }

    /* LightCNN -- initialization *///修改
	lightcnn faceFeature;
    ret = faceFeature.init(model_dir);
    if(ret <0)
    {
        std::cout << "can not initialize light cnn."<< endl;
        release_tengine_library();
        return -1;
    }
    struct timeval t0, t1;
    float timeAlgo;

    
    
    /* MTCNN -- detect faces in image */
	gettimeofday(&t0, NULL);
	detector.detect(imageA, face_info);
	
    gettimeofday(&t1, NULL);
    timeAlgo = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout <<  "Time on face detection(image A) is " << timeAlgo << "ms" << endl;

    if (face_info.size()==0)
    {
        std::cout <<  "Can not detect face in " << imgA_name << endl;
        release_tengine_library();
        return -2;
    }

    /* Get face from original image */
	get_face_image(imageA, face_info, faceImage);
    fcv::imwrite(saveA_name, faceImage);
    /* extract feature from face image with light cnn*/
    featureA = (float *)malloc(sizeof(float)*FEATURESIZE);
    gettimeofday(&t0, NULL);
    faceFeature.featureExtract(faceImage,featureA);
    gettimeofday(&t1, NULL);
    timeAlgo = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout <<  "Time on feature extraction (image A) is " << timeAlgo << "ms" << endl;

    /* MTCNN -- detect faces in image */
    gettimeofday(&t0, NULL);
    detector.detect(imageB, face_info);
    gettimeofday(&t1, NULL);

    timeAlgo = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout <<  "Time on face detection(image B) is " << timeAlgo << "ms" << endl;

    if (face_info.size()==0)
    {
        std::cout <<  "Can not detect face in " << imgB_name << endl;
        release_tengine_library();
        free(featureA);
        return -2;
    }

    /* Get face from original image */
    get_face_image(imageB, face_info, faceImage);

    fcv::imwrite(saveB_name, faceImage);

    /* extract feature from face image with light cnn*/
    featureB = (float *)malloc(sizeof(float)*FEATURESIZE);
    gettimeofday(&t0, NULL);
    faceFeature.featureExtract(faceImage,featureB);
    gettimeofday(&t1, NULL);
    timeAlgo = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    std::cout <<  "Time on feature extraction (image B) is " << timeAlgo << "ms" << endl;


    /* Tengine -- deinitialization *///修改
    release_tengine_library();


    /* calculte similarity of two face features */
    float similarity = cosine_dist(featureA, featureB, FEATURESIZE);
    std::cout << "Similarity: " << similarity << endl;


    free(featureA);
    free(featureB);


    return  0;
}
