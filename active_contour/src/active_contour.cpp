//============================================================================
// Name        : active_contour.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
/*
#include <iostream>
using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}
*/

//============================================================================
// Name        : opencv_test.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
/*
#include <iostream>
using namespace std;

int main() {
	cout << "!!!Hello Crueal World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}
*/

//cvErode(img, img, NULL, f2);
// TrainingTools.cpp
//

//g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o prz  cvsnakeimage.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching




//#include "stdafx.h"
#include <iostream>
#include <string.h>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <fstream>
//opencv2/legacy/legacy.hpp﻿
#include "opencv2/legacy/legacy.hpp"


/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*///
//#include "precomp.hpp"


#define _CV_SNAKE_BIG 2.e+38f
#define _CV_SNAKE_IMAGE 1
#define _CV_SNAKE_GRAD  2
//#define CV_IMPL CV_EXTERN_C

#ifndef IPPI_CALL
#  define IPPI_CALL(func) CV_Assert((func) >= 0)
#endif

//#include "precomp.hpp"
/* IPP-compatible return codes */

#define  CV_VALUE  1
#define  CV_ARRAY  2


#define _CV_SNAKE_BIG 2.e+38f
#define _CV_SNAKE_IMAGE 1
#define _CV_SNAKE_GRAD  2


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:      icvSnake8uC1R
//    Purpose:
//    Context:
//    Parameters:
//               src - source image,
//               srcStep - its step in bytes,
//               roi - size of ROI,
//               pt - pointer to snake points array
//               n - size of points array,
//               alpha - pointer to coefficient of continuity energy,
//               beta - pointer to coefficient of curvature energy,
//               gamma - pointer to coefficient of image energy,
//               coeffUsage - if CV_VALUE - alpha, beta, gamma point to single value
//                            if CV_MATAY - point to arrays
//               criteria - termination criteria.
//               scheme - image energy scheme
//                         if _CV_SNAKE_IMAGE - image intensity is energy
//                         if _CV_SNAKE_GRAD  - magnitude of gradient is energy
//    Returns:
//F*/

static CvStatus
icvSnake8uC1R( unsigned char *src,
               int srcStep,
               CvSize roi,
               CvPoint * pt,
               int n,
               float *alpha,
               float *beta,
               float *gamma,
               int coeffUsage,
               CvSize win,
               CvTermCriteria criteria,
               int scheme )
{
    int i, j, k;
    int neighbors = win.height * win.width;

    int centerx = win.width >> 1;	// przesuniecie bitowe to tak jak podzielic przez 2 (db do wyzn. zgbr. polowy dlugosci)
    int centery = win.height >> 1;

    float invn;
    int iteration = 0;
    int converged = 0;


    float *Econt;
    float *Ecurv;
    float *Eimg;
    float *E;

    float _alpha, _beta, _gamma;

    /*#ifdef GRAD_SNAKE */
    float *gradient = NULL;
    float *gradient2 = NULL;

    //IplImage* myIplImage2 = cvCreateImageHeader(cvSize(roi.width, roi.height), IPL_DEPTH_32F, 1);

    uchar *map = NULL;
    int map_width = ((roi.width - 1) >> 3) + 1; 			// x>>3 == x/2^3
    int map_height = ((roi.height - 1) >> 3) + 1;
    #define WTILE_SIZE 8
    #define TILE_SIZE (WTILE_SIZE + 2)
    short dx[TILE_SIZE*TILE_SIZE], dy[TILE_SIZE*TILE_SIZE];
    CvMat _dx = cvMat( TILE_SIZE, TILE_SIZE, CV_16SC1, dx );
    CvMat _dy = cvMat( TILE_SIZE, TILE_SIZE, CV_16SC1, dy );
    CvMat _src = cvMat( roi.height, roi.width, CV_8UC1, src );
    cv::Ptr<cv::FilterEngine> pX, pY;

    /* inner buffer of convolution process */
    //char ConvBuffer[400];

    /*#endif */


    /* check bad arguments */
    if( src == NULL )
        return CV_NULLPTR_ERR;
    if( (roi.height <= 0) || (roi.width <= 0) )
        return CV_BADSIZE_ERR;
    if( srcStep < roi.width )
        return CV_BADSIZE_ERR;
    if( pt == NULL )
        return CV_NULLPTR_ERR;
    if( n < 3 )
        return CV_BADSIZE_ERR;
    if( alpha == NULL )
        return CV_NULLPTR_ERR;
    if( beta == NULL )
        return CV_NULLPTR_ERR;
    if( gamma == NULL )
        return CV_NULLPTR_ERR;
    if( coeffUsage != CV_VALUE && coeffUsage != CV_ARRAY )
        return CV_BADFLAG_ERR;
    if( (win.height <= 0) || (!(win.height & 1)))
        return CV_BADSIZE_ERR;
    if( (win.width <= 0) || (!(win.width & 1)))
        return CV_BADSIZE_ERR;

    invn = 1 / ((float) n);

    if( scheme == _CV_SNAKE_GRAD )
    {
        pX = cv::createDerivFilter( CV_8U, CV_16S, 1, 0, 3, cv::BORDER_REPLICATE );
        pY = cv::createDerivFilter( CV_8U, CV_16S, 0, 1, 3, cv::BORDER_REPLICATE );
        gradient = (float *) cvAlloc( roi.height * roi.width * sizeof( float ));

        gradient2 = (float *) cvAlloc( roi.height * roi.width * sizeof( float ));
        //gradient2 = (float *) cvAlloc( 100 * 150 * sizeof( float ));

        /*

        IplImage* ipl_image_p = cvCreateImageHeader(cvSize(150, 100), IPL_DEPTH_32F, 1);
        int ix = 1, iy = 1;

        for (; iy < 100 * 150 *2; iy++)
        {

			gradient2[iy] = 1;

			ipl_image_p->imageData = (char *)gradient2;
			ipl_image_p->imageDataOrigin = ipl_image_p->imageData;
			//cvSetData(myIplImage3, gradient2, 150);

			cvShowImage( "myIplImage_d", ipl_image_p );
			cvWaitKey(1);
        }
        */

        map = (uchar *) cvAlloc( map_width * map_height );
        /* clear map - no gradient computed */
        memset( (void *) map, 0, map_width * map_height );
    }
    Econt = (float *) cvAlloc( neighbors * sizeof( float ));
    Ecurv = (float *) cvAlloc( neighbors * sizeof( float ));
    Eimg = (float *) cvAlloc( neighbors * sizeof( float ));
    E = (float *) cvAlloc( neighbors * sizeof( float ));

    while( !converged )
    {
        float ave_d = 0;
        int moved = 0;

        converged = 0;
        iteration++;
        /* compute average distance */ 								// average distance between adjacent/neighboring points *----*--*-------*-----* -> *----.*
        for( i = 1; i < n; i++ )
        {
            int diffx = pt[i - 1].x - pt[i].x; // gradient
            int diffy = pt[i - 1].y - pt[i].y;

            ave_d += cvSqrt( (float) (diffx * diffx + diffy * diffy) ); //
        }
        ave_d += cvSqrt( (float) ((pt[0].x - pt[n - 1].x) * (pt[0].x - pt[n - 1].x) +
                                  (pt[0].y - pt[n - 1].y) * (pt[0].y - pt[n - 1].y)));

        ave_d *= invn; // 13 pixels ?
        /* average distance computed */

        //for each nodule point/knot calculate the energies
        for( i = 0; i < n; i++ )
        {


            /* Calculate Econt */
            float maxEcont = 0;
            float maxEcurv = 0;
            float maxEimg = 0;
            float minEcont = _CV_SNAKE_BIG;
            float minEcurv = _CV_SNAKE_BIG;
            float minEimg = _CV_SNAKE_BIG;
            float Emin = _CV_SNAKE_BIG;

            int offsetx = 0;
            int offsety = 0;
            float tmp;

            // compute bounds to prevent exiting the image boundaries (check if pt.x is smaller than win.width/2) (something like setting the ROI)
            /*|---|**************
             *|***|**************
             *|***|**************
             *|---|**************
             ********|----|******
             ********|****|******
             ********|****|******
             ********|****|******
             ********|****|******
             ********|----|******
             ********************
             */
            /* compute bounds */
            int left = MIN( pt[i].x, win.width >> 1 );
            int right = MIN( roi.width - 1 - pt[i].x, win.width >> 1 );
            int upper = MIN( pt[i].y, win.height >> 1 );
            int bottom = MIN( roi.height - 1 - pt[i].y, win.height >> 1 );

            //maxEcont = 0;
            //minEcont = _CV_SNAKE_BIG;
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    int diffx, diffy;
                    float energy;

                    if( i == 0 )
                    {
                        //diffx = pt[n - 1].x - (pt[i].x + k);
                        //diffy = pt[n - 1].y - (pt[i].y + j);
                        diffx = pt[i].x - (pt[i+1].x + k);
                        diffy = pt[i].y - (pt[i+1].y + j);
                    }
                    /**/
                    else if( i == n - 1 )
                    {
                        diffx = pt[i].x - (pt[i-1].x + k);
                        diffy = pt[i].y - (pt[i-1].y + j);
                    }

                    else
                    {
                        //diffx = pt[i - 1].x - (pt[i].x + k);
                        //diffy = pt[i - 1].y - (pt[i].y + j);
                        diffx = pt[i].x - (pt[i + 1].x + k);
                        diffy = pt[i].y - (pt[i + 1].y + j);
                    }

                    //printf("co to za wartosc indeksu Ecount? %d", (j + centery) * win.width + k + centerx); 	// fabs   f abs?

                    Econt[(j + centery) * win.width + k + centerx] =
                    	energy =
                        (float) fabs( ave_d - cvSqrt( (float)(diffx * diffx + diffy * diffy) ));
                    // roznica miedzy srednim dystansem a kazdym miedzy punktowym osobno
                    // jesli pkt sa rownoodlegle to energia = 0;


                    maxEcont = MAX( maxEcont, energy );
                    minEcont = MIN( minEcont, energy );
                }
            }

            /// 		//  normalizacja
            tmp = maxEcont - minEcont;
            tmp = (tmp == 0) ? 0 : (1 / tmp);
            for( k = 0; k < neighbors; k++ )
            {
                Econt[k] = (Econt[k] - minEcont) * tmp;
            }







            /*  Calculate Ecurv */
            maxEcurv = 0;
            minEcurv = _CV_SNAKE_BIG;
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    int tx, ty;
                    float energy;

                    if( i == 0 )
                    {
                        tx = pt[n - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[n - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    else if( i == n - 1 )
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[0].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[0].y;
                    }
                    else
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    Ecurv[(j + centery) * win.width + k + centerx] =
                    	energy =
                    	(float)  cvSqrt( (float)(tx * tx + ty * ty) );
                        //(float) (tx * tx + ty * ty);

                    maxEcurv = MAX( maxEcurv, energy );
                    minEcurv = MIN( minEcurv, energy );
                }
            }
            tmp = maxEcurv - minEcurv;
            tmp = (tmp == 0) ? 0 : (1 / tmp);
            for( k = 0; k < neighbors; k++ )
            {
                Ecurv[k] = (Ecurv[k] - minEcurv) * tmp;
            }









            /* Calculate Eimg */
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    float energy;

                    if( scheme == _CV_SNAKE_GRAD )
                    {
                        /* look at map and check status */
                        int x = (pt[i].x + k)/WTILE_SIZE;
                        int y = (pt[i].y + j)/WTILE_SIZE;
                        //printf("x = %d, y = %d\n", x, y);

                        if( map[y * map_width + x] == 0 )
                        {
                            int l, m;

                            /* evaluate block location */
                            int upshift = y ? 1 : 0;
                            int leftshift = x ? 1 : 0;
                            int bottomshift = MIN( 1, roi.height - (y + 1)*WTILE_SIZE );
                            int rightshift = MIN( 1, roi.width - (x + 1)*WTILE_SIZE );
                            CvRect g_roi = { x*WTILE_SIZE - leftshift, y*WTILE_SIZE - upshift,
                                leftshift + WTILE_SIZE + rightshift, upshift + WTILE_SIZE + bottomshift };
                            CvMat _src1;
                            cvGetSubArr( &_src, &_src1, g_roi );

                            cv::Mat _src_ = cv::cvarrToMat(&_src1);
                            cv::Mat _dx_ = cv::cvarrToMat(&_dx);
                            cv::Mat _dy_ = cv::cvarrToMat(&_dy);

                            pX->apply( _src_, _dx_, cv::Rect(0,0,-1,-1), cv::Point(), true );
                            pY->apply( _src_, _dy_, cv::Rect(0,0,-1,-1), cv::Point(), true );

                            for( l = 0; l < WTILE_SIZE + bottomshift; l++ )
                            {
                                for( m = 0; m < WTILE_SIZE + rightshift; m++ )
                                {

                                    gradient[(y*WTILE_SIZE + l) * roi.width + x*WTILE_SIZE + m] =
/*
										(float) (abs(dx[(l + upshift) * TILE_SIZE + m + leftshift]) +
												 abs(dy[(l + upshift) * TILE_SIZE + m + leftshift])
												   )/50;
*/
                                        (float) (dx[(l + upshift) * TILE_SIZE + m + leftshift] *
                                                 dx[(l + upshift) * TILE_SIZE + m + leftshift] +
                                                 dy[(l + upshift) * TILE_SIZE + m + leftshift] *
                                                 dy[(l + upshift) * TILE_SIZE + m + leftshift]);          		  //gradient[(y * WTILE_SIZE + l) * (roi.width) + x * WTILE_SIZE + m] = (float) (dx[(l + upshift) * TILE_SIZE + m + leftshift] * dx[(l + upshift) * TILE_SIZE + m + leftshift] +dy[(l + upshift) * TILE_SIZE + m + leftshift] * dy[(l + upshift) * TILE_SIZE + m + leftshift]);
                                    if( scheme == _CV_SNAKE_GRAD )
                                    {
gradient2[(y*WTILE_SIZE + l) * roi.width + x*WTILE_SIZE + m] = gradient[(y*WTILE_SIZE + l) * roi.width + x*WTILE_SIZE + m] / 5000;
                                    }
                                }
                            }
                            map[y * map_width + x] = 1;
                        }
                        Eimg[(j + centery) * win.width + k + centerx] = energy =
                            gradient[(pt[i].y + j) * roi.width + pt[i].x + k];

                    }
                    else
                    {
                        Eimg[(j + centery) * win.width + k + centerx] = energy =
                            src[(pt[i].y + j) * srcStep + pt[i].x + k];
                    }

                    maxEimg = MAX( maxEimg, energy );
                    minEimg = MIN( minEimg, energy );
                }
            }

            tmp = (maxEimg - minEimg);
            tmp = (tmp == 0) ? 0 : (1 / tmp);

            for( k = 0; k < neighbors; k++ )
            {
                Eimg[k] = (minEimg - Eimg[k]) * tmp;
            }


            //cvShowImage("gradient", gradient);
            //cvWaitKey();

            //IplImage* image2 = cvCreateImage(win, 8, 1);
            //memcpy( image2->imageData, src, win.width*3);
            //cvNamedWindow( "from_the_inside", 1 );
            //cvShowImage( "from_the_inside", image2 );
            //cvReleaseImage(&image2);



            /* locate coefficients */
            if( coeffUsage == CV_VALUE)
            {
                _alpha = *alpha;
                _beta = *beta;
                _gamma = *gamma;
            }
            else
            {
                _alpha = alpha[i];
                _beta = beta[i];
                _gamma = gamma[i];
            }



            /* Find Minimize point in the neighbors */
            for( k = 0; k < neighbors; k++ )
            {
                E[k] = _alpha * Econt[k] + _beta * Ecurv[k] + _gamma * Eimg[k];
            }

            Emin = _CV_SNAKE_BIG;
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    if( E[(j + centery) * win.width + k + centerx] < Emin )
                    {
                        Emin = E[(j + centery) * win.width + k + centerx];
                        offsetx = k;
                        offsety = j;
                    }
                }
            }

            if( offsetx || offsety )
            {
                pt[i].x += offsetx;
                pt[i].y += offsety;
                moved++;
            }
        }

        converged = (moved == 0);
        if( (criteria.type & CV_TERMCRIT_ITER) && (iteration >= criteria.max_iter) )
            converged = 1;
        if( (criteria.type & CV_TERMCRIT_EPS) && (moved <= criteria.epsilon) )
            converged = 1;
    }




    cvFree( &Econt );
    cvFree( &Ecurv );
    cvFree( &Eimg );
    cvFree( &E );

    if( scheme == _CV_SNAKE_GRAD )
    {
        IplImage* myIplImage = cvCreateImageHeader(cvSize(roi.width, roi.height), IPL_DEPTH_32F, 1);

        myIplImage->imageData = (char *)gradient2;
        myIplImage->imageDataOrigin = myIplImage->imageData;



        //cvSetData(myIplImage, gradient2, roi.width);
        //cvSetData(myIplImage, dx, roi.width);
        			///cvSaveImage("test_img.png", myIplImage);
        //cvSetData(myIplImage, src, roi.width);
        cvShowImage( "myIplImage_d", myIplImage );

        //cvShowImage( "myIplImage_dx", _dx_ );

        //IplImage* ipl_image_p = cvCreateImageHeader(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 1);
        //ipl_image_p->imageData = gradient;
        //ipl_image_p->imageDataOrigin = ipl_image_p->imageData;
        //cvShowImage( "ipl_image_p", ipl_image_p );


        cvFree( &gradient );
        cvFree( &gradient2 );
        cvReleaseImageHeader(&myIplImage);
        cvFree( &map );
    }
    return CV_OK;
}


static CvStatus
icvSnake8uC1R_grad( unsigned char *src,
               int srcStep,
               CvSize roi,
               CvPoint * pt,
               int n,
               float *alpha,
               float *beta,
               float *gamma,
               int coeffUsage,
               CvSize win, CvTermCriteria criteria, int scheme )
{
    int i, j, k;
    int neighbors = win.height * win.width;

    int centerx = win.width >> 1;	// przesuniecie bitowe to tak jak podzielic przez 2 (db do wyzn. zgbr. polowy dlugosci)
    int centery = win.height >> 1;

    float invn;
    int iteration = 0;
    int converged = 0;


    float *Econt;
    float *Ecurv;
    float *Eimg;
    float *E;

    float _alpha, _beta, _gamma;

    /*#ifdef GRAD_SNAKE */
    float *gradient = NULL;
    float *gradient2 = NULL;
    float *energy_vis = NULL;

    //IplImage* myIplImage2 = cvCreateImageHeader(cvSize(roi.width, roi.height), IPL_DEPTH_32F, 1);

    uchar *map = NULL;
    int map_width = ((roi.width - 1) >> 3) + 1; 			// x>>3 == x/2^3
    int map_height = ((roi.height - 1) >> 3) + 1;
    #define WTILE_SIZE 8
    #define TILE_SIZE (WTILE_SIZE + 2)
    short dx[TILE_SIZE*TILE_SIZE], dy[TILE_SIZE*TILE_SIZE];
    CvMat _dx = cvMat( TILE_SIZE, TILE_SIZE, CV_16SC1, dx );
    CvMat _dy = cvMat( TILE_SIZE, TILE_SIZE, CV_16SC1, dy );
    CvMat _src = cvMat( roi.height, roi.width, CV_8UC1, src );
    cv::Ptr<cv::FilterEngine> pX, pY;

    /* inner buffer of convolution process */
    //char ConvBuffer[400];

    /*#endif */


    /* check bad arguments */
    if( src == NULL )
        return CV_NULLPTR_ERR;
    if( (roi.height <= 0) || (roi.width <= 0) )
        return CV_BADSIZE_ERR;
    if( srcStep < roi.width )
        return CV_BADSIZE_ERR;
    if( pt == NULL )
        return CV_NULLPTR_ERR;
    if( n < 3 )
        return CV_BADSIZE_ERR;
    if( alpha == NULL )
        return CV_NULLPTR_ERR;
    if( beta == NULL )
        return CV_NULLPTR_ERR;
    if( gamma == NULL )
        return CV_NULLPTR_ERR;
    if( coeffUsage != CV_VALUE && coeffUsage != CV_ARRAY )
        return CV_BADFLAG_ERR;
    if( (win.height <= 0) || (!(win.height & 1)))
        return CV_BADSIZE_ERR;
    if( (win.width <= 0) || (!(win.width & 1)))
        return CV_BADSIZE_ERR;

    invn = 1 / ((float) n);

    if( scheme == _CV_SNAKE_GRAD )
    {
        pX = cv::createDerivFilter( CV_8U, CV_16S, 1, 0, 3, cv::BORDER_REPLICATE );
        pY = cv::createDerivFilter( CV_8U, CV_16S, 0, 1, 3, cv::BORDER_REPLICATE );
        gradient = (float *) cvAlloc( roi.height * roi.width * sizeof( float ));

        gradient2 = (float *) cvAlloc( roi.height * roi.width * sizeof( float ));
        energy_vis = (float *) cvAlloc( roi.height * roi.width * sizeof( float ));

        //gradient2 = (float *) cvAlloc( 100 * 150 * sizeof( float ));

        /*

        IplImage* ipl_image_p = cvCreateImageHeader(cvSize(150, 100), IPL_DEPTH_32F, 1);
        int ix = 1, iy = 1;

        for (; iy < 100 * 150 *2; iy++)
        {

			gradient2[iy] = 1;

			ipl_image_p->imageData = (char *)gradient2;
			ipl_image_p->imageDataOrigin = ipl_image_p->imageData;
			//cvSetData(myIplImage3, gradient2, 150);

			cvShowImage( "myIplImage_d", ipl_image_p );
			cvWaitKey(1);
        }
        */

        map = (uchar *) cvAlloc( map_width * map_height );
        /* clear map - no gradient computed */
        //memset( (void *) map, 0, map_width * map_height );
        memset( (void *) map, NULL, map_width * map_height );
        // Sets the first (map_width * map_height) bytes of the block of memory
        // pointed by (map) to the specified value (interpreted as an unsigned char).

    }
    Econt = (float *) cvAlloc( neighbors * sizeof( float ));
    Ecurv = (float *) cvAlloc( neighbors * sizeof( float ));
    Eimg = (float *) cvAlloc( neighbors * sizeof( float ));
    E = (float *) cvAlloc( neighbors * sizeof( float ));

    while( !converged )
    {
        float ave_d = 0;
        int moved = 0;

        converged = 0;
        iteration++;
        /* compute average distance */ 								// average distance between adjacent/neighboring points *----*--*-------*-----* -> *----.*
        for( i = 1; i < n; i++ )
        {
            int diffx = pt[i - 1].x - pt[i].x;
            int diffy = pt[i - 1].y - pt[i].y;

            ave_d += cvSqrt( (float) (diffx * diffx + diffy * diffy) ); //
        }
        ave_d += cvSqrt( (float) ((pt[0].x - pt[n - 1].x) * (pt[0].x - pt[n - 1].x) +
                                  (pt[0].y - pt[n - 1].y) * (pt[0].y - pt[n - 1].y)));

        ave_d *= invn; // 13 pix
        /* average distance computed */

        //for each nodule point/knot calculate the energies
        for( i = 0; i < n; i++ )
        {
            /* Calculate Econt */
            float maxEcont = 0;
            float maxEcurv = 0;
            float maxEimg = 0;
            float minEcont = _CV_SNAKE_BIG;
            float minEcurv = _CV_SNAKE_BIG;
            float minEimg = _CV_SNAKE_BIG;
            float Emin = _CV_SNAKE_BIG;

            int offsetx = 0;
            int offsety = 0;
            float tmp;

            // compute bounds to prevent exiting the image boundaries (check if pt.x is smaller than win.width/2) (something like setting the ROI)
            /*|---|**************
             *|***|**************
             *|***|**************
             *|---|**************
             ********|----|******
             ********|****|******
             ********|****|******
             ********|****|******
             ********|****|******
             ********|----|******
             ********************
             */
            /* compute bounds */
            int left = MIN( pt[i].x, win.width >> 1 );
            int right = MIN( roi.width - 1 - pt[i].x, win.width >> 1 );
            int upper = MIN( pt[i].y, win.height >> 1 );
            int bottom = MIN( roi.height - 1 - pt[i].y, win.height >> 1 );

            //maxEcont = 0;
            //minEcont = _CV_SNAKE_BIG;
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    int diffx, diffy;
                    float energy;

                    // dla orkegu wykomentowac kod, odkomentowac komentaz
                    if( i == 0 )
                    {
                        //diffx = pt[n - 1].x - (pt[i].x + k);
                        //diffy = pt[n - 1].y - (pt[i].y + j);
                        diffx = pt[i].x - (pt[i+1].x + k);
                        diffy = pt[i].y - (pt[i+1].y + j);
                    }
                    /**/
                    else if( i == n - 1 )
                    {
                        diffx = pt[i].x - (pt[i-1].x + k);
                        diffy = pt[i].y - (pt[i-1].y + j);
                    }

                    else
                    {
                        //diffx = pt[i - 1].x - (pt[i].x + k);
                        //diffy = pt[i - 1].y - (pt[i].y + j);
                        diffx = pt[i].x - (pt[i + 1].x + k);
                        diffy = pt[i].y - (pt[i + 1].y + j);
                    }
                    //

                    //printf("wartosc indeksu Ecount? %d", (j + centery) * win.width + k + centerx); 	// fabs   f abs?

                    //Econt[(j + centery) * win.width + k + centerx] =
                    //	energy = (float) fabs( ave_d - cvSqrt( (float)(diffy * diffy + diffy * diffy) )); //
                    energy = (float) ( ave_d - cvSqrt( (float)(diffx * diffx + diffy * diffy) )); //
                    //energy_vis[(pt[i].y + j) * roi.width + pt[i].x + k] = energy/5;


                    //if(energy <= 0)
                    //	printf("p%d=%.3f, ", i, energy);

                    //if (energy < 0)
                    //	energy = energy*energy; // -energy*2;

                    Econt[(j + centery) * win.width + k + centerx] = energy;
                    // roznica miedzy srednim dystansem a kazdym miedzy punktowym osobno
                    // jesli pkt sa rownoodlegle to energia = 0;

                    maxEcont = MAX( maxEcont, energy );
                    minEcont = MIN( minEcont, energy );
                }
            }
            //printf("\n");
            /// normalizacja do przedzialu 0 - 1
            tmp = maxEcont - minEcont;
            tmp = (tmp == 0) ? 0 : (1 / tmp);
            for( k = 0; k < neighbors; k++ )
            {//printf("pr_Econt[%d] = %f\n", k, Econt[k]);
                Econt[k] = (Econt[k] - minEcont) * tmp;
                //printf("po_Econt[%d] = %f\n\n", k, Econt[k]);
            }







            /*  Calculate Ecurv */
            maxEcurv = 0;
            minEcurv = _CV_SNAKE_BIG;
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    int tx, ty;
                    float energy;

                    if( i == 0 )
                    {
                        tx = pt[i].x - 2 * (pt[i + 1].x + k) + pt[i + 2].x; //pt[n - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[i].y - 2 * (pt[i + 1].y + j) + pt[i + 2].y; //pt[n - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    else if( i == n - 1 )
                    {
                        tx = pt[i - 2].x - 2 * (pt[i - 1].x + k) + pt[i].x; //pt[i - 1].x - 2 * (pt[i].x + k) + pt[0].x;
                        ty = pt[i - 2].y - 2 * (pt[i - 1].y + j) + pt[i].y; //pt[i - 1].y - 2 * (pt[i].y + j) + pt[0].y;
                    }
                    else
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }

                    /*
                    // dla okregu
                    if( i == 0 )
                    {
                        tx = pt[n - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[n - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    else if( i == n - 1 )
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[0].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[0].y;
                    }
                    else
                    {
                        tx = pt[i - 1].x - 2 * (pt[i].x + k) + pt[i + 1].x;
                        ty = pt[i - 1].y - 2 * (pt[i].y + j) + pt[i + 1].y;
                    }
                    */

                    Ecurv[(j + centery) * win.width + k + centerx] =
                    	energy =
                    	(float)  cvSqrt( (float)(tx * tx + ty * ty) );
                        //(float) (tx * tx + ty * ty);

                    if( scheme == _CV_SNAKE_GRAD )
                    {
                    	energy_vis[(pt[i].y + j) * roi.width + pt[i].x + k] = energy/20;
                    }

                    maxEcurv = MAX( maxEcurv, energy );
                    minEcurv = MIN( minEcurv, energy );
                }
            }
            tmp = maxEcurv - minEcurv;
            tmp = (tmp == 0) ? 0 : (1 / tmp);
            for( k = 0; k < neighbors; k++ )
            {//printf("pr_Ecurv[%d] = %f\n", k, Ecurv[k]);
                Ecurv[k] = (Ecurv[k] - minEcurv) * tmp; //printf("pr_Ecurv[%d] = %f\n", k, Ecurv[k]);
            };









            /* Calculate Eimg */
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                    float energy;

                    if( scheme == _CV_SNAKE_GRAD )
                    {
                        /* look at map and check status */
                        int x = (pt[i].x + k)/WTILE_SIZE;
                        int y = (pt[i].y + j)/WTILE_SIZE;
                        //printf("x = %d, y = %d\n", x, y);

                        //printf("map[%d] = %f\n",y * map_width + x, map[y * map_width + x]);
                        if( map[y * map_width + x] == 0 )
                        {
                            int l, m;

                            /* evaluate block location */
                            int upshift = y ? 1 : 0;
                            int leftshift = x ? 1 : 0;
                            int bottomshift = MIN( 1, roi.height - (y + 1)*WTILE_SIZE );
                            int rightshift = MIN( 1, roi.width - (x + 1)*WTILE_SIZE );
                            CvRect g_roi = { x*WTILE_SIZE - leftshift, y*WTILE_SIZE - upshift,
                                leftshift + WTILE_SIZE + rightshift, upshift + WTILE_SIZE + bottomshift };
                            CvMat _src1;
                            cvGetSubArr( &_src, &_src1, g_roi );

                            cv::Mat _src_ = cv::cvarrToMat(&_src1);
                            cv::Mat _dx_ = cv::cvarrToMat(&_dx);
                            cv::Mat _dy_ = cv::cvarrToMat(&_dy);

                            pX->apply( _src_, _dx_, cv::Rect(0,0,-1,-1), cv::Point(), true );
                            pY->apply( _src_, _dy_, cv::Rect(0,0,-1,-1), cv::Point(), true );

                            for( l = 0; l < WTILE_SIZE + bottomshift; l++ )
                            {
                                for( m = 0; m < WTILE_SIZE + rightshift; m++ )
                                {

                                    gradient[(y*WTILE_SIZE + l) * roi.width + x*WTILE_SIZE + m] =
/*
										(float) (abs(dx[(l + upshift) * TILE_SIZE + m + leftshift]) +
												 abs(dy[(l + upshift) * TILE_SIZE + m + leftshift])
												   )/50;
*/
                                        (float) (dx [(l + upshift) * TILE_SIZE + m + leftshift] *
                                                 dx[(l + upshift) * TILE_SIZE + m + leftshift] +
                                                 dy[(l + upshift) * TILE_SIZE + m + leftshift] *
                                                 dy[(l + upshift) * TILE_SIZE + m + leftshift]);          		  //gradient[(y * WTILE_SIZE + l) * (roi.width) + x * WTILE_SIZE + m] = (float) (dx[(l + upshift) * TILE_SIZE + m + leftshift] * dx[(l + upshift) * TILE_SIZE + m + leftshift] +dy[(l + upshift) * TILE_SIZE + m + leftshift] * dy[(l + upshift) * TILE_SIZE + m + leftshift]);

gradient2[(y*WTILE_SIZE + l) * roi.width + x*WTILE_SIZE + m] = gradient[(y*WTILE_SIZE + l) * roi.width + x*WTILE_SIZE + m] / 5000;

                                }
                            }
                            map[y * map_width + x] = 1;
                        }
                        Eimg[(j + centery) * win.width + k + centerx] = energy =
                            gradient[(pt[i].y + j) * roi.width + pt[i].x + k];

                    }
                    else
                    {
                        Eimg[(j + centery) * win.width + k + centerx] = energy =
                            src[(pt[i].y + j) * srcStep + pt[i].x + k];
                    }

                    maxEimg = MAX( maxEimg, energy );
                    minEimg = MIN( minEimg, energy );
                }
            }

            tmp = (maxEimg - minEimg);
            tmp = (tmp == 0) ? 0 : (1 / tmp);

            for( k = 0; k < neighbors; k++ )
            {//printf("pr_Eimg[%d] = %f\n", k, Eimg[k]);
                Eimg[k] = (minEimg - Eimg[k]) * tmp; //printf("pr_Eimg[%d] = %f\n", k, Eimg[k]);
            }


            //cvShowImage("gradient", gradient);
            //cvWaitKey();

            //IplImage* image2 = cvCreateImage(win, 8, 1);
            //memcpy( image2->imageData, src, win.width*3);
            //cvNamedWindow( "from_the_inside", 1 );
            //cvShowImage( "from_the_inside", image2 );
            //cvReleaseImage(&image2);



            /* locate coefficients */
            if( coeffUsage == CV_VALUE)
            {
                _alpha = *alpha;
                _beta = *beta;
                _gamma = *gamma;
            }
            else
            {
                _alpha = alpha[i];
                _beta = beta[i];
                _gamma = gamma[i];
            }



            /* Find Minimize point in the neighbors */
            for( k = 0; k < neighbors; k++ )
            {
                E[k] = _alpha * Econt[k] + _beta * Ecurv[k] + _gamma * Eimg[k];
            }

            Emin = _CV_SNAKE_BIG;
            for( j = -upper; j <= bottom; j++ )
            {
                for( k = -left; k <= right; k++ )
                {
                	// optimize searching minimum

                    if( E[(j + centery) * win.width + k + centerx] < Emin )
                    {
                        Emin = E[(j + centery) * win.width + k + centerx];
                        offsetx = k;
                        offsety = j;
                    }
                }
            }

            if( offsetx || offsety )
            {
                pt[i].x += offsetx;
                pt[i].y += offsety;
                moved++;
            }
        }

        converged = (moved == 0);
        if( (criteria.type & CV_TERMCRIT_ITER) && (iteration >= criteria.max_iter) )
            converged = 1;
        if( (criteria.type & CV_TERMCRIT_EPS) && (moved <= criteria.epsilon) )
            converged = 1;
    }




    cvFree( &Econt );
    cvFree( &Ecurv );
    cvFree( &Eimg );
    cvFree( &E );

    if( scheme == _CV_SNAKE_GRAD )
    {
        IplImage* myIplImage = cvCreateImageHeader(cvSize(roi.width, roi.height), IPL_DEPTH_32F, 1);

        myIplImage->imageData = (char *)energy_vis;
        myIplImage->imageDataOrigin = myIplImage->imageData;



        //cvSetData(myIplImage, gradient2, roi.width);
        //cvSetData(myIplImage, dx, roi.width);
        			///cvSaveImage("test_img.png", myIplImage);
        //cvSetData(myIplImage, src, roi.width);
        cvShowImage( "myIplImage_d_energy", myIplImage );

        //cvShowImage( "myIplImage_dx", _dx_ );

        //IplImage* ipl_image_p = cvCreateImageHeader(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 1);
        //ipl_image_p->imageData = gradient;
        //ipl_image_p->imageDataOrigin = ipl_image_p->imageData;
        //cvShowImage( "ipl_image_p", ipl_image_p );


        cvFree( &gradient );
        cvFree( &gradient2 );
        cvFree( &energy_vis );

        cvReleaseImageHeader(&myIplImage);
        cvFree( &map );
    }
    return CV_OK;
}

CV_IMPL void
cvSnakeImage2( const IplImage* src, CvPoint* points,
              int length, float *alpha,
              float *beta, float *gamma,
              int coeffUsage, CvSize win,
              CvTermCriteria criteria, int calcGradient )
{
    uchar *data;
    CvSize size;
    int step;

    //for (int i = 0; i < length; i++)
    //{
    //	printf("%.3f, %.3f, %.3f \n", alpha[i], beta[i], gamma[i]);
    //}

    if( src->nChannels != 1 )
        CV_Error( CV_BadNumChannels, "input image has more than one channel" );

    if( src->depth != IPL_DEPTH_8U )
        CV_Error( CV_BadDepth, cvUnsupportedFormat );

    /////////// GRADIENT
    //uchar *grad_data;
    //CvSize grad_size;
    //int grad_step;
    //cvGetRawData( src, &grad_data, &grad_step, &grad_size );

    cvGetRawData( src, &data, &step, &size );


    //IplImage* image1 = cvCreateImage(size, 8, 1);
    //memcpy( image1->imageData, data, size.width * 3);
    //cvNamedWindow( "corners1", 1 );
    //cvShowImage( "corners1", image1 );

    IPPI_CALL( icvSnake8uC1R_grad( data, step, size, points, length,
                              alpha, beta, gamma, coeffUsage, win, criteria,
                              calcGradient ? _CV_SNAKE_GRAD : _CV_SNAKE_IMAGE ));
}

/* end of file */


void set_contour_nodules(CvPoint * point, int length) {
	int width = 40; //30;
	int height = 350; //170;

	for (int i = 0; i < length; i++)
	{
		point[i].x = width * sin(2* CV_PI * i/length/2 + CV_PI/2) + 300; //          -200; //360; //
		point[i].y = height * -cos(2* CV_PI * i/length/2 + CV_PI/2) + 258 - 110; //+50         - 140; //144; //
	}
}

int Thresholdness = 141;
int ialpha = 35;
int ibeta = 85;
int igamma = 80;
int neighbor = 27;
int ifiltr = 25;
int f2 = 25;


void set_nodule_weights(float * alpha_pts, float * beta_pts, float * gamma_pts, int length, int set) {

	int j = 0;
	int k = 0;
	int l = 0;
	alpha_pts[0] = 0;

	for (int i = 0; i < length; i++, k++)
	{
		if (set == 0) {

			alpha_pts[i] = ialpha; // i+
			beta_pts[i] = ibeta; // i+
			gamma_pts[i] = igamma; // i+igamma; //
			//printf("nd: %d = %.1f\n", i, alpha_pts[i]);
		}
		else if (set == 1) {
			//alpha_pts[i] = ialpha; // i+
			//beta_pts[i] = ibeta; // i+
			gamma_pts[i] = igamma; // i+igamma; //

			if ((i >= (length >> 1) - 2) || (i <= (length >> 1) + 2))
			{
				//beta_pts[i] += j;
				//printf("j = %d \n", j);
			}

			if (i < length>>1)
				j++;
			else if (i == length>>1)
				beta_pts[i] = 1;
			else
				j--;

			if (i == 0)
				alpha_pts[i] = j*1.5 + ialpha;
			else if (i == length >> 1)
				alpha_pts[i] = alpha_pts[i-1] + j*1.5; //j*1.5 + ialpha;


			beta_pts[i] = ibeta; //j * 1.5;
			//printf("jj = %d %.1f\n", i, alpha_pts[i]);

			//k++;
		}
	}
}

void filter( IplImage* img) {

	int msk_size = 0;

	if (f2%2 == 1)
		msk_size = f2;
	else
		msk_size = f2++;

	if (ifiltr%2 == 0)
		ifiltr++;

    cvSmooth(img, img, CV_MEDIAN, ifiltr, ifiltr);
    cvSmooth(img, img, CV_GAUSSIAN, 5, 5);
    cvErode(img, img, NULL, f2);
    //cvSmooth(image, image, CV_MEDIAN, 5, 5);
}

void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
              double threshold2, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(image), dst = cv::cvarrToMat(edges);
    CV_Assert( src.size == dst.size && src.depth() == CV_8U && dst.type() == CV_8U );

    cv::Canny(src, dst, threshold1, threshold2, aperture_size & 255,
              (aperture_size & CV_CANNY_L2_GRADIENT) != 0);
}
void cv::Canny( InputArray _src, OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size, bool L2gradient )
{
    Mat src = _src.getMat();
    CV_Assert( src.depth() == CV_8U );

    _dst.create(src.size(), CV_8U);
    Mat dst = _dst.getMat();

    if (!L2gradient && (aperture_size & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
    {
        //backward compatibility
        aperture_size &= ~CV_CANNY_L2_GRADIENT;
        L2gradient = true;
    }

    if ((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)))
        CV_Error(CV_StsBadFlag, "");

    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::canny(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
        return;
#endif

#ifdef USE_IPP_CANNY
    if( aperture_size == 3 && !L2gradient &&
        ippCanny(src, dst, (float)low_thresh, (float)high_thresh) )
        return;
#endif

    const int cn = src.channels();
    Mat dx(src.rows, src.cols, CV_16SC(cn));
    Mat dy(src.rows, src.cols, CV_16SC(cn));

    Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0) low_thresh *= low_thresh;
        if (high_thresh > 0) high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);

    ptrdiff_t mapstep = src.cols + 2;
    AutoBuffer<uchar> buffer((src.cols+2)*(src.rows+2) + cn * mapstep * 3 * sizeof(int));

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            if (!L2gradient)
            {
                for (int j = 0; j < src.cols*cn; j++)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                for (int j = 0; j < src.cols*cn; j++)
                    _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
            }

            if (cn > 1)
            {
                for(int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for(int k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);

        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j-1] && m >= _mag[j+1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
__ocv_canny_push:
            if (!prev_flag && m > high && _map[j-mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}



IplImage *image = 0 ;
IplImage *image2 = 0 ;
IplImage* image3 = 0;
IplImage* image4 = 0;

IplImage* image_grad_x = 0;  //cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
IplImage* image_grad_y = 0;  //cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
IplImage* image_grad_magn = 0;
IplImage* image_grad_magn_8u = 0;
IplImage* image_grad_magn_color = 0;

IplImage* grad_div = 0; //cvCreateImage( cvSize(image->width, image->height), image->depth, 1 );


//image = cvCreateImage(cvGetSize(image2), image2->depth, 1);

using namespace std;


int length = 40*4;
CvPoint* point = NULL; //new CvPoint[length];


//float * alpha_values = NULL;
float * alpha_values = NULL;


//cvFree( &Econt );
float * beta_values = NULL;
float * gamma_values = NULL;

int set_unique_weights = 0;

void onChange(int pos) {

	cvCopy(image3, image2);
	cvCopy(image4, image);

	int center_x = image->width/2-60; 	// 300
	int center_y = image->height/2-30;	// 258
	//if(image2) cvReleaseImage(&image2);
	//if(image) cvReleaseImage(&image);


	//CvMemStorage* storage = cvCreateMemStorage(0);

	/*
		float xx = (float)((11/1.9)*cos(a));
		float yy = (float)((11/1.1)*sin(a));
		float x = (float)(xx * cos(-CV_PI/8) + yy *sin(-CV_PI/8) +150);
		float y = (float)(xx * -sin(-CV_PI/8) + yy *cos(-CV_PI/8)+250);
	*/


	//CvPoint pt = cvPoint(0,0);

	if (point == NULL) {
		//point = new CvPoint[length];
		point = (CvPoint *) cvAlloc( length * sizeof( CvPoint ));

		//alpha_values = new float[length];
		alpha_values = (float *) cvAlloc( length * sizeof( float ));
		beta_values  = (float *) cvAlloc( length * sizeof( float ));
		gamma_values = (float *) cvAlloc( length * sizeof( float ));


		set_contour_nodules(point, length);


		set_nodule_weights(alpha_values, beta_values, gamma_values, length, set_unique_weights);

	} /*else {

		int iii = 0;
		while(iii < length) {
			//printf("x: %d y: %d \n", point[iii].x, point[iii].y);
			iii++;
		}
	}*/
	//cvReleaseMemStorage(&storage);

	float alpha=ialpha/100.0f;
	float beta=ibeta/100.0f;
	float gamma=igamma/100.0f;

	if (neighbor%2 == 0)
		neighbor++;

	filter(image);
	cvShowImage("po_filtr", image);


	cvSobel(image, image_grad_x, 1, 0, 3);
	cvSobel(image, image_grad_y, 0, 1, 3);

	/*
    for (int i = 0, k = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++, k++)
        {
            double mag = sqrt(pow(dft[k][0],2) + pow(dft[k][1], 2));
            //mag = 255*(mag/max);
            ((uchar*)(img2->imageData + i * img2->widthStep))[j] = mag;
        }
    }
    */
	cvCartToPolar(image_grad_x, image_grad_y, image_grad_magn);

	//magnitude(image_grad_x, image_grad_y, image_grad_magn);

	//cvPow(const CvArr* src, CvArr* dst, double power)
	//cvSqrt(float value)

	//IplImage* src = cvloadimage(argv[1]);
	/*this ensures you'll end up with an image of the same type as the source image. *and it also will allocate the memory for you this must be done ahead.*/
	//IplImage* dest = cvcreateimage( cvsize(src->width, src->height), src->depth, src->nchannels );
	/*we use this for the division*/
	//IplImage* div= cvcreateimage( cvsize(src->width, src->height), src->depth, src->nchannels );
	/*this sets every scalar to 2*/




	double min_val=0, max_val=0;
	CvPoint min_loc, max_loc;

	cvMinMaxLoc(image_grad_magn, &min_val, &max_val, &min_loc, &max_loc);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//cvConvertScaleAbs(image_grad_magn, image_grad_magn, 255, 0);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




	cvSet( grad_div, cvScalar(max_val, 1, 1), NULL);
	cvDiv( image_grad_magn, grad_div, image_grad_magn, 1 );
	//cvConvert(image_grad_magn, image_grad_magn);
	cvConvertScale(image_grad_magn, image_grad_magn_8u, 255);

	cvShowImage("image_grad_magn_8u", image_grad_magn_8u);
	double sigma1 = 0.5;
	double sigma2 = 0.5;
	int borderType = 0; //CV_BORDER_DEFAULT;
	CvSize ksize;
	ksize.width = 5;
	ksize.height = 5;
    double a[9]={  1.0/9.0, 1.0/9.0, 1.0/9.0,
                    1.0/9.0, 1.0/9.0, 1.0/9.0,
                    1.0/9.0, 1.0/9.0, 1.0/9.0};
    CvMat k;
    cvInitMatHeader( &k, 3, 3, CV_64FC1, a );

    cvFilter2D( image_grad_magn_8u ,image_grad_magn_8u, &k, cvPoint(-1,-1));



    //cvSmooth(image_grad_magn, image_grad_magn, CV_GAUSSIAN, 25, 25);
	cvShowImage("image_grad_magn_3x3", image_grad_magn_8u);



	cvCvtColor(image_grad_magn, image_grad_magn_color, CV_GRAY2RGB);



	cvShowImage("image_grad_magn", image_grad_magn);


	/*
	int ii = 0;
	while(ii < 3) {
		int x = point[ii].x;
		int y = point[ii].y;
		printf("x: %d y: %d \n", x, y);
		ii++;
	}
	*/
	set_nodule_weights(alpha_values, beta_values, gamma_values, length, set_unique_weights);

	int iterations1 = 50;
	int iterations2 = 50;
	int iterations3 = 50;
	int iterations4 = 50;

	CvSize size;
	size.width = 25;
	size.height = 25;
	CvTermCriteria criteria;
	criteria.type = CV_TERMCRIT_ITER;
	criteria.max_iter = 5; //000;
	criteria.epsilon = 0.1;



	/*
	for (int l = 0; l < iterations4; l++) {

		for (int k = 0; k < iterations3; k++) {

			for (int j = 0; j < iterations2; j++) {

				for (int i = 0; i < iterations1; i++) {


				}
			}
		}
	}
*/




	//cvLaplace(image, image_grad_magn, 7);
	//cvConvertScaleAbs(image_grad_magn_8u, image_grad_magn_8u, 255, 0);
	//cvShowImage("cvLaplace", image_grad_magn_8u);

	//cvSnakeImage2(image, point, length, &alpha, &beta, &gamma, 1, size, criteria, 1);

	cvSnakeImage2(image, point, length, alpha_values, beta_values, gamma_values, 1, size, criteria, 1);
	//cvSnakeImage2(image, point, length, alpha_values, beta_values, gamma_values, 1, size, criteria, 0);
	//cvSnakeImage2(image_grad_magn_8u, point, length, &alpha, &beta, &gamma, 1, size, criteria, 0);


	for(int i=0; i<length; 	i++)
	{
			int j = (i+1)%length;
			cvLine(image2, point[i], point[j],CV_RGB( 0, 255, 0), 1, 8, 0 );
			cvCircle(image2, point[i], 2, CV_RGB(250, 0, 0), -1);

			cvLine(image_grad_magn_color, point[i], point[j],CV_RGB( 0, 255, 0), 1, 8, 0 );
			cvCircle(image_grad_magn_color, point[i], 2, CV_RGB(250, 0, 0), -1);

	}

	//void cvAddWeighted(const CvArr* src1, double alpha, const CvArr* src2, double beta, double gamma, CvArr* dst)

	for(int i=0;i<length;i++)
	{
			int j = (i+1)%length;
			//cvLine ( image4, point[i], point[j], CV_RGB( 205, 205, 205), 1, 8, 0 );
			//cvCircle(image, point[i], 2, CV_RGB(250, 250, 250), -1);
	}

	cvShowImage("image2", image2);
	cvShowImage("image_grad_magn_color", image_grad_magn_color);
	//cvShowImage("image4",image);

	//delete []point;

	//cvReleaseImage(&image);
	////cvReleaseImage(&image2);
	//cvReleaseImage(&image3);
	//cvReleaseImage(&image4);
	//cvReleaseImage(&image_grad_x);
	//cvReleaseImage(&image_grad_y);
	//cvReleaseImage(&image_grad_magn);
	//cvReleaseImage(&image_grad_magn_8u);
	//cvReleaseImage(&image_grad_magn_color);
	//cvReleaseImage(&grad_div);

}

void createFilter(double gKernel[][5], int kerSize)
{
    // set standard deviation to 1.0
    double sigma = 1.0;
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    double sum = 0.0;

    // generate 5x5 kernel
    for (int x = -2; x <= 2; x++)
    {
        for(int y = -2; y <= 2; y++)
        {
            r = sqrt(x*x + y*y);
            gKernel[x + 2][y + 2] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + 2][y + 2];
        }
    }

    // normalize the Kernel
    for(int i = 0; i < 5; ++i)
        for(int j = 0; j < 5; ++j)
            gKernel[i][j] /= sum;

}

void createFilterX(double gKernel[][11], int kerSize)
{
    // set standard deviation to 1.0
    double sigma = 1.0;
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    double sum = 0.0;

    // generate 5x5 kernel
    for (int x = -5; x <= 5; x++)
    {
        for(int y = -5; y <= 5; y++)
        {
            r = sqrt(x*x + y*y);
            gKernel[x + 5][y + 5] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + 5][y + 5];
        }
    }

    // normalize the Kernel
    for(int i = 0; i < 11; ++i)
        for(int j = 0; j < 11; ++j)
            gKernel[i][j] /= sum;

}
void createFilter2(double * gKernel, int kerSize)
{
    // set standard deviation to 1.0
    double sigma = 1.0;
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    double sum = 0.0;

    int kerSizeHalf = kerSize/2;
    // generate X x X kernel
    for (int x = -kerSizeHalf; x <= kerSizeHalf; x++)
    {
        for(int y = -kerSizeHalf; y <= kerSizeHalf; y++)
        {
            r = sqrt(x*x + y*y);
            gKernel[x + kerSizeHalf + y*kerSize + kerSizeHalf*kerSize] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + kerSizeHalf + y*kerSize + kerSizeHalf*kerSize];
        }
    }

    // normalize the Kernel
    for(int i = 0; i < kerSize; ++i)
        for(int j = 0; j < kerSize; ++j)
            gKernel[i + j*kerSize] /= sum;

}
//arr[x + y*width] = 5;

int main(int argc, char* argv[]) {

/*
	int kerSize = 11;
	cout << kerSize/2 << endl;
	cout << (-11)/2 << endl;
    double gKernel[kerSize][11];
    createFilterX(gKernel, kerSize);
    double check = 0;
    for(int i = 0; i < kerSize; ++i)
    {
        for (int j = 0; j < kerSize; ++j){
        	check += gKernel[i][j];
            cout<<gKernel[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout << check << endl;

    cout << "NEWWWWWWWWWWWWWWWWWWWW" << endl;
    int kerSize2 = 11;
    double * gKernel2;
    gKernel2 = (double*)malloc(kerSize2 * sizeof(double));
    createFilter2(gKernel2, kerSize);

    for(int i = 0; i < kerSize2; ++i)
    {
        for (int j = 0; j < kerSize2; ++j){
        	check += gKernel2[i + j*kerSize2];
        	//cout << check << endl;
            cout<<gKernel2[i + j*kerSize2]<<"\t";
        }
        cout<<endl;
    }
    cout << check << endl;

*/



	const char * filename = "065.avi";

    CvCapture* capture=0;
    capture = cvCaptureFromAVI(filename); // read AVI video
    if( !capture )
        throw "Error when reading steam_avi";

    image2 = cvQueryFrame( capture );

    //test_im.png
    //image2 = cvLoadImage("test_im.png", 1);

    //gray = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
    image = cvCreateImage(cvGetSize(image2), image2->depth, 1);
    image4 = cvCreateImage(cvGetSize(image2), image2->depth, 1);
    //cvCopy(image, image4);
    cvCvtColor(image2, image, CV_RGB2GRAY);
    cvCvtColor(image2, image4, CV_RGB2GRAY);
    printf("dal");
	//const char * image_path = "img_28.png"; 		//"img_30.png";
	//image = cvLoadImage(image_path, 0);
	//image2 = cvLoadImage(image_path, 1);

    image3 = cvCreateImage(cvGetSize(image2), image2->depth, image2->nChannels);
    //image3 = cvCreateImage(cvSize(image2.cols, image2.rows), IPL_DEPTH_8U, 1);
    //iplImage2->imageData = (char *) imgn.data;
    cvCopy(image2, image3);
	image3 = cvCloneImage(image2);

    //cvCopy(image, image4);
	//image4 = cvCloneImage(image);

	//cvShowImage("image1", image);



	image_grad_x = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F, 1);
	image_grad_y = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F, 1);
	image_grad_magn = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	image_grad_magn_8u = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	image_grad_magn_color = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 3);


	//div = cvCreateImage( cvSize(image->width, image->height), image->depth, image->nChannels );
	grad_div = cvCreateImage( cvGetSize(image), image->depth, image->nChannels );



	printf("hi ]p[]';lL:""|\n");
	cvNamedWindow("win1",0);
	cvCreateTrackbar("Thd", "win1", &Thresholdness, 255, onChange);
	cvCreateTrackbar("alpha_cont", "win1", &ialpha, 100, onChange);
	cvCreateTrackbar("beta_curv", "win1", &ibeta, 100, onChange);
	cvCreateTrackbar("gamma_img", "win1", &igamma, 100, onChange);
	cvCreateTrackbar("neigh", "win1", &neighbor, 250, onChange);
	cvCreateTrackbar("filter", "win1", &ifiltr, 70, onChange);
	cvCreateTrackbar("f2", "win1", &f2, 70, onChange);
	cvResizeWindow("win1", 300, 500);

	onChange(1);

	int keypress = -1;
	int global_ster = 0;
	int wait_time = 0; //10;

	int temporary = 0;

	while (1)
	{
		keypress = cvWaitKey(wait_time);
			if(keypress == 27 || keypress == 'q')
				break;
			if(keypress == 'k') // 107
			{
				while (temporary != 's')
				{
					;
					//global_ster = 1;
					temporary = cvWaitKey(20);
					onChange(1);
				}
				temporary = 0;
			}

			if(keypress == 's')
				global_ster = 0;
			/**/

			if(keypress == 114) {
				printf("r\n");
				set_contour_nodules(point, length);
				/*
				for (int i = 0; i < length; i++)
				{
						//point[i]=pt;
						point[i].x = 20 * sin(2* CV_PI * i/length/2) + 300;
						point[i].y = 130 * cos(2* CV_PI * i/length/2) + 258;
				}
				*/
				onChange(1);
			}
			if(keypress == 105 || global_ster == 1)
				onChange(1);

			if(keypress == 'z')
				printf("wait_time = %d\n", wait_time += 5);

			if(keypress == 'x')
				printf("wait_time = %d\n", wait_time -= 5);

			if(keypress == 'n') {
			    image2 = cvQueryFrame( capture );
				if(!image2){
					printf("error queryFrame capture");
					break;
				}
			    cvCvtColor(image2, image, CV_RGB2GRAY);

			    cvCopy(image2, image3);
			    cvCopy(image, image4);
				//image3 = cvCloneImage(image2); //////////////////////////////////////////////////////////////////
			    /////////////////////////////////
			    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			    //////////////////////////////////////////////////////////////////
			    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			    //////////////////////////////////////////////////////////////////
			    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//cvShowImage("w", image);
				onChange(1);
			}

			cvShowImage("win1",image2);

	}
	cvReleaseCapture( &capture );

	//delete alpha_values;
	cvFree( &alpha_values );
	cvFree( &beta_values );
	cvFree( &gamma_values );
	printf("free\n");
	cvFree( &point );
	//delete point;
	printf("free2\n");
	return 0;
}

