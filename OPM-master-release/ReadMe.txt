###################################################################
#                                                                 #
#       OPM:A Fast and High Performance Object Proposals          #
#    and Bounding Box Merge: Application to Object Detection      #
#                           version 1.0                           #
#                                                                 #
###################################################################

The software was developed under ubuntu14.04.
- The code has been test on 64-bit ubuntu14.04 and 64-bit ubuntu16.04 systems respectively.

1. Introduction:

OPM has a minimal computational complexity and good objectness measure-
ment ability. On Ubuntu 14.04 system and 2.20 GHz Intel Xeon e5-2630 V4 CPU, 
the average computing time of OPM for each image in voc2007 test set 
and mscoco2017 validation data set is 0.0015s and 0.0022s respectively, 
which is the most efficient method of all current methods.  

--------------------
2. License:

Copyright (C) 2019 * * 

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. 
--------------------

3. This source code is free for academic usage under LICENSE.

--------------------
4. Code Style:

- [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

--------------------
5. How to run the code:

- cd "yourpath"/OPM-master/Src
- change the path in CMakeLists.txt: target_link_libraries(${PROJECT_NAME}  opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs ${EXTERNAL_LIBS} OPM LibLinear "yourpath"/OPM-master/Src/libMatplot.a "yourpath"/OPM-master/Src/libSynthesize.so)
- Modify opencv version and path in CMakeLists.txt
- change the path in main.cpp: DataSetVOC voc2007("YOUR_PATH_TO_THE_DATASET");
- cmake -DCMAKE_BUILD_TYPE=Release ./
- make
- ./OPM_linux
- Select the algorithm variant that you want to compute. For example, select variant OPMBQ and enter the number 1.
- Enter the overlap threshold (0.5/0.55/0.6)

--------------------

6. Acknowledgement:

We refer the program shown in the link https://github.com/tfzhou/BINGObjectness

--------------------

7. Support:

Contact jiangchao0516@126.com. 
