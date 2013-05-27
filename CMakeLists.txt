#Déclaration du projet
# cmake . -DCMAKE_CXX_COMPILER=/usr/bin/g++-4.7 -DCMAKE_CC_COMPILER=/usr/bin/gcc-4.7
cmake_minimum_required(VERSION 2.8)
project(Particles_Simulation)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CC_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_FLAGS "-Wall -O5")

SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-5.0)
SET(CUDA_SDK_ROOT_DIR /usr/local/cuda-5.0)
SET(CUDA_INCLUDE_DIRS /usr/local/cuda-5.0/include)
SET(CUDA_CUDART_LIBRARY /usr/local/cuda-5.0/lib/libcudart.so /usr/local/cuda-5.0/lib/libcudadevrt.a /usr/local/cuda-5.0/lib/libcublas_device.a)


option ( PTXAS_OPTION "Activate --ptxas-options=-v option." OFF )
if ( PTXAS_OPTION )
   SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ptxas-options=-v" )
endif()


find_library(QTCORE_LIBRARY NAMES "QtCore" PATHS "/usr/lib/i386-linux-gnu" REQUIRED)
find_library(QTGUI_LIBRARY NAMES "QtGui" PATHS "/usr/lib/i386-linux-gnu" REQUIRED)
find_library(QGLVIEWER_QT4_LIBRARY NAMES "qglviewer-qt4" PATHS "/usr/lib/i386-linux-gnu" REQUIRED)
find_library(QT_OPENGL_LIBRARY NAMES "QtOpenGL" PATHS "/usr/lib/i386-linux-gnu" REQUIRED)
find_library(GLU_LIBRARY NAMES "GLU" PATHS "/usr/lib" REQUIRED)
find_library(GLEE_LIBRARY NAMES "GLee" PATHS "/usr/local/lib" REQUIRED)
find_library(GLEW_LIBRARY NAMES "GLEW" PATHS "/usr/lib" REQUIRED)
find_library(GLEWMX_LIBRARY NAMES "GLEWmx" PATHS "/usr/lib/" REQUIRED)
find_library(GLUT_LIBRARY NAMES "glut" PATHS "/usr/lib" REQUIRED)
find_library(CUDA_LIBRARY NAMES "cuda" PATHS "/usr/local/cuda-5.0/lib")
find_library(CUDART_LIBRARY NAMES "cudart" PATHS "/usr/local/cuda-5.0/lib" REQUIRED)
find_library(CUDADEVRT_LIBRARY NAMES "libcudadevrt.a" PATHS "/usr/local/cuda-5.0/lib" REQUIRED)
find_library(CUBLAS_LIBRARY NAMES "libcublas_device.a" PATHS "/usr/local/cuda-5.0/lib" REQUIRED)
find_library(TINYXML_LIBRARY NAMES "tinyxml" PATHS "/usr/lib" REQUIRED)

link_directories(/usr/local/lib)
link_directories(/usr/lib)
link_directories(/usr/local/cuda-5.0/lib)

include_directories(/usr/include)
include_directories(/usr/local/include)
include_directories(/usr/include/qt4)
include_directories(/usr/include/qt4/QtCore)
include_directories(/usr/include/qt4/QtGui)
include_directories(/usr/include/qt4/QtOpenGL)
include_directories(/usr/include/qt4/QtXml)
include_directories(/usr/local/cuda-5.0/include)
include_directories(/usr/local/cuda-5.0/samples/common/inc/)
include_directories(dependencies/GLee-5.4.0-src)

include_directories(src)

include_directories(src/IHM)
include_directories(src/IHM/DisplayIHM)
include_directories(src/IHM/SystemIHM)
include_directories(src/IHM/CollisionIHM)
include_directories(src/IHM/CollisionIHM/HeightField)
include_directories(src/IHM/ForceExtIHM)
include_directories(src/IHM/EmitterIHM)
include_directories(src/IHM/SimulationDataIHM)

include_directories(src/Kernel)

include_directories(src/Kernel/System)
include_directories(src/Kernel/System/SimpleSystem)
include_directories(src/Kernel/System/Particle_Interaction_System)
include_directories(src/Kernel/System/Fluid_System/SPHSystem)
include_directories(src/Kernel/System/Fluid_System/SPHSystem/SPH)
include_directories(src/Kernel/System/Fluid_System/SPHSystem/PCISPH)

include_directories(src/Kernel/ForcesExt)
include_directories(src/Kernel/ForcesExt/Constante)
include_directories(src/Kernel/ForcesExt/Constante/Gravity)
include_directories(src/Kernel/ForcesExt/Periodic)
include_directories(src/Kernel/ForcesExt/Periodic/Trochoide)

include_directories(src/Kernel/Collision)
include_directories(src/Kernel/Collision/MeshCollision)
include_directories(src/Kernel/Collision/SphereCollision)
include_directories(src/Kernel/Collision/BoxCollision)
include_directories(src/Kernel/Collision/PlaneCollision)
include_directories(src/Kernel/Collision/CylinderCollision)
include_directories(src/Kernel/Collision/HeightField)
include_directories(src/Kernel/Collision/HeightField/Linear)
include_directories(src/Kernel/Collision/HeightField/Periodic)
include_directories(src/Kernel/Collision/HeightField/Gaussian)
include_directories(src/Kernel/Collision/HeightField/Combined)

include_directories(src/Kernel/Emitters)
include_directories(src/Kernel/Emitters/Box)
include_directories(src/Kernel/Emitters/Elipsoide)
include_directories(src/Kernel/Emitters/Mesh)
include_directories(src/Kernel/Emitters/Cylinder)
include_directories(src/Kernel/Emitters/Sphere)
include_directories(src/Kernel/Emitters/Girly)

include_directories(src/Kernel/Surface)
include_directories(src/Kernel/Surface/SPH)

include_directories(src/Kernel/Extras)

include_directories(src/Utils)

include_directories(src/Utils/Math)
include_directories(src/Utils/Math/Vector)

include_directories(src/Utils/Neighborhood/UniformGrid)

include_directories(src/Utils/Geometry)
include_directories(src/Utils/Geometry/Primitives)
include_directories(src/Utils/Geometry/ComplexObjects)
include_directories(src/Utils/Geometry/ComplexObjects/HeightField)
include_directories(src/Utils/Geometry/ComplexObjects/HeightField/Linear)
include_directories(src/Utils/Geometry/ComplexObjects/HeightField/Periodic)
include_directories(src/Utils/Geometry/ComplexObjects/HeightField/Gaussian)
include_directories(src/Utils/Geometry/ComplexObjects/HeightField/Combined)

include_directories(src/Utils/Exporter/Particles)
include_directories(src/Utils/Exporter/Particles/Txt)
include_directories(src/Utils/Exporter/Particles/XML)
include_directories(src/Utils/Exporter/Particles/Mitsuba)

include_directories(src/Utils/Loader)
include_directories(src/Utils/Loader/Mesh)
include_directories(src/Utils/Loader/Mesh/OBJ)
include_directories(src/Utils/Loader/Particles)
include_directories(src/Utils/Loader/Particles/Txt)
include_directories(src/Utils/Loader/Particles/XML)
include_directories(src/Utils/Loader/SimulationData)

include_directories(src/Utils/MC)

#"-gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35")
INCLUDE(/usr/share/cmake-2.8/Modules/FindCUDA.cmake)
set(CUDA_SEPARATE_COMPILATION ON)
SET(CUDA_NVCC_FLAGS "-G -g -arch=sm_21")
#SET(CUDA_NVCC_FLAGS "-G -g -arch=sm_21 --relocatable-device-code true")
#SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options=-dc)

set(CUDA_VERBOSE_BUILD OFF )
SET(CMAKE_VERBOSE_MAKEFILE OFF) 
FIND_PACKAGE (Qt4 REQUIRED)
ADD_DEFINITIONS (${QT_DEFINITIONS})
SET(QT_USE_QTXML TRUE )
SET(QT_USE_QTOPENGL TRUE )
SET(QT_WRAP_CPP TRUE )

SET(CUDA_FILES
			src/common.cu
			src/Kernel/System/SimpleSystem/System.cu
			src/Kernel/System/SimpleSystem/ParticleKernel.cu
			src/Kernel/System/Particle_Interaction_System/CudaSystem.cu
			src/Kernel/System/Particle_Interaction_System/CudaParticle_Kernel.cu
			src/Kernel/System/Fluid_System/SPHSystem/SPH/SphSystem.cu
			src/Kernel/System/Fluid_System/SPHSystem/SPH/SphKernel.cu
			src/Kernel/System/Fluid_System/SPHSystem/PCISPH/PciSphSystem.cu
			src/Kernel/System/Fluid_System/SPHSystem/PCISPH/PciSphSystem_Kernel.cu
			src/Kernel/ForcesExt/ForceExt.cu
			src/Kernel/ForcesExt/Constante/ForceExt_Constante.cu
			src/Kernel/ForcesExt/Periodic/Trochoide/ForceExt_Trochoide.cu
			src/Kernel/Collision/MeshCollision/MeshCollision.cu
			src/Kernel/Collision/MeshCollision/MeshCollision_Kernel.cu
			src/Kernel/Collision/SphereCollision/SphereCollision.cu
			src/Kernel/Collision/SphereCollision/SphereCollision_Kernel.cu
			src/Kernel/Collision/BoxCollision/BoxCollision.cu
			src/Kernel/Collision/BoxCollision/BoxCollision_Kernel.cu
			src/Kernel/Collision/PlaneCollision/PlaneCollision.cu
			src/Kernel/Collision/PlaneCollision/PlaneCollision_Kernel.cu
			src/Kernel/Collision/CylinderCollision/CylinderCollision.cu
			src/Kernel/Collision/CylinderCollision/CylinderCollision_Kernel.cu
			src/Kernel/Collision/HeightField/Linear/LinearHeightFieldCollision.cu
			src/Kernel/Collision/HeightField/Linear/LinearHeightFieldCollision_Kernel.cu
			src/Kernel/Collision/HeightField/Periodic/PeriodicHeightFieldCollision.cu
			src/Kernel/Collision/HeightField/Periodic/PeriodicHeightFieldCollision_Kernel.cu
			src/Kernel/Collision/HeightField/Gaussian/GaussianHeightFieldCollision.cu
			src/Kernel/Collision/HeightField/Gaussian/GaussianHeightFieldCollision_Kernel.cu
			src/Kernel/Collision/HeightField/Combined/CombinedHeightFieldCollision.cu
			src/Kernel/Collision/HeightField/Combined/CombinedHeightFieldCollision_Kernel.cu
			src/Utils/Geometry/ComplexObjects/HeightField/HeightField.cu
			src/Utils/Geometry/ComplexObjects/HeightField/HeightField_Kernel.cu
			src/Utils/Geometry/ComplexObjects/HeightField/Linear/LinearHeightField.cu
			src/Utils/Geometry/ComplexObjects/HeightField/Linear/LinearHeightField_Kernel.cu
			src/Utils/Geometry/ComplexObjects/HeightField/Gaussian/GaussianHeightField.cu
			src/Utils/Geometry/ComplexObjects/HeightField/Gaussian/GaussianHeightField_Kernel.cu
			src/Utils/Geometry/ComplexObjects/HeightField/Periodic/PeriodicHeightField.cu
			src/Utils/Geometry/ComplexObjects/HeightField/Periodic/PeriodicHeightField_Kernel.cu
			src/Utils/Neighborhood/UniformGrid/UniformGrid.cu
			src/Utils/Neighborhood/UniformGrid/UniformGridKernel.cu
			src/Utils/MC/MarchingCube.cu
			src/Utils/MC/MarchingCube_Kernel.cu
)

SET(App_sources 	
			src/Kernel/System/Particle.cpp
			src/Kernel/System/System.cpp
			src/Kernel/System/SimulationDatas.cpp
			src/Kernel/System/SimulationData.cpp
			src/Kernel/System/shaders.cpp

			src/Kernel/System/SimpleSystem/SimpleSystem.cpp
			src/Kernel/System/SimpleSystem/SimulationData_SimpleSystem.cpp

			src/Kernel/System/Particle_Interaction_System/CudaParticle.cpp
			src/Kernel/System/Particle_Interaction_System/CudaSystem.cpp
			src/Kernel/System/Particle_Interaction_System/SimulationData_CudaSystem.cpp

			src/Kernel/System/Fluid_System/SPHSystem/SPH/SphParticle.cpp
			src/Kernel/System/Fluid_System/SPHSystem/SPH/SphSystem.cpp
			src/Kernel/System/Fluid_System/SPHSystem/SPH/SimulationData_SPHSystem.cpp

			src/Kernel/System/Fluid_System/SPHSystem/PCISPH/PciSphParticle.cpp
			src/Kernel/System/Fluid_System/SPHSystem/PCISPH/PciSphSystem.cpp
			src/Kernel/System/Fluid_System/SPHSystem/PCISPH/SimulationData_PCI_SPHSystem.cpp

			src/Kernel/ForcesExt/ForcesExt.cpp
			src/Kernel/ForcesExt/ForceExt.cpp
			src/Kernel/ForcesExt/Constante/ForceExt_Constante.cpp
			src/Kernel/ForcesExt/Constante/Gravity/ForceExt_Gravity.cpp
			src/Kernel/ForcesExt/Periodic/ForceExt_Periodic.cpp
			src/Kernel/ForcesExt/Periodic/Trochoide/ForceExt_Trochoide.cpp

			src/Kernel/Collision/Collision.cpp
			src/Kernel/Collision/ObjectCollision.cpp
			src/Kernel/Collision/MeshCollision/MeshCollision.cpp	
			src/Kernel/Collision/SphereCollision/SphereCollision.cpp
			src/Kernel/Collision/BoxCollision/BoxCollision.cpp
			src/Kernel/Collision/PlaneCollision/PlaneCollision.cpp
			src/Kernel/Collision/CylinderCollision/CylinderCollision.cpp
			src/Kernel/Collision/HeightField/HeightFieldCollision.cpp
			src/Kernel/Collision/HeightField/Linear/LinearHeightFieldCollision.cpp
			src/Kernel/Collision/HeightField/Periodic/PeriodicHeightFieldCollision.cpp
			src/Kernel/Collision/HeightField/Gaussian/GaussianHeightFieldCollision.cpp
			src/Kernel/Collision/HeightField/Combined/CombinedHeightFieldCollision.cpp

			src/Kernel/Emitters/Emitters.cpp
			src/Kernel/Emitters/Emitter.cpp
			src/Kernel/Emitters/Elipsoide/EmitterElipsoide.cpp
			src/Kernel/Emitters/Box/EmitterBox.cpp
			src/Kernel/Emitters/Mesh/EmitterMesh.cpp
			src/Kernel/Emitters/Cylinder/EmitterCylinder.cpp
			src/Kernel/Emitters/Sphere/EmitterSphere.cpp
			src/Kernel/Emitters/Girly/EmitterGirly.cpp

			src/Kernel/Surface/Surface.cpp
			src/Kernel/Surface/SPH/SurfaceSPH.cpp

			src/Utils/Loader/Mesh/MeshLoader.cpp
			src/Utils/Loader/Mesh/OBJ/ObjLoader.cpp
			src/Utils/Loader/Particles/ParticleLoader.cpp
			src/Utils/Loader/Particles/Txt/ParticleLoader_Txt.cpp
			src/Utils/Loader/Particles/XML/ParticleLoader_XML.cpp
			src/Utils/Loader/SimulationData/SimulationDataLoader.cpp

			src/Utils/Exporter/Particles/ParticleExporter.cpp
			src/Utils/Exporter/Particles/Txt/ParticleExporter_Txt.cpp
			src/Utils/Exporter/Particles/XML/ParticleExporter_XML.cpp
			src/Utils/Exporter/Particles/Mitsuba/ParticleExporter_Mitsuba.cpp

			src/Utils/Geometry/ObjectGeo.cpp
			src/Utils/Geometry/Primitives/Face.cpp
			src/Utils/Geometry/Primitives/Triangle.cpp
			src/Utils/Geometry/Primitives/Quadrilateral.cpp
			src/Utils/Geometry/ComplexObjects/Box.cpp
			src/Utils/Geometry/ComplexObjects/Mesh.cpp
			src/Utils/Geometry/ComplexObjects/Sphere.cpp
			src/Utils/Geometry/ComplexObjects/Cylinder.cpp
			src/Utils/Geometry/ComplexObjects/HeightField/HeightField.cpp
			src/Utils/Geometry/ComplexObjects/HeightField/Linear/LinearHeightField.cpp
			src/Utils/Geometry/ComplexObjects/HeightField/Gaussian/GaussianHeightField.cpp
			src/Utils/Geometry/ComplexObjects/HeightField/Periodic/PeriodicHeightField.cpp
			src/Utils/Geometry/ComplexObjects/HeightField/Combined/CombinedHeightField.cpp

			src/IHM/window.cpp
			src/IHM/sceneWindow.cpp
			src/IHM/SystemIHM/windowConfiguration_SimpleSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_CudaSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_SPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_PCI_SPHSystem.cpp
			src/IHM/CollisionIHM/windowConfiguration_Sphere.cpp
			src/IHM/CollisionIHM/windowConfiguration_Box.cpp
			src/IHM/CollisionIHM/windowConfiguration_Mesh.cpp
			src/IHM/CollisionIHM/windowConfiguration_Plan.cpp
			src/IHM/CollisionIHM/windowConfiguration_Cylinder.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_LinearHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_PeriodicHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_GaussianHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_CombinedHeightField.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Box.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Elipsoide.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Mesh.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Cylinder.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Sphere.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Girly.cpp
			src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Trochoide.cpp
			src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Constante.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_SimpleSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_SPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_PCI_SPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_CudaSystem.cpp
			src/IHM/glWidget.cpp
			src/IHM/DisplayIHM/glDisplay.cpp
			src/IHM/main.cpp
)

QT4_WRAP_CPP(MOC_headers src/IHM/glWidget.h
			 src/IHM/window.h
			 src/IHM/sceneWindow.h
			 src/IHM/SystemIHM/windowConfiguration_SimpleSystem.h
			 src/IHM/SystemIHM/windowConfiguration_CudaSystem.h
			 src/IHM/SystemIHM/windowConfiguration_SPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_PCI_SPHSystem.h
			 src/IHM/CollisionIHM/windowConfiguration_Sphere.h
			 src/IHM/CollisionIHM/windowConfiguration_Box.h
			 src/IHM/CollisionIHM/windowConfiguration_Mesh.h
			 src/IHM/CollisionIHM/windowConfiguration_Plan.h
			 src/IHM/CollisionIHM/windowConfiguration_Cylinder.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_LinearHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_PeriodicHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_GaussianHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_CombinedHeightField.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Box.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Elipsoide.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Mesh.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Cylinder.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Sphere.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Girly.h
			 src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Trochoide.h
			 src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Constante.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_SimpleSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_SPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_PCI_SPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_CudaSystem.h
)

set(PARTICLES_LIBRARIES ${QTCORE_LIBRARY} ${QTGUI_LIBRARY}  ${QT_OPENGL_LIBRARY} ${QGLVIEWER_QT4_LIBRARY} ${GLU_LIBRARY} ${GLEE_LIBRARY} ${GLEW_LIBRARY} ${GLEWMX_LIBRARY} ${GLUT_LIBRARY} ${CUDA_LIBRARY} ${CUDADEVRT_LIBRARY} ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${TINYXML_LIBRARY})

CUDA_COMPILE(CUDA_FILES_O ${CUDA_FILES})
CUDA_ADD_EXECUTABLE(bin/oldSPH ${App_sources} ${MOC_headers} ${CUDA_FILES_O} )
TARGET_LINK_LIBRARIES(bin/oldSPH ${PARTICLES_LIBRARIES} ${CUDA_LIBRARIES})

message ( "CUDA_VERSION_STRING: " ${CUDA_VERSION_STRING} )
message ( "CUDA_TOOLKIT_ROOT_DIR: " ${CUDA_TOOLKIT_ROOT_DIR} )
message ( "CUDA_SDK_ROOT_DIR: " ${CUDA_SDK_ROOT_DIR} )
message ( "CUDA_INCLUDE_DIRS: " "${CUDA_INCLUDE_DIRS}" )
message ( "CUDA_LIBRARIES: " "${CUDA_LIBRARIES}" )
message ( "NVCC FLAGS: " "${CUDA_NVCC_FLAGS}" )
