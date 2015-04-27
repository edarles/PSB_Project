
include_directories(src/IHM)
include_directories(src/IHM/DisplayIHM)
include_directories(src/IHM/SystemIHM)
include_directories(src/IHM/CollisionIHM)
include_directories(src/IHM/CollisionIHM/HeightField)
include_directories(src/IHM/ForceExtIHM)
include_directories(src/IHM/EmitterIHM)
include_directories(src/IHM/SimulationDataIHM)
include_directories(src/IHM/ExtrasIHM/HeightField)

SET(IHM_APP_SOURCES
			src/IHM/window.cpp
			src/IHM/sceneWindow.cpp
			src/IHM/glWidget.cpp
			src/IHM/DisplayIHM/glDisplay.cpp
			src/IHM/main.cpp

			src/IHM/SystemIHM/windowConfiguration_SimpleSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_CudaSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_SPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_PCI_SPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_WCSPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_MSPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_HSPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_SWSPHSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_SPH2DSystem.cpp
			src/IHM/SystemIHM/windowConfiguration_HybridSPHSystem.cpp

			src/IHM/CollisionIHM/windowConfiguration_Sphere.cpp
			src/IHM/CollisionIHM/windowConfiguration_Box.cpp
			src/IHM/CollisionIHM/windowConfiguration_Mesh.cpp
			src/IHM/CollisionIHM/windowConfiguration_Plan.cpp
			src/IHM/CollisionIHM/windowConfiguration_Cylinder.cpp

			src/IHM/CollisionIHM/HeightField/windowConfiguration_LinearHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_PeriodicHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_GaussianHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_CombinedHeightField.cpp
			src/IHM/CollisionIHM/HeightField/windowConfiguration_AnimatedPeriodicHeightField.cpp

			src/IHM/EmitterIHM/windowConfiguration_Emitter_Box.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Elipsoide.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Mesh.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Cylinder.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Sphere.cpp
			src/IHM/EmitterIHM/windowConfiguration_Emitter_Girly.cpp

			src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Trochoide.cpp
			src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Constante.cpp
			src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Soliton.cpp

			src/IHM/SimulationDataIHM/WindowConfiguration_Data.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_SimpleSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_SPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_PCI_SPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_WCSPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_CudaSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_MSPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_HSPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_SWSPHSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_SPH2DSystem.cpp
			src/IHM/SimulationDataIHM/WindowConfiguration_Data_HybridSPHSystem.cpp

			src/IHM/ExtrasIHM/HeightField/windowConfiguration_AnimatedHeightField.cpp
)

QT4_WRAP_CPP(IHM_MOC_FILES
			 src/IHM/glWidget.h
			 src/IHM/window.h
			 src/IHM/sceneWindow.h

			 src/IHM/SystemIHM/windowConfiguration_SimpleSystem.h
			 src/IHM/SystemIHM/windowConfiguration_CudaSystem.h
			 src/IHM/SystemIHM/windowConfiguration_SPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_PCI_SPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_WCSPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_MSPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_HSPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_SWSPHSystem.h
			 src/IHM/SystemIHM/windowConfiguration_SPH2DSystem.h
			 src/IHM/SystemIHM/windowConfiguration_HybridSPHSystem.h

			 src/IHM/CollisionIHM/windowConfiguration_Sphere.h
			 src/IHM/CollisionIHM/windowConfiguration_Box.h
			 src/IHM/CollisionIHM/windowConfiguration_Mesh.h
			 src/IHM/CollisionIHM/windowConfiguration_Plan.h
			 src/IHM/CollisionIHM/windowConfiguration_Cylinder.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_LinearHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_PeriodicHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_GaussianHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_CombinedHeightField.h
			 src/IHM/CollisionIHM/HeightField/windowConfiguration_AnimatedPeriodicHeightField.h

			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Box.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Elipsoide.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Mesh.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Cylinder.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Sphere.h
			 src/IHM/EmitterIHM/windowConfiguration_Emitter_Girly.h

			 src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Trochoide.h
			 src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Constante.h
			 src/IHM/ForceExtIHM/windowConfiguration_ForceExt_Soliton.h

			 src/IHM/SimulationDataIHM/WindowConfiguration_Data.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_SimpleSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_SPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_PCI_SPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_WCSPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_CudaSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_MSPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_HSPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_SWSPHSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_SPH2DSystem.h
			 src/IHM/SimulationDataIHM/WindowConfiguration_Data_HybridSPHSystem.h

			 src/IHM/ExtrasIHM/HeightField/windowConfiguration_AnimatedHeightField.h
)
