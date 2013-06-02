
include_directories(src/Kernel/Emitters)
include_directories(src/Kernel/Emitters/Box)
include_directories(src/Kernel/Emitters/Elipsoide)
include_directories(src/Kernel/Emitters/Mesh)
include_directories(src/Kernel/Emitters/Cylinder)
include_directories(src/Kernel/Emitters/Sphere)
include_directories(src/Kernel/Emitters/Girly)

SET(EMITTERS_APP_SOURCES
			src/Kernel/Emitters/Emitters.cpp
			src/Kernel/Emitters/Emitter.cpp
			src/Kernel/Emitters/Elipsoide/EmitterElipsoide.cpp
			src/Kernel/Emitters/Box/EmitterBox.cpp
			src/Kernel/Emitters/Mesh/EmitterMesh.cpp
			src/Kernel/Emitters/Cylinder/EmitterCylinder.cpp
			src/Kernel/Emitters/Sphere/EmitterSphere.cpp
			src/Kernel/Emitters/Girly/EmitterGirly.cpp
)
