
include_directories(src/Extras/AnimatedHeightField)
include_directories(src/Extras/AnimatedHeightField/Periodic)

SET(EXTRAS_CUDA_FILES
			src/Extras/AnimatedHeightField/Periodic/AnimatedPeriodicHeightField.cu
			src/Extras/AnimatedHeightField/Periodic/AnimatedPeriodicHeightField_Kernel.cu
)

SET(EXTRAS_APP_SOURCES
			src/Extras/AnimatedHeightField/AnimatedHeightField.cpp
			src/Extras/AnimatedHeightField/Periodic/AnimatedPeriodicHeightField.cpp
)
