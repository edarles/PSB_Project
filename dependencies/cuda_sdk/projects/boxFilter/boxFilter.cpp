/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

/*
    Image box filtering example

    This sample uses CUDA to perform a simple box filter on an image
    and uses OpenGL to display the results.

    It processes rows and columns of the image in parallel.

    The box filter is implemented such that it has a constant cost,
    regardless of the filter width.

    Press '=' to increment the filter radius, '-' to decrease it

    Version 1.1 - modified to process 8-bit RGBA images
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

#define MIN_EPSILON_ERROR 1e-3f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "lena_10.ppm",
    "lena_14.ppm",
    "lena_18.ppm",
    "lena_22.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_10.ppm",
    "ref_14.ppm",
    "ref_18.ppm",
    "ref_22.ppm",
    NULL
};

char *image_filename = "lena.ppm";
int iterations = 1;
int filter_radius = 10;
int nthreads = 32; //64;

unsigned int width, height;
unsigned int * h_img = NULL;
unsigned int * d_img = NULL;
unsigned int * d_temp = NULL;
//cudaArray* d_array, *d_tempArray;

GLuint pbo;     // OpenGL pixel buffer object
GLuint texid;   // texture
GLuint shader;

unsigned int timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)


extern "C" void computeGold(float *id, float *od, int w, int h, int n);

// These are CUDA functions to handle allocation and launching the kernels
extern "C" void initTexture(int width, int height, void *pImage);
extern "C" void freeTextures();

extern "C" void boxFilter(float *d_src, float *d_temp, float *d_dest, int width, int height,
                          int radius, int iterations, int nthreads);

extern "C" void boxFilterRGBA(unsigned int *d_src, unsigned int *d_temp, unsigned int *d_dest, 
                              int width, int height, int radius, int iterations, int nthreads);

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "AutoTest: CUDA Box Filter (radius=%d)", filter_radius );
	    glutSetWindowTitle(temp);

        if (filter_radius >= 10) g_Index++;
        filter_radius += 4;

		if (filter_radius > 22) {
			printf("Summary: %d errors!\n", g_TotalErrors);
			printf("Test %s!\n", (g_TotalErrors==0) ? "PASSED" : "FAILED");
			exit(0);
		}
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "%sCUDA Box Filter (radius=%d): %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), filter_radius, ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        CUT_SAFE_CALL(cutResetTimer(timer));  

        AutoQATest();
    }
}

// display results using OpenGL
void display()
{
    CUT_SAFE_CALL(cutStartTimer(timer));  

    // execute filter, writing results to pbo
    unsigned int *d_result;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_result, pbo));
    boxFilterRGBA(d_img, d_temp, d_result, width, height, filter_radius, iterations, nthreads);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // load texture from pbo
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glVertex2f(0, 0); glTexCoord2f(0, 0);
    glVertex2f(0, 1); glTexCoord2f(1, 0);
    glVertex2f(1, 1); glTexCoord2f(1, 1);
    glVertex2f(1, 0); glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        printf("> (Frame %d) Readback BackBuffer\n", frameCount);
        g_CheckRender->readback( width, height, NULL );
        g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);
        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MIN_EPSILON_ERROR)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }
    glutSwapBuffers();

    CUT_SAFE_CALL(cutStopTimer(timer));  

    computeFPS();
}

// display results using OpenGL FBO's
void displayFBO()
{
    CUT_SAFE_CALL(cutStartTimer(timer));  

    // execute filter, writing results to pbo
    unsigned int *d_result;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_result, pbo));
    boxFilterRGBA(d_img, d_temp, d_result, width, height, filter_radius, iterations, nthreads);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

	// Calling the FBO RenderPath
	{
		if (g_CheckRender && g_CheckRender->IsFBO()) {
			// bind to the FrameBuffer Object
			g_CheckRender->bindRenderPath();
		}

		// display results
		glClear(GL_COLOR_BUFFER_BIT);

		// load texture from pbo
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// fragment program is required to display floating point texture
		glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
		glEnable(GL_FRAGMENT_PROGRAM_ARB);
		glDisable(GL_DEPTH_TEST);

		glBegin(GL_QUADS);
		glVertex2f(0, 0); glTexCoord2f(0, 0);
		glVertex2f(0, 1); glTexCoord2f(1, 0);
		glVertex2f(1, 1); glTexCoord2f(1, 1);
		glVertex2f(1, 0); glTexCoord2f(0, 1);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_FRAGMENT_PROGRAM_ARB);

		if (g_CheckRender && g_CheckRender->IsFBO()) {
			g_CheckRender->unbindRenderPath();
		}
	}

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        printf("> (Frame %d) readback BackBuffer\n", frameCount);
        g_CheckRender->readback( width, height, NULL );
        g_CheckRender->savePPM ( sOriginal[g_Index], true, NULL);
        if (!g_CheckRender->PPMvsPPM( sOriginal[g_Index], sReference[g_Index], MIN_EPSILON_ERROR )) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }

	// rebind the FBO and now just render a quad to screen
	{
        if (g_CheckRender) g_CheckRender->bindTexture();
		glGenerateMipmapEXT( GL_TEXTURE_2D );
		if (g_CheckRender) g_CheckRender->unbindTexture();

		// now render to the full screen using this texture
		glClearColor(0.2, 0.2, 0.2, 0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw textured quad
        if (g_CheckRender) {
		    g_CheckRender->bindFragmentProgram();
		    g_CheckRender->bindTexture();
        }

		glBegin(GL_QUADS);
		{
			glVertex2f(0, 0); glTexCoord2f(0, 0);
			glVertex2f(0, 1); glTexCoord2f(1, 0);
			glVertex2f(1, 1); glTexCoord2f(1, 1);
			glVertex2f(1, 0); glTexCoord2f(0, 1);
		}
		glEnd();

		if (g_CheckRender) g_CheckRender->unbindTexture();
	}
	glutSwapBuffers();

    CUT_SAFE_CALL(cutStopTimer(timer));  

    computeFPS();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case '=':
        case '+':
            filter_radius++;
            break;
        case '-':
            if (filter_radius > 1) filter_radius--;
            break;
        case ']':
            iterations++;
            break;
        case '[':
            if (iterations>1) iterations--;
            break;
        default:
            break;
    }
    printf("radius = %d, iterations = %d\n", filter_radius, iterations);
    glutPostRedisplay();
}

void idle()
{
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void initCuda()
{
    // allocate device memory
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_img,  (width * height * sizeof(unsigned int)) ));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_temp, (width * height * sizeof(unsigned int)) ));

    // Refer to boxFilter_kernel.cu for implementation
    initTexture(width, height, h_img); 

    CUT_SAFE_CALL( cutCreateTimer( &timer));
}

void cleanup()
{
    CUT_SAFE_CALL( cutDeleteTimer( timer));
//    free(h_img);

    CUDA_SAFE_CALL(cudaFree(d_img));
    CUDA_SAFE_CALL(cudaFree(d_temp));

    // Refer to boxFilter_kernel.cu for implementation
    freeTextures();

	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);
    glDeleteTextures(1, &texid);
    glDeleteProgramsARB(1, &shader);

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
}

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initOpenGL()
{
    // create pixel buffer object
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, h_img, GL_STREAM_DRAW_ARB);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // warm-up
    boxFilterRGBA(d_img, d_temp, d_temp, width, height, filter_radius, iterations, nthreads);

    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStartTimer( timer));

    // execute the kernel
    boxFilterRGBA(d_img, d_temp, d_img, width, height, filter_radius, iterations, nthreads);

    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Error: boxFilterRGBA Kernel execution failed!");

    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer( timer));

    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
	printf("%.2f Mpixels/sec\n", (width*height / (cutGetTimerValue( timer) / 1000.0f)) / 1e6);

    // allocate mem for the result on host side
    unsigned int size = width * height * sizeof(float);
    float* h_odata = (float*) malloc(size);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_img, size, cudaMemcpyDeviceToHost) );

    CUT_SAFE_CALL( cutSavePGMf( "data/output.pgm", h_odata, width, height));

    // compute reference solution
    float *h_odata_ref = (float*) malloc(size);
    CUT_SAFE_CALL( cutStartTimer( timer));
//    computeGold(h_img, h_odata_ref, width, height, filter_radius);
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf("CPU Processing time: %f (ms)\n", cutGetTimerValue( timer));

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        CUT_SAFE_CALL( cutSavePGMf( "data/regression.pgm", h_odata, width, height));
        printf("Wrote 'regression.pgm'\n");
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res = cutComparefe( h_odata, h_odata_ref, width*height, MIN_EPSILON_ERROR );
        printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    free(h_odata);
    free(h_odata_ref);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    CUT_DEVICE_INIT(argc, argv);

    bool bQAReadback = false;
    bool bFBODisplay = false;

    if (argc > 1) {
        cutGetCmdLineArgumenti( argc, (const char**) argv, "threads", &nthreads );
        cutGetCmdLineArgumenti( argc, (const char**) argv, "radius", &filter_radius);
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest")) {
            bQAReadback = true;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "fbo")) {
            bFBODisplay = true;
        }
    }

    // load image
    char* image_path = cutFindFilePath(image_filename, argv[0]);
    if (image_path == 0) {
        fprintf(stderr, "Error finding image file '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }

    CUT_SAFE_CALL( cutLoadPPM4ub(image_path, (unsigned char **) &h_img, &width, &height));
    if (!h_img) {
        printf("Error opening file '%s'\n", image_path);
        exit(-1);
    }
    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);

    // initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Box Filter");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
	if (bFBODisplay) {
        if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 2.0\n");
            fprintf(stderr, "  GL_ARB_fragment_program\n");
            fprintf(stderr, "  GL_EXT_framebuffer_object\n");
            exit(-1);
        }
	} else {
		if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
			fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
			fprintf(stderr, "This sample requires:\n");
			fprintf(stderr, "  OpenGL version 1.5\n");
			fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
			fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
			exit(-1);
		}
	}
    initCuda();
    initOpenGL();

    if (bQAReadback) {
        if (bFBODisplay) {
            glutDisplayFunc(displayFBO);
            g_CheckRender = new CheckFBO(width, height, 4);
        } else {
            g_CheckRender = new CheckBackBuffer(width, height, 4);
        }
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }

    printf("Press '+' and '-' to change filter width\n"
           "Press ']' and '[' to change number of iterations\n");
    fflush(stdout);

    atexit(cleanup);

//    runTest( argc, argv);
    glutMainLoop();
    CUT_EXIT(argc, argv);
}
