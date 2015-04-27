#include <QApplication>
#include <window.h>
#include <common.cuh>
#include <GLUT/glut.h>

int main(int argc,char **argv)
{
  printf("\nParticles Simulator Benchmarking (PSB) version 1.0.3 (Linux, 64 bits), Copyrigth (c) 2014 Emmanuelle Darles\n\n");
  if(cudaInit(argc,argv)){
        glutInit(&argc, argv);
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
	setenv ("LC_NUMERIC","POSIX",1);  
  	QApplication app(argc,argv);
  	Window *wind = new Window();
        wind->show();
        return app.exec();
   }
  else {
	printf("No CUDA Capable devices found, exiting...\n");
	return 0;
  }
}



