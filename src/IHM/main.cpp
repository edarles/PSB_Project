#include <QApplication>
#include <window.h>
#include <common.cuh>

int main(int argc,char **argv)
{
  if(cudaInit(argc,argv)){
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
  	QApplication app(argc,argv);
  	Window wind;
        wind.show();
        return app.exec();
   }
  else {
	printf("No CUDA Capable devices found, exiting...\n");
	return 0;
  }
}



