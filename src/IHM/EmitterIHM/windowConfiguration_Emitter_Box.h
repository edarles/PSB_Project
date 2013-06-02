#ifndef WINDOW_CONFIGURATION_EMITTER_BOX_H
#define WINDOW_CONFIGURATION_EMITTER_BOX_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <WindowConfiguration_Data_SimpleSystem.h>
#include <WindowConfiguration_Data_SPHSystem.h>
#include <WindowConfiguration_Data_PCI_SPHSystem.h>
#include <WindowConfiguration_Data_CudaSystem.h>
#include <WindowConfiguration_Data_MSPHSystem.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_Emitter_Box : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Emitter_Box(GLWidget *glWidget);
    ~WindowConfiguration_Emitter_Box();

    EmitterBox *getEmitter();

public slots:

    void accept();
    void cancel();
    void displayBox(double);

private:
	
    EmitterBox *B;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    WindowConfiguration_Data *configData;

    QVBoxLayout *layout;

    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *depth, *VX, *VY, *VZ;
    QLabel *widthLabel, *lengthLabel, *depthLabel, *originLabel, *velocityLabel;
    
};

#endif
