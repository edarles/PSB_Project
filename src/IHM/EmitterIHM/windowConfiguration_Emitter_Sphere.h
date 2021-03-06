#ifndef WINDOW_CONFIGURATION_EMITTER_SPHERE_H
#define WINDOW_CONFIGURATION_EMITTER_SPHERE_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <WindowConfiguration_Data_SimpleSystem.h>
#include <WindowConfiguration_Data_SPHSystem.h>
#include <WindowConfiguration_Data_PCI_SPHSystem.h>
#include <WindowConfiguration_Data_CudaSystem.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_Emitter_Sphere : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Emitter_Sphere(GLWidget *glWidget);
    ~WindowConfiguration_Emitter_Sphere();

    EmitterSphere *getEmitter();

public slots:

    void accept();
    void cancel();
    void displaySphere(double);

private:
	
    EmitterSphere *B;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    WindowConfiguration_Data *configData;

    QVBoxLayout *layout;

    QDoubleSpinBox *OX, *OY, *OZ, *radius, *VX, *VY, *VZ;
    QLabel *originLabel, *radiusLabel, *velocityLabel;
    
};

#endif
