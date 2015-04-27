#ifndef WINDOW_CONFIGURATION_EMITTER_ELISPOIDE_H
#define WINDOW_CONFIGURATION_EMITTER_ELISPOIDE_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <WindowConfiguration_Data_SimpleSystem.h>
#include <WindowConfiguration_Data_SPHSystem.h>
#include <WindowConfiguration_Data_SPH2DSystem.h>
#include <WindowConfiguration_Data_CudaSystem.h>
#include <WindowConfiguration_Data_PCI_SPHSystem.h>
#include <WindowConfiguration_Data_WCSPHSystem.h>
#include <WindowConfiguration_Data_MSPHSystem.h>
#include <WindowConfiguration_Data_SWSPHSystem.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_Emitter_Elipsoide : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Emitter_Elipsoide(GLWidget *glWidget);
    ~WindowConfiguration_Emitter_Elipsoide();

    EmitterElipsoide *getEmitter();

public slots:

    void accept();
    void cancel();
    void displayElipsoide(double);
    void displayElipsoide(int);

private:
	
    EmitterElipsoide *B;

    WindowConfiguration_Data *configData;

    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;

    QDoubleSpinBox *OX, *OY, *OZ, *radius, *DX, *DY, *DZ, *VX, *VY, *VZ;
    QSpinBox *durationTime;
    QLabel *originLabel, *velocityLabel, *radiusLabel, *directionLabel, *durationTimeLabel;
    
};

#endif
