#ifndef WINDOW_CONFIGURATION_EMITTER_GIRLY_H
#define WINDOW_CONFIGURATION_EMITTER_GIRLY_H

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
class WindowConfiguration_Emitter_Girly : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Emitter_Girly(GLWidget *glWidget);
    ~WindowConfiguration_Emitter_Girly();

    EmitterGirly *getEmitter();

public slots:

    void accept();
    void cancel();
    void displayEmitter(double);

private:
	
    EmitterGirly *B;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    WindowConfiguration_Data *configData;

    QVBoxLayout *layout;

    QDoubleSpinBox *VX, *VY, *VZ;
    QLabel *velocityLabel;
    
};

#endif
