#ifndef WINDOW_CONFIGURATION_DATA_CUDA_SYSTEM_H
#define WINDOW_CONFIGURATION_DATA_CUDA_SYSTEM_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <WindowConfiguration_Data.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_Data_CudaSystem : public WindowConfiguration_Data
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Data_CudaSystem(QWidget*);
    ~WindowConfiguration_Data_CudaSystem();

public slots:

    void changeData(double);
    void changeData(QColor newValue);
    void setColor();

private:
	
    QDoubleSpinBox *particleRadius, *particleMass, *interactionRadius, *spring, *damping, *shear, *attraction;
    QPushButton *colorButton;
    QLabel *particleRadiusLabel, *particleMassLabel, *interactionRadiusLabel, *springLabel, *dampingLabel, *shearLabel, *attractionLabel;

    QColor color;
    
};

#endif
