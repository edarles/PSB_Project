#ifndef WINDOW_CONFIGURATION_DATA_PCI_SPH_SYSTEM_H
#define WINDOW_CONFIGURATION_DATA_PCI_SPH_SYSTEM_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <WindowConfiguration_Data.h>
#include <SimulationData_PCI_SPHSystem.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Data_PCI_SPHSystem : public WindowConfiguration_Data
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Data_PCI_SPHSystem(QWidget*);
    ~WindowConfiguration_Data_PCI_SPHSystem();

public slots:

    void changeData(double);
    void changeData(QColor newValue);
    void setColor();

private:

    QDoubleSpinBox *particleRadius, *particleMass, *restDensity, *viscosity, *surfaceTension, *gasStiffness, *kernelParticles;
    QPushButton *colorButton;
    QLabel *particleRadiusLabel, *particleMassLabel, *restDensityLabel, *viscosityLabel, *surfaceTensionLabel, 
	   *gasStiffnessLabel, *kernelParticlesLabel;

    QColor color;
    
};

#endif
