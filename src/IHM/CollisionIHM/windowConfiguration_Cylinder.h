#ifndef WINDOW_CONFIGURATION_CYLINDER_H
#define WINDOW_CONFIGURATION_CYLINDER_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <CylinderCollision.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Cylinder : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Cylinder(GLWidget *glWidget);
    ~WindowConfiguration_Cylinder();

    CylinderCollision *getCylinder();

public slots:

    void accept();
    void cancel();
    void displayCylinder(double);
    void changedIsContainer(int state);

private:
	
    CylinderCollision *C;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *OX, *OY, *OZ, *DX, *DY, *DZ, *length, *radius;
    QDoubleSpinBox *elast;
    QCheckBox *container;

    QLabel *originLabel, *directionLabel, *lengthLabel, *radiusLabel, *elastLabel, *frictionLabel;
    
};

#endif
