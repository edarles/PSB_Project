#ifndef WINDOW_CONFIGURATION_SPHERE_H
#define WINDOW_CONFIGURATION_SPHERE_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <SphereCollision.h>
#include <glWidget.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Sphere : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Sphere(GLWidget *glWidget);
    ~WindowConfiguration_Sphere();

    SphereCollision *getSphere();

public slots:

    void accept();
    void cancel();
    void displaySphere(double);
    void changedIsContainer(int state);

private:
	
    SphereCollision *S;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *radius, *OX, *OY, *OZ;
    QDoubleSpinBox *elast, *friction;
    QCheckBox *container;

    QLabel *radiusLabel, *originLabel, *elastLabel, *frictionLabel;
    
};

#endif
