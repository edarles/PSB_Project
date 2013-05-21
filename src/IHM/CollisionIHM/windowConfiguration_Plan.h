#ifndef WINDOW_CONFIGURATION_PLAN_H_
#define WINDOW_CONFIGURATION_PLAN_H_

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_Plan : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Plan(GLWidget *glWidget);
    ~WindowConfiguration_Plan();

    PlaneCollision *getPlan();

public slots:

    void accept();
    void cancel();
    void displayPlan(double);

private:
	
    PlaneCollision *B;

    GLWidget    *glWidget;    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;
    QTabWidget  *onglets;

    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *depth, *DX, *DY, *DZ;
    QDoubleSpinBox *elast, *friction;
    QCheckBox *container;

    QLabel *directionLabel, *widthLabel, *lengthLabel, *depthLabel, *originLabel, *elastLabel, *frictionLabel;
    
};

#endif
