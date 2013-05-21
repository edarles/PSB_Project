#ifndef WINDOW_CONFIGURATION_BOX_H
#define WINDOW_CONFIGURATION_BOX_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Box : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Box(GLWidget *glWidget);
    ~WindowConfiguration_Box();

    BoxCollision *getBox();

public slots:

    void accept();
    void cancel();
    void displayBox(double);
    void changedIsContainer(int state);

private:
	
    BoxCollision *B;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *depth;
    QDoubleSpinBox *elast;
    QCheckBox *container;

    QLabel *widthLabel, *lengthLabel, *depthLabel, *originLabel, *elastLabel, *frictionLabel;
    
};

#endif
