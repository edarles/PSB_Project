#ifndef WINDOW_CONFIGURATION_GAUSSIAN_HEIGHT_FIELD_H
#define WINDOW_CONFIGURATION_GAUSSIAN_HEIGHT_FIELD_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <GaussianHeightFieldCollision.h>

class WindowConfiguration_CombinedHeightField;
//************************************************************************/
//************************************************************************/
class WindowConfiguration_GaussianHeightField : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_GaussianHeightField(WindowConfiguration_CombinedHeightField* parent, GLWidget *glWidget);
    ~WindowConfiguration_GaussianHeightField();

    Gaussian_HeightFieldCollision *getHeightField();

public slots:

    void accept();
    void add();
    void cancel();
    void displayHeightField(double);

private:
	
    Gaussian_HeightFieldCollision *H;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *dx, *dz, *A, *p1, *p2;
    QDoubleSpinBox *elast;
    QLabel *widthLabel, *lengthLabel, *dxLabel, *dzLabel, *originLabel, *ALabel, *p1Label, *p2Label, *elastLabel;
    
    WindowConfiguration_CombinedHeightField* parent;
};

#endif
