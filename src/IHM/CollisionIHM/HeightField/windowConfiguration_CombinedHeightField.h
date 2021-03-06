#ifndef WINDOW_CONFIGURATION_COMBINED_HEIGHT_FIELD_H
#define WINDOW_CONFIGURATION_COMBINED_HEIGHT_FIELD_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <vector>

#include <CombinedHeightFieldCollision.h>

using namespace std;
//************************************************************************/
//************************************************************************/
class WindowConfiguration_CombinedHeightField : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_CombinedHeightField(GLWidget *glWidget);
    ~WindowConfiguration_CombinedHeightField();

    Combined_HeightFieldCollision *getHeightField();

public slots:

    void accept();
    void cancel();
    void displayHeightField(double);

    void addLinearHeightField();
    void addGaussianHeightField();
    void addPeriodicHeightField();

public:
	
    Combined_HeightFieldCollision *H;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;
    
    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *dx, *dz;
    QDoubleSpinBox *elast;
    QLabel *widthLabel, *lengthLabel, *dxLabel, *dzLabel, *originLabel, *elastLabel;
    
    QPushButton *addLinear;
    QPushButton *addGaussian;
    QPushButton *addPeriodic;

    vector<HeightField*> Hfields;
};

#endif
