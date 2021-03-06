#ifndef WINDOW_CONFIGURATION_LINEAR_HEIGHT_FIELD_H
#define WINDOW_CONFIGURATION_LINEAR_HEIGHT_FIELD_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <LinearHeightFieldCollision.h>

class WindowConfiguration_CombinedHeightField;
//************************************************************************/
//************************************************************************/
class WindowConfiguration_LinearHeightField : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_LinearHeightField(WindowConfiguration_CombinedHeightField*, GLWidget *glWidget);
    ~WindowConfiguration_LinearHeightField();

    Linear_HeightFieldCollision *getHeightField();

public slots:

    void accept();
    void cancel();
    void displayHeightField(double);
    void add();

private:
	
    Linear_HeightFieldCollision *H;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QVBoxLayout *layout;
    QTabWidget *onglets;

    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *dx, *dz, *a, *b;
    QDoubleSpinBox *elast;
    QLabel *widthLabel, *lengthLabel, *dxLabel, *dzLabel, *originLabel, *aLabel, *bLabel, *elastLabel;

    WindowConfiguration_CombinedHeightField* parent;
    
};

#endif
