#ifndef WINDOW_CONFIGURATION_DATA_SIMPLE_SYSTEM_H
#define WINDOW_CONFIGURATION_DATA_SIMPLE_SYSTEM_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <WindowConfiguration_Data.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_Data_SimpleSystem : public WindowConfiguration_Data
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Data_SimpleSystem(QWidget*);
    ~WindowConfiguration_Data_SimpleSystem();

public slots:

    void changeData(double);
    void changeData(QColor newValue);
    void setColor();

private:
	
    QDoubleSpinBox *particleRadius, *particleMass;
    QPushButton *colorButton;
    QLabel *particleRadiusLabel, *particleMassLabel;

    QColor color;
    
};

#endif
