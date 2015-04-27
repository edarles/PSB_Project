#ifndef WINDOW_CONFIGURATION_MSPH_SYSTEM_H
#define WINDOW_CONFIGURATION_MSPH_SYSTEM_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ForceExt_Gravity.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_MSPHSystem : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_MSPHSystem(GLWidget *glWidget);
    ~WindowConfiguration_MSPHSystem();

    QSize sizeHint() const;

public slots:

    void accept();
    void cancel();
    void changeData(double);
    void changeGravityData(double);
    void changeMinS(double);
    void changeMaxS(double);

private:

    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QDoubleSpinBox *gravity, *deltaTime, *MinSX, *MinSY, *MinSZ, *MaxSX, *MaxSY, *MaxSZ;
    QLabel *gravityLabel, *deltaTimeLabel, *MinS, *MaxS;
   
};

#endif
