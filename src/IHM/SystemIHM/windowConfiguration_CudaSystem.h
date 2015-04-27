#ifndef WINDOW_CONFIGURATION_CUDA_SYSTEM_H
#define WINDOW_CONFIGURATION_CUDA_SYSTEM_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ForceExt_Gravity.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_CudaSystem : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_CudaSystem(GLWidget *glWidget);
    ~WindowConfiguration_CudaSystem();

    QSize sizeHint() const;

public slots:

    void accept();
    void cancel();
    void changeData(double);
    void changeGravityData(double);

private:

    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QDoubleSpinBox *gravity, *deltaTime;
    QLabel *gravityLabel, *deltaTimeLabel;
    
};

#endif
