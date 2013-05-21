#ifndef WINDOW_CONFIGURATION_SIMPLE_SYSTEM_H
#define WINDOW_CONFIGURATION_SIMPLE_SYSTEM_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ForceExt_Gravity.h>

class Window;
//************************************************************************/
//************************************************************************/
class WindowConfiguration_SimpleSystem : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_SimpleSystem(Window* parent, QVBoxLayout* mainLayoutParent, QHBoxLayout* layoutRightParent, GLWidget *glWidget);
    ~WindowConfiguration_SimpleSystem();

    QSize sizeHint() const;

public slots:

    void accept();
    void cancel();
    void changeData(double);
    void changeGravityData(double);

private:
	
    GLWidget *glWidget;
    QWidget *page;
    Window* parent; QVBoxLayout* mainLayoutParent; QHBoxLayout* layoutRightParent;

    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QDoubleSpinBox *gravity, *deltaTime;
    QLabel *gravityLabel, *deltaTimeLabel;

    void createPage_SceneWindow();
    
};

#endif
