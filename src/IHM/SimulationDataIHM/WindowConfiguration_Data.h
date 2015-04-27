#ifndef WINDOW_CONFIGURATION_DATA_H
#define WINDOW_CONFIGURATION_DATA_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_Data : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_Data(QWidget*);
    ~WindowConfiguration_Data();

    SimulationData *getData();

protected:
	
    SimulationData *data;
};

#endif
