#ifndef WINDOW_CONFIGURATION_PERIODIC_HEIGHT_FIELD_H
#define WINDOW_CONFIGURATION_PERIODIC_HEIGHT_FIELD_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <PeriodicHeightFieldCollision.h>

class WindowConfiguration_CombinedHeightField;
//************************************************************************/
//************************************************************************/
class WindowConfiguration_PeriodicHeightField : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_PeriodicHeightField(WindowConfiguration_CombinedHeightField*, GLWidget *glWidget);
    ~WindowConfiguration_PeriodicHeightField();

    Periodic_HeightFieldCollision *getHeightField();

public slots:

    void accept();
    void cancel();
    void displayHeightField(double);
    void displayHeightField(int);
    void loadSpectrum();
    void saveSpectrum();

    void add();

private:
	
    Periodic_HeightFieldCollision *H;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;
    QPushButton *buttonLoad;
    QPushButton *buttonSave;

    QVBoxLayout *layout;
    QTabWidget *onglets;

   
    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *dx, *dz, *AMin, *AMax, *kMin, *kMax, *thetaMin, *thetaMax,
		   *phiMin, *phiMax, *wMin, *wMax, *elast;
    QSpinBox *nbFunc;
    QLabel *widthLabel, *lengthLabel, *dxLabel, *dzLabel, *originLabel, *aLabel, *bLabel, 
	   *nbFuncLabel, *AMinLabel, *AMaxLabel, *kMinLabel, *kMaxLabel, *thetaMinLabel, *thetaMaxLabel, 
	   *phiMinLabel, *phiMaxLabel, *wMinLabel, *wMaxLabel, *elastLabel;

    string getOpenFileName(const QString & caption, const QStringList & filters,int * ind_filter);
    QDir current_dir;

    WindowConfiguration_CombinedHeightField* parent;
    
};

#endif
