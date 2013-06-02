#ifndef WINDOW_CONFIGURATION_ANIMATED_HEIGHT_FIELD_H
#define WINDOW_CONFIGURATION_ANIMATED_HEIGHT_FIELD_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>

#include <AnimatedPeriodicHeightField.h>

//************************************************************************/
//************************************************************************/
class WindowConfiguration_AnimatedHeightField : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_AnimatedHeightField(GLWidget *glWidget);
    ~WindowConfiguration_AnimatedHeightField();

    AnimatedPeriodic_HeightField *getHeightField();

public slots:

    void accept();
    void cancel();

    void displayHeightField(double);
    void displayHeightField(int);

    void loadSpectrum();
    void saveSpectrum();

private:
	
    AnimatedPeriodic_HeightField *H;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;
    QPushButton *buttonLoad;
    QPushButton *buttonSave;

    QVBoxLayout *layout;
    QTabWidget *onglets;

   
    QDoubleSpinBox *OX, *OY, *OZ, *length, *width, *dx, *dz, *AMin, *AMax, *kMin, *kMax, *thetaMin, *thetaMax,
		   *phiMin, *phiMax, *elast, *omegaMin, *omegaMax, *t;
    QSpinBox *nbFunc;
    QLabel *widthLabel, *lengthLabel, *dxLabel, *dzLabel, *originLabel, *aLabel, *bLabel, 
	   *nbFuncLabel, *AMinLabel, *AMaxLabel, *kMinLabel, *kMaxLabel, *thetaMinLabel, *thetaMaxLabel, 
	   *phiMinLabel, *phiMaxLabel, *omegaMinLabel, *omegaMaxLabel, *tLabel, *elastLabel;

    string getOpenFileName(const QString & caption, const QStringList & filters,int * ind_filter);
    QDir current_dir;
    
};

#endif
