#ifndef WINDOW_CONFIGURATION_FORCE_EXT_TROCHOIDE_H
#define WINDOW_CONFIGURATION_FORCE_EXT_TROCHOIDE_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ForceExt_Trochoide.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_ForceExt_Trochoide : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_ForceExt_Trochoide(GLWidget *glWidget);
    ~WindowConfiguration_ForceExt_Trochoide();

    QSize sizeHint() const;

public slots:

    void accept();
    void cancel();
    void createForce(double);

private:
	
    ForceExt_Trochoide *T;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QDoubleSpinBox *A, *lambda, *theta, *f, *phi;
    QLabel *ALabel, *lambdaLabel, *thetaLabel, *fLabel, *phiLabel;
    
};

#endif
