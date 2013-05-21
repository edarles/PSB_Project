#ifndef WINDOW_CONFIGURATION_FORCE_EXT_CONSTANTE_H
#define WINDOW_CONFIGURATION_FORCE_EXT_CONSTANTE_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <glWidget.h>
#include <ForceExt_Constante.h>
//************************************************************************/
//************************************************************************/
class WindowConfiguration_ForceExt_Constante : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:

    WindowConfiguration_ForceExt_Constante(GLWidget *glWidget);
    ~WindowConfiguration_ForceExt_Constante();

    QSize sizeHint() const;

public slots:

    void accept();
    void cancel();
    void createForce(double);

private:
	
    ForceExt_Constante *F;
    GLWidget *glWidget;
    
    QPushButton *buttonOK;
    QPushButton *buttonCancel;

    QDoubleSpinBox *DX, *DY, *DZ, *A;
    QLabel *DLabel, *ALabel;
    
};

#endif
