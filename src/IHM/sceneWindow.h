#ifndef SCENE_WINDOW_H
#define SCENE_WINDOW_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <QtGui>
#include <QMainWindow>

#include <QScrollArea>
#include <QTabWidget>
#include <System.h>
#include <SimpleSystem.h>

//************************************************************************/
//************************************************************************/
class SceneWindow : public QScrollArea
{
    Q_OBJECT

//************************************************************************/
public:
    SceneWindow();

    void clearTabWidget();

    void addPageSystem();
   // void addPageEmitterSystem(System*);
   // void addPageCollisionSystem(System*);

private :

    //************************************************************************/
   QTabWidget *tabWidget;
   QVBoxLayout* mainLayout;

   QWidget* createPage_SimpleSystem();   

};
//************************************************************************/
//************************************************************************/
#endif

