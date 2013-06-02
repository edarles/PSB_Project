/****************************************************************************
** Meta object code from reading C++ file 'windowConfiguration_PeriodicHeightField.h'
**
** Created: Mon Jun 3 01:19:11 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "windowConfiguration_PeriodicHeightField.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'windowConfiguration_PeriodicHeightField.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_WindowConfiguration_PeriodicHeightField[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      41,   40,   40,   40, 0x0a,
      50,   40,   40,   40, 0x0a,
      59,   40,   40,   40, 0x0a,
      86,   40,   40,   40, 0x0a,
     110,   40,   40,   40, 0x0a,
     125,   40,   40,   40, 0x0a,
     140,   40,   40,   40, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_WindowConfiguration_PeriodicHeightField[] = {
    "WindowConfiguration_PeriodicHeightField\0"
    "\0accept()\0cancel()\0displayHeightField(double)\0"
    "displayHeightField(int)\0loadSpectrum()\0"
    "saveSpectrum()\0add()\0"
};

void WindowConfiguration_PeriodicHeightField::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WindowConfiguration_PeriodicHeightField *_t = static_cast<WindowConfiguration_PeriodicHeightField *>(_o);
        switch (_id) {
        case 0: _t->accept(); break;
        case 1: _t->cancel(); break;
        case 2: _t->displayHeightField((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->displayHeightField((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->loadSpectrum(); break;
        case 5: _t->saveSpectrum(); break;
        case 6: _t->add(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData WindowConfiguration_PeriodicHeightField::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject WindowConfiguration_PeriodicHeightField::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_WindowConfiguration_PeriodicHeightField,
      qt_meta_data_WindowConfiguration_PeriodicHeightField, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &WindowConfiguration_PeriodicHeightField::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *WindowConfiguration_PeriodicHeightField::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *WindowConfiguration_PeriodicHeightField::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_WindowConfiguration_PeriodicHeightField))
        return static_cast<void*>(const_cast< WindowConfiguration_PeriodicHeightField*>(this));
    return QWidget::qt_metacast(_clname);
}

int WindowConfiguration_PeriodicHeightField::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
