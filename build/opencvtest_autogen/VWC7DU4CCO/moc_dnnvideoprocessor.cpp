/****************************************************************************
** Meta object code from reading C++ file 'dnnvideoprocessor.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../DNN/dnnvideoprocessor.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'dnnvideoprocessor.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_DnnVideoProcessor_t {
    QByteArrayData data[13];
    char stringdata0[133];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DnnVideoProcessor_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DnnVideoProcessor_t qt_meta_stringdata_DnnVideoProcessor = {
    {
QT_MOC_LITERAL(0, 0, 17), // "DnnVideoProcessor"
QT_MOC_LITERAL(1, 18, 10), // "frameReady"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 5), // "frame"
QT_MOC_LITERAL(4, 36, 15), // "detectionsReady"
QT_MOC_LITERAL(5, 52, 26), // "std::vector<YoloDetection>"
QT_MOC_LITERAL(6, 79, 10), // "detections"
QT_MOC_LITERAL(7, 90, 5), // "error"
QT_MOC_LITERAL(8, 96, 3), // "msg"
QT_MOC_LITERAL(9, 100, 8), // "finished"
QT_MOC_LITERAL(10, 109, 5), // "start"
QT_MOC_LITERAL(11, 115, 12), // "processFrame"
QT_MOC_LITERAL(12, 128, 4) // "stop"

    },
    "DnnVideoProcessor\0frameReady\0\0frame\0"
    "detectionsReady\0std::vector<YoloDetection>\0"
    "detections\0error\0msg\0finished\0start\0"
    "processFrame\0stop"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DnnVideoProcessor[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   49,    2, 0x06 /* Public */,
       4,    1,   52,    2, 0x06 /* Public */,
       7,    1,   55,    2, 0x06 /* Public */,
       9,    0,   58,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      10,    0,   59,    2, 0x0a /* Public */,
      11,    0,   60,    2, 0x0a /* Public */,
      12,    0,   61,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QImage,    3,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, QMetaType::QString,    8,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void DnnVideoProcessor::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<DnnVideoProcessor *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->frameReady((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 1: _t->detectionsReady((*reinterpret_cast< const std::vector<YoloDetection>(*)>(_a[1]))); break;
        case 2: _t->error((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 3: _t->finished(); break;
        case 4: _t->start(); break;
        case 5: _t->processFrame(); break;
        case 6: _t->stop(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (DnnVideoProcessor::*)(const QImage & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DnnVideoProcessor::frameReady)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (DnnVideoProcessor::*)(const std::vector<YoloDetection> & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DnnVideoProcessor::detectionsReady)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (DnnVideoProcessor::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DnnVideoProcessor::error)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (DnnVideoProcessor::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DnnVideoProcessor::finished)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject DnnVideoProcessor::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_DnnVideoProcessor.data,
    qt_meta_data_DnnVideoProcessor,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *DnnVideoProcessor::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DnnVideoProcessor::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DnnVideoProcessor.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int DnnVideoProcessor::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void DnnVideoProcessor::frameReady(const QImage & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void DnnVideoProcessor::detectionsReady(const std::vector<YoloDetection> & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void DnnVideoProcessor::error(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void DnnVideoProcessor::finished()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
