#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QThread>
#include <iostream>
#include "ONNXRUNTIME/streamprocessor.h"
#include "opencvimage/imageprovider.h"
#include "opencvimage/framesource.h"

// ONNX Runtime 方案（StreamProcessor + OnnxYoloDetector）
// 使用 ONNX Runtime 推理引擎加载 .onnx 模型进行目标检测

int main(int argc, char *argv[])
{
    std::cout << "[main] Step 1: Starting..." << std::endl;

    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);

    // ── 1. 图像提供者（供 QML 显示） ──
    ImageProvider *provider = new ImageProvider();

    // ── 2. FrameSource（QML 属性绑定，替代 Timer 刷新） ──
    FrameSource *frameSource = new FrameSource();

    // ── 3. ONNX Runtime 视频处理器（工作线程） ──
    QThread *videoThread = new QThread;
    StreamProcessor *processor = new StreamProcessor;

    // 打开本地视频文件
    if (!processor->openVideo("E:\\baizhuangjia2.mp4")) {
        std::cerr << "[main] openVideo FAILED" << std::endl;
        delete processor;
        delete videoThread;
        return -1;
    }

    // 加载 ONNX Runtime YOLO 模型 + 类别名称
    if (!processor->loadYoloModel("E:\\QTproject\\yolov3model2\\best.onnx",
                                   "E:\\QTproject\\yolov4model\\test.names")) {
        std::cerr << "[main] loadYoloModel FAILED" << std::endl;
        delete processor;
        delete videoThread;
        return -1;
    }

    // 调节 ONNX Runtime 推理参数
    processor->setConfThreshold(0.4f);
    processor->setNmsThreshold(0.5f);
    processor->setInputSize(416, 416);
    processor->setTargetFps(30);

    // 可选：开启图像增强处理
    // processor->setAutoEnhance(true);
    // processor->setDenoise(true);

    processor->moveToThread(videoThread);

    // 工作线程每帧 → 主线程: ① 更新 ImageProvider  ② 触发 QML 刷新
    QObject::connect(processor, &StreamProcessor::frameReady, qApp,
                     [provider, frameSource](const QImage &img) {
        provider->updateImage(img);
        frameSource->refresh();         // QML 属性绑定自动触发 requestImage
    });

    QObject::connect(processor, &StreamProcessor::errorOccurred,
                     [](const QString &msg) {
        std::cerr << "[StreamProcessor] " << msg.toStdString() << std::endl;
    });

    QObject::connect(videoThread, &QThread::started,
                     processor, &StreamProcessor::start);

    QObject::connect(processor, &StreamProcessor::finished,
                     videoThread, &QThread::quit);

    QObject::connect(videoThread, &QThread::finished,
                     processor, &QObject::deleteLater);
    QObject::connect(videoThread, &QThread::finished,
                     videoThread, &QObject::deleteLater);

    videoThread->start();
    std::cout << "[main] Step 2: Video thread started" << std::endl;

    // ── 4. QML 界面 ──
    QQmlApplicationEngine engine;

    // 注册 ImageProvider（供 image:// 协议）
    engine.addImageProvider(QLatin1String("imageProvider"), provider);

    // 将 FrameSource 暴露给 QML 根上下文（属性绑定）
    engine.rootContext()->setContextProperty(
        QStringLiteral("frameSource"), frameSource);

    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl) {
            std::cerr << "[main] QML object creation FAILED!" << std::endl;
            QCoreApplication::exit(-1);
        }
    }, Qt::QueuedConnection);

    engine.load(url);
    std::cout << "[main] Step 3: QML loaded" << std::endl;

    // 优雅关闭
    QObject::connect(&app, &QCoreApplication::aboutToQuit, [processor, videoThread]() {
        processor->stop();
        videoThread->quit();
    });

    std::cout << "[main] Step 4: Entering event loop" << std::endl;
    return app.exec();
}


// ===================================================================
// 备选方案，保留参考
// ===================================================================
//
// ── Darknet YOLOv3-tiny 方案（DnnVideoProcessor + OpenCV DNN） ──
// #include "DNN/dnnvideoprocessor.h"
//
// QThread *videoThread = new QThread;
// DnnVideoProcessor *processor = new DnnVideoProcessor;
// processor->openVideo("E:\\baizhuangjia2.mp4");
// processor->loadYoloModel("yolov3-tiny_mens.cfg",
//                          "yolov3_tiny_mens.weights",
//                          "test.names");
// processor->setConfThreshold(0.4f);
// processor->setNmsThreshold(0.5f);
// processor->setInputSize(416, 416);
// processor->moveToThread(videoThread);
//
// ── ONNX OpenCV DNN 方案（ONNXVideoProcessor + OpenCV DNN） ──
// #include "DNN/onnxvideoprocessor.h"
//
// QThread *videoThread = new QThread;
// ONNXVideoProcessor *processor = new ONNXVideoProcessor;
// processor->openVideo("E:\\lvzhuangjia.mp4");
// processor->loadONNXModel("best.onnx", "test.names");
// processor->setConfThreshold(0.5f);
// processor->setNmsThreshold(0.4f);
// processor->moveToThread(videoThread);
//
// ── ORB 模板匹配方案（VideoProcessor + ORB 特征匹配） ──
// #include "opencv/videoprocessor.h"
//
// QThread *videoThread = new QThread;
// VideoProcessor *processor = new VideoProcessor;
// processor->openVideo("E:\\tank.mp4");
// processor->addTemplateFile("E:\\tank.png", "tank_side");
// processor->moveToThread(videoThread);
