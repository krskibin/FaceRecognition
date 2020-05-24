package com.example.facerecognition;

import android.content.Context;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    JavaCameraView javaCameraView;
    File cascadeFile;
    CascadeClassifier faceDetector;

    private Mat mRGBa, mGrey;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = findViewById(R.id.javaCameraView);

        // check if OpenCV is loaded
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseCallback);

        } else {
            try {
                baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGBa = new Mat();
        mGrey = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRGBa.release();
        mGrey.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRGBa = inputFrame.rgba();
        mGrey = inputFrame.gray();

        // face detecting code
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mRGBa, faceDetections);

        for (Rect rect: faceDetections.toArray()) {
            // set graphical rectangle on detected face in camera mode
            Imgproc.rectangle(mRGBa, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0));
        }

        return mRGBa;
    }

    private BaseLoaderCallback baseCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) throws IOException {
            switch (status) {
                case SUCCESS : {

                    // after successfully initialized OpenCV loader, app needs to load face recognition
                    // algorithm in this case: Haarcascade from res/raw (haarcascade_frontalface_alt2.xml
                    // was shipped with OpenCV library

                    InputStream inputStream = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDirectory = getDir("cascade", Context.MODE_PRIVATE);
                    cascadeFile = new File(cascadeDirectory, "haarcascade_frontalface_alt2.xml");

                    FileOutputStream fileOutputStream;

                    try {
                        fileOutputStream = new FileOutputStream(cascadeFile);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                        return;
                    }

                    byte[] buffer = new byte[4096];
                    int byteRead;

                    while ((byteRead = inputStream.read(buffer)) != -1) {
                        fileOutputStream.write(buffer, 0, byteRead);
                    }

                    inputStream.close();
                    fileOutputStream.close();

                    // Initialize faceDetector based on byte code from haarcascade file
                    faceDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());

                    if (faceDetector.empty()) {
                        faceDetector = null;
                        return;
                    }
                    cascadeDirectory.delete();

                    // Enable java camera view
                    javaCameraView.enableView();
                }
                break;

                default: {
                    super.onManagerConnected(status);
                }
            }
        }
    };
}
