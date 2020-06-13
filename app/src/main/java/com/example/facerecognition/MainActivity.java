package com.example.facerecognition;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
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

/**
 * Main Activity class that handle the main activity layout, also
 * implements CameraBridgeViewBase.CVCameraViewListener2 make usage of
 * openCV camera view
 */
public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase mCameraView;
    File cascadeFile;
    CascadeClassifier faceDetector;
    private SharedPreferences prefs;

    private Mat mRGBa, mGray;

    /**
     * Overridden onCreate method that handle app onCreate state, which
     * means that this method is called immediately after running the app.
     * It is mainly used to create the app state.
      * @param savedInstanceState
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prefs = PreferenceManager.getDefaultSharedPreferences(this);

        mCameraView = findViewById(R.id.java_camera_view);
        mCameraView.setCameraIndex(prefs.getInt("mCameraIndex", CameraBridgeViewBase.CAMERA_ID_FRONT));
        mCameraView.setVisibility(SurfaceView.VISIBLE);
        mCameraView.setCvCameraViewListener(this);

        final GestureDetector mGestureDetector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onDown(MotionEvent e) {
                return true;
            }
            @Override
            public boolean onDoubleTap(MotionEvent e) {
                // Flip camera
                mCameraView.flipCamera();
                return true;
            }
        });

        mCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                return mGestureDetector.onTouchEvent(event);
            }
        });

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
    }

    /**
     * onDestroy method that handles on destroy activity state.
     * It is responsible for disabling the mCameraView
     */
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mCameraView != null)
            mCameraView.disableView();
    }

    /**
     * onCameraViewStarted method that sets mGray and mRGBa state
     * after launching the Camera
     * @param width -  the width of the frames that will be delivered
     * @param height - the height of the frames that will be delivered
     */
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRGBa = new Mat();
    }

    /**
     * onCameraViewStopped is responsible for releasing the mGray and mRGBa
     * global state when application is on hold.
     */
    public void onCameraViewStopped() {
        mGray.release();
        mRGBa.release();
    }

    /**
     * Method that takes every camera frame and
     * sets its orientation, detects face, and draws the rectangle on them.
     * @param inputFrame
     * @return
     */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mGrayTmp = inputFrame.gray();
        Mat mRgbaTmp = inputFrame.rgba();

        // Flip image to get mirror effect
        int orientation = mCameraView.getScreenOrientation();
        if (mCameraView.isEmulator()) // Treat emulators as a special case
            Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
        else {
            switch (orientation) { // RGB image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    if (mCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mRgbaTmp, mRgbaTmp, 0); // Flip along x-axis
                    else
                        Core.flip(mRgbaTmp, mRgbaTmp, -1); // Flip along both axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    if (mCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
                    break;
            }
            switch (orientation) { // Grayscale image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    if (mCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mGrayTmp, mGrayTmp, -1); // Flip along both axis
                    else
                        Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;

                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    if (mCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_BACK)
                        Core.flip(mGrayTmp, mGrayTmp, 0); // Flip along x-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                    if (mCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    Core.flip(mGrayTmp, mGrayTmp, 0); // Flip along x-axis
                    if (mCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_BACK)
                        Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
            }
        }

        mRGBa = mRgbaTmp;
        mGray = mGrayTmp;

        // face detecting code
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mRGBa, faceDetections);

        for (Rect rect: faceDetections.toArray()) {
            // set graphical rectangle on detected face in camera mode
            Imgproc.rectangle(mRGBa, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0));
        }

        return mRGBa;
    }

    /**
     * onResume method that handles onResume devices state
     * after getting back to the application
     * also checks if OpenCVLoader works
     */
    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, baseCallback);
        } else {
            try {
                baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } catch (IOException e) {

            }
        }
    }

    /**
     * BaseLoaderCallback is a baseCallback instance that takes app context
     * and handle the OpenCV connected state
     */
    private BaseLoaderCallback baseCallback = new BaseLoaderCallback(this) {

        /**
         * onManagerConnected method that takes OpenCV status
         * and if Open CV is loaded, methods opens the file stream
         * to load haarcascade_frontalface algorythm and then enables
         * the OpenCV Camera state
         * @param status
         * @throws IOException
         */
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
                    mCameraView.enableView();
                }
                break;

                default: {
                    super.onManagerConnected(status);
                }
            }
        }
    };



}
