package android.xodotech.facerecog.activities;

import android.content.Intent;
import android.os.Handler;
import android.os.Looper;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.xodotech.facerecog.R;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.Date;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.CustomCameraView;
import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.MatName;
import ch.zhaw.facerecognitionlibrary.Helpers.MatOperation;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessorFactory;
import ch.zhaw.facerecognitionlibrary.Recognition.Recognition;
import ch.zhaw.facerecognitionlibrary.Recognition.RecognitionFactory;

public class AddNewFace extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private CustomCameraView cameraView;
    private FileHelper fileHelper;
    private PreProcessorFactory ppF;
    private String name;
    private int totalCaptures;
    private long lastTime;
    private static final long TIME_DIFF = 500;
    private static final int NUMBER_OF_PICTURES = 30;
    private static final int EXPOSURE_COMPENSATION = 50;

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_new_face);

        Intent intent = getIntent();
        name = intent.getStringExtra("Name");

        totalCaptures = 0;
        lastTime = new Date().getTime();
        fileHelper = new FileHelper();

        cameraView = (CustomCameraView) findViewById(R.id.AddPersonPreview);
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setMaxFrameSize(640, 480);
        cameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onResume()
    {
        super.onResume();

        ppF = new PreProcessorFactory(this);
        cameraView.enableView();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (cameraView != null)
            cameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (cameraView != null)
            cameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        cameraView.setExposure(EXPOSURE_COMPENSATION);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat imgRgba = inputFrame.rgba();
        Mat imgCopy = new Mat();
        imgRgba.copyTo(imgCopy);

        Core.flip(imgRgba,imgRgba,1);

        long time = new Date().getTime();

        if(totalCaptures < NUMBER_OF_PICTURES) {
            if (lastTime + TIME_DIFF < time) {
                lastTime = time;

                // Check that only 1 face is found. Skip if any or more than 1 are found.
                List<Mat> images = ppF.getCroppedImage(imgCopy);
                if (images != null && images.size() == 1) {
                    Mat img = images.get(0);
                    if (img != null) {
                        Rect[] faces = ppF.getFacesForRecognition();
                        //Only proceed if 1 face has been detected, ignore if 0 or more than 1 face have been detected
                        if ((faces != null) && (faces.length == 1)) {
                            faces = MatOperation.rotateFaces(imgRgba, faces, ppF.getAngleForRecognition());

                            MatName m = new MatName(name + "_" + totalCaptures, img);

                            String wholeFolderPath = fileHelper.TRAINING_PATH + name;
                            new File(wholeFolderPath).mkdirs();
                            fileHelper.saveMatToImage(m, wholeFolderPath + "/");

                            for (int i = 0; i < faces.length; i++) {
                                MatOperation.drawRectangleAndLabelOnPreview(imgRgba, faces[i], String.valueOf(totalCaptures), true);
                            }

                            totalCaptures++;

                            // Stop after numberOfPictures (settings option)
                            if (totalCaptures >= NUMBER_OF_PICTURES) {
                                final Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                                intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                                startActivity(intent);
                            }
                        }
                    }
                }
            }
        }
        return imgRgba;
    }
}
