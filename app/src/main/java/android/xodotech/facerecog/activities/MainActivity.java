package android.xodotech.facerecog.activities;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.xodotech.facerecog.R;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.MatName;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessorFactory;
import ch.zhaw.facerecognitionlibrary.Recognition.Recognition;
import ch.zhaw.facerecognitionlibrary.Recognition.RecognitionFactory;

public class MainActivity extends AppCompatActivity {


    private Thread trainingThread;

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button trainingButton = (Button) findViewById(R.id.trainingButton);
        Button recogniseButton = (Button) findViewById(R.id.recogniseButton);

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        sharedPreferences.edit().putString("key_modelFileTensorFlow", "optimized_facenet.pb").commit();

        trainingButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, EnterNameActivity.class);
                startActivity(intent);
            }
        });

        recogniseButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, RecognisingActivity.class);
                startActivity(intent);
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (trainingThread != null) trainingThread.interrupt();
    }

    public void onDestroy() {
        super.onDestroy();
        if (trainingThread != null) trainingThread.interrupt();
    }

    @Override
    protected void onResume() {
        super.onResume();
        startTraining();
    }

    private void startTraining() {

        trainingThread = new Thread(new Runnable() {
            public void run() {
                if(!Thread.currentThread().isInterrupted()){
                    PreProcessorFactory ppF = new PreProcessorFactory(getApplicationContext());

                    FileHelper fileHelper = new FileHelper();
                    fileHelper.createDataFolderIfNotExsiting();
                    final File[] persons = fileHelper.getTrainingList();
                    if (persons.length > 0) {
                        Recognition rec = RecognitionFactory.getRecognitionAlgorithm(getApplicationContext(), Recognition.TRAINING, "TensorFlow with SVM or KNN");
                        for (File person : persons) {
                            if (person.isDirectory()){
                                File[] files = person.listFiles();
                                for (File file : files) {
                                    if (FileHelper.isFileAnImage(file)){
                                        Mat imgRgb = Imgcodecs.imread(file.getAbsolutePath());
                                        Imgproc.cvtColor(imgRgb, imgRgb, Imgproc.COLOR_BGRA2RGBA);
                                        Mat processedImage = new Mat();
                                        imgRgb.copyTo(processedImage);
                                        List<Mat> images = ppF.getProcessedImage(processedImage, PreProcessorFactory.PreprocessingMode.RECOGNITION);
                                        if (images == null || images.size() > 1) {
                                            // More than 1 face detected --> cannot use this file for training
                                            continue;
                                        } else {
                                            processedImage = images.get(0);
                                        }
                                        if (processedImage.empty()) {
                                            continue;
                                        }
                                        // The last token is the name --> Folder name = Person name
                                        String[] tokens = file.getParent().split("/");
                                        final String name = tokens[tokens.length - 1];

                                        MatName m = new MatName("processedImage", processedImage);
                                        fileHelper.saveMatToImage(m, FileHelper.DATA_PATH);

                                        rec.addImage(processedImage, name, false);
                                    }
                                }
                            }
                        }

                        rec.train();
                    } else {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        });

        trainingThread.start();
    }
}
