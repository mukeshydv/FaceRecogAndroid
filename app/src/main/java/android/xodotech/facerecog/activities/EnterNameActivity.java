package android.xodotech.facerecog.activities;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.xodotech.facerecog.R;

public class EnterNameActivity extends AppCompatActivity {

    EditText nameEditText;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_enter_name);

        nameEditText = (EditText) findViewById(R.id.nameEditText);

        Button startButton = (Button) findViewById(R.id.startButton);

        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String name = nameEditText.getText().toString();
                Intent intent = new Intent(EnterNameActivity.this, AddNewFace.class);
                intent.putExtra("Name", name);
                startActivity(intent);
            }
        });
    }
}
