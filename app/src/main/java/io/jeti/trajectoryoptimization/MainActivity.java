package io.jeti.trajectoryoptimization;

import android.graphics.Color;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.LinearLayout.LayoutParams;
import android.widget.ScrollView;
import android.widget.TextView;
import io.jeti.components.CheckBoxAndText;
import io.jeti.editables.EditableDouble;
import io.jeti.editables.EditableInteger;
import java.lang.ref.WeakReference;

public class MainActivity extends AppCompatActivity {

    /* Used to load the 'native-lib' library on application startup. */
    static {
        System.loadLibrary("native-lib");
    }

    EditableInteger iterations;
    EditableDouble  tolerance;
    CheckBoxAndText adaptive_mu_strategy;
    CheckBoxAndText hessian_approximation;
    CheckBoxAndText sparse_forward;
    CheckBoxAndText sparse_reverse;
    EditableInteger print_level;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        final ScrollView scrollView = new ScrollView(this);

        final LinearLayout linearLayout = new LinearLayout(this);
        linearLayout.setOrientation(LinearLayout.VERTICAL);
        scrollView.addView(linearLayout,
                new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        final LayoutParams params = new LayoutParams(LayoutParams.MATCH_PARENT,
                LayoutParams.WRAP_CONTENT, 0);

        final Button recalculateButton = new Button(this);
        recalculateButton.setText("Recalculate");
        linearLayout.addView(recalculateButton, params);

        /* All of the editable fields */
        iterations = new EditableInteger(this, "iterations");
        iterations.set(100);
        linearLayout.addView(iterations, params);

        tolerance = new EditableDouble(this, "tolerance");
        tolerance.set(1e-3);
        linearLayout.addView(tolerance, params);

        print_level = new EditableInteger(this, "print_level");
        print_level.set(0);
        linearLayout.addView(print_level, params);

        adaptive_mu_strategy = new CheckBoxAndText(this, "adaptive_mu_strategy",
                View.generateViewId());
        adaptive_mu_strategy.setChecked(true);
        linearLayout.addView(adaptive_mu_strategy, params);

        hessian_approximation = new CheckBoxAndText(this, "hessian_approximation",
                View.generateViewId());
        hessian_approximation.setChecked(true);
        linearLayout.addView(hessian_approximation, params);

        sparse_forward = new CheckBoxAndText(this, "sparse_forward", View.generateViewId());
        sparse_forward.setChecked(true);
        linearLayout.addView(sparse_forward, params);

        sparse_reverse = new CheckBoxAndText(this, "sparse_reverse", View.generateViewId());
        sparse_reverse.setChecked(true);
        linearLayout.addView(sparse_reverse, params);

        TextView line = new TextView(this);
        line.setBackgroundColor(Color.BLUE);
        linearLayout.addView(line, new LayoutParams(LayoutParams.MATCH_PARENT, 20));

        /* The text view that will hold the output */
        final TextView textView = new TextView(this);
        textView.setTextSize(10);
        linearLayout.addView(textView,
                new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        /* Finally, set the button to trigger the optimization */
        recalculateButton.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                recalculateButton.setEnabled(false);
                textView.setText("Recalculating");
                Thread thread = new Thread(new Optimize(textView, recalculateButton));
                thread.setPriority(Thread.MAX_PRIORITY);
                thread.run();
            }
        });

        setContentView(scrollView);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI(int iterations, double tolerance,
            boolean adaptive_mu_strategy, boolean hessian_approximation, boolean sparse_forward,
            boolean sparse_reverse, int print_level);

    private class Optimize implements Runnable {

        private final WeakReference<TextView> textViewWeakReference;
        private final WeakReference<Button>   buttonWeakReference;

        public Optimize(TextView textView, Button button) {
            this.textViewWeakReference = new WeakReference<>(textView);
            this.buttonWeakReference = new WeakReference<>(button);
        }

        @Override
        public void run() {
            final String result = stringFromJNI(iterations.get(), tolerance.get(),
                    adaptive_mu_strategy.isChecked(), hessian_approximation.isChecked(),
                    sparse_forward.isChecked(), sparse_reverse.isChecked(), print_level.get());
            TextView textView;
            if ((textView = textViewWeakReference.get()) != null) {
                textView.setText(result);
            }
            Button button;
            if ((button = buttonWeakReference.get()) != null) {
                button.setEnabled(true);
            }

        }
    }
}
