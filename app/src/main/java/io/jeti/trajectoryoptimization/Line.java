package io.jeti.trajectoryoptimization;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

public class Line extends View {
    Paint paint = new Paint();

    private void init() {
        paint.setColor(Color.BLACK);
    }

    public Line(Context context) {
        super(context);
        init();
    }

    public Line(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public Line(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init();
    }

    @Override
    public void onDraw(Canvas canvas) {
        canvas.drawLine(0, 0, 20, 20, paint);
        canvas.drawLine(20, 0, 0, 20, paint);
    }

}