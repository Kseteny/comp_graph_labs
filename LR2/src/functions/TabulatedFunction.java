package functions;
public class TabulatedFunction {
    FunctionPoint[] points;
    TabulatedFunction(double leftX, double rightX, int pointsCount) {
        if (leftX > rightX){
            double temp = leftX;
            leftX = rightX;
            rightX = temp;
        }
        //10, 0, 5
        points = new FunctionPoint[pointsCount];
        double step = (rightX - leftX)/(pointsCount);
        for (int i = 0; i < pointsCount; i++) {
            points[i] = new FunctionPoint(leftX + i * step, 0); // Инициализация точек
        }
    }
    TabulatedFunction(double leftX, double rightX, double[] values) {
        points = new FunctionPoint[values.length];
        double step = (rightX - leftX) / (values.length - 1);
        for (int i = 0; i < values.length; i++) {
            points[i] = new FunctionPoint(leftX + i * step, values[i]);
        }
    }
}
