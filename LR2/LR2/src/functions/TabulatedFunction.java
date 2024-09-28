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
            points[0] = new FunctionPoint()
        }
    }

    TabulatedFunction(double leftX, double rightX, double[] values) {

    }
}
