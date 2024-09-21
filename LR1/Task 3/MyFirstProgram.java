class MyFirstClass {
      public static void main(String[] s) {
         MySecondClass o = new MySecondClass(111, 255);
	System.out.println(o.slojenie());
	for (int i = 1; i <= 8; i++) {
 	for (int j = 1; j <= 8; j++) {
     	o.setA(i);
     	o.setB(j);
     	System.out.print(o.slojenie());
     	System.out.print(" ");
 	}
 	System.out.println();
	}

      }
   }

class MySecondClass{
	private int a;
	private int b;
	public MySecondClass() { 
	a = 0;
	b = 0; 
	}
	public MySecondClass(int A , int B) { 
	a = A;
	b = B; 
	}
	public int getA(){ return a; }
	public int getB(){ return b; }
	public void setA(int A) { a = A; }
	public void setB(int B) { b = B; }
	public int slojenie() { return a+b; }
}