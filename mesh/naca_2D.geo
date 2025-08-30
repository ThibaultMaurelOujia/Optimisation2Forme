
SetFactory("OpenCASCADE");

N_fact = 2;

// Parametres
c   = 1.0;		// corde
m   = 0.02; 		// cambrure max  
p   = 0.4;		// position cambrure 
t   = 0.12; 		// epaisseur relative
N   = 50;		// nombre de points demi profil
lc  = 0.01;		// taille de maillage
alpha = -2.5*Pi/180;	// angle d’attaque

pos_x = 0.5;

// Creation des points
For i In {0:N-1}
	xr = i/N;
	// ligne de cambrure yc
	//yc = 0; // symetrique
	If (xr<=p)  
		yc = m/p^2*(2*p*xr - xr^2); 
	EndIf
	If (xr<=p)  
		yc = m/(1-p)^2*((1-2*p) + 2*p*xr - xr^2); 
	EndIf
	// distribution d’epaisseur yt
	yt = 5*t*c*(	0.2969*Sqrt(xr)
			-0.1260*xr
			-0.3516*xr^2
			+0.2843*xr^3
			-0.1015*xr^4 );
	x = xr*c;
	// Extrados
	Point(i+1)	= {x+pos_x, yc+yt, 0, lc};
	// Intrados 
	If (i>0)  
		Point(2*N-i)	= {x+pos_x, yc-yt, 0, lc};
	EndIf
EndFor


Translate { 0.008, -0.006, 0.0 } {
  Point{N};
}
Rotate { {0,0,1}, {0,0,0}, alpha } {
  Point{1:2*N-1};
}
Translate { 0.05, 0.0, 0.0 } {
  Point{1:2*N-1};
}

//// Tracer spline
//pts[] = {};
//For i In {1:2*N-1}
//  pts[] += i;
//EndFor
//pts[] += 1; 
//Spline(1) = pts[];
//Line Loop(1) = {1};
////Plane Surface(1) = {1};



// Tracer spline
pts1[] = {};
For i In {5:50} 
  pts1[] += i;
EndFor
Spline(1) = pts1[];

pts2[] = {};
For i In {50:96} 
  pts2[] += i;
EndFor
Spline(2) = pts2[];

pts3[] = {};
For i In {96:99} 
  pts3[] += i;
EndFor
For i In {1:5} 
  pts3[] += i;
EndFor
Spline(3) = pts3[];





//+
Point(100) = {0.4, 1, 0, 1.0};
//+
Point(101) = {0.4, -1, 0, 1.0};
//+
Point(102) = {1.6, 1, -0, 1.0};
//+
Point(103) = {1.6, -1, -0, 1.0};
//+
Point(104) = {3, 1, -0, 1.0};
//+
Point(105) = {3, -1, -0, 1.0};
//+
Point(106) = {3, -0.04689, 0, 0.0};
//+
Point(107) = {0, 0, 0, 1.0};

//+
Spline(4) = {101, 107, 100};
//+
Line(5) = {100, 102};
//+
Line(6) = {101, 103};
//+
Line(7) = {102, 104};
//+
Line(8) = {103, 105};
//+
Line(9) = {104, 106};
//+
Line(10) = {105 , 106};

//+
Line(11) = {100, 5};
//+
Line(12) = {101, 96};
//+
Line(13) = {102, 50};
//+
Line(14) = {103, 50};
//+
Line(15) = {106, 50};




//+
Transfinite Curve {3, 4} = N_fact * 15 Using Progression 1;
//+
Transfinite Curve {1, 2, 5, 6} = N_fact * 30 Using Progression 1;
//+
Transfinite Curve {9, 10, 11, 13, 12, 14} = N_fact * 20 Using Progression 0.9;
//+
Transfinite Curve {7, 8, 15} = N_fact * 10 Using Progression 1;


//+
Curve Loop(1) = {11, -3, -12, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, -13, -5, 11};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {12, -2, -14, -6};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {15, -14, 8, -10};
//+
Curve Loop(5) = {14, -15, 10, -8};
//+
Plane Surface(4) = {5};
//+
Curve Loop(6) = {9, 15, -13, 7};
//+
Plane Surface(5) = {6};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Recombine Surface {1, 2, 3, 4, 5};



//+
//Extrude {0, 0, 1} {
//  Surface{1}; Surface{2}; Surface{5}; Surface{4}; Surface{3}; Layers {1}; Recombine;
//}
//+
//Physical Volume("Fluid", 41) = {1, 2, 3, 4, 5};
//+
//Physical Surface("Inlet", 42) = {9};
//+
//Physical Surface("FreeStream", 43) = {13, 17, 24, 21};
//+
//Physical Surface("Outlet", 44) = {15, 20};
//+
//Physical Surface("Airfoil", 45) = {11, 23, 7};
//+
//Physical Surface("Side", 46) = {2, 5, 4, 3, 25, 14, 18, 22, 1, 10};




//+
Physical Curve("Inlet", 16) = {4};
//+
Physical Curve("FreeStream", 17) = {5, 7, 6, 8};
//+
Physical Curve("Outlet", 18) = {9, 10};
//+
Physical Curve("Airfoil", 19) = {1, 2, 3};
//+
Physical Surface("Fluid", 20) = {1, 2, 3, 5, 4};


