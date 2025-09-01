
SetFactory("OpenCASCADE");

// gmsh naca_2D_shape_0.geo -2 -format msh2 -o naca_2D_shape_0.msh

N_fact = 2;

// Parametres
c   = 1.0;		// corde
m   = 0.01500000; 		// cambrure max  
p   = 0.61400000;		// position cambrure 
t   = 0.12; 		// epaisseur relative
N   = 200;		// nombre de points demi profil
lc  = 0.01;		// taille de maillage
alpha = -0*Pi/180;	// angle d’attaque

//pos_x = 0.5;
pos_x = 0.355;

// Creation des points
For i In {0:N-1}
	xr = i/N;
	// ligne de cambrure yc
	//yc = 0; // symetrique
	If (xr<=p)  
		yc = m/p^2*(2*p*xr - xr^2); 
	EndIf
	If (xr>p)  
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





// Tracer spline
pts1[] = {};
For i In {10:N} 
  pts1[] += i;
EndFor
Spline(1) = pts1[];

pts2[] = {};
For i In {N+1:2*N-10} 
  pts2[] += i;
EndFor
Spline(2) = pts2[];

pts3[] = {};
For i In {2*N-10:2*N-1} 
  pts3[] += i;
EndFor
For i In {1: 10} 
  pts3[] += i;
EndFor
Spline(3) = pts3[];



//+
Point(400) = {0.4, 1, 0, 1.0};
//+
Point(401) = {0.4, -1, 0, 1.0};
//+
Point(402) = {1.35, 1, -0, 1.0};
//+
Point(403) = {1.35, -1, -0, 1.0};
//+
Point(404) = {3, 1, -0, 1.0};
//+
Point(405) = {3, -1, -0, 1.0};
//+
Point(406) = {3, -0.00, 0, 0.0};
//+
Point(407) = {0.4-1, 0, 0, 1.0};
//+
Point(408) = {0.4-Sqrt(2)/2, -Sqrt(2)/2, 0, 1.0};
//+
Point(409) = {0.4-Sqrt(2)/2, +Sqrt(2)/2, 0, 1.0};
//+
Point(410) = {0.4-Sqrt(2-Sqrt(2))/2, -Sqrt(2+Sqrt(2))/2, 0, 1.0};
//+
Point(411) = {0.4-Sqrt(2-Sqrt(2))/2, +Sqrt(2+Sqrt(2))/2, 0, 1.0};
//+
Point(412) = {3, -0.002, 0, 0.0};


//+
Spline(4) = {401, 410, 408, 407, 409, 411, 400};
//+
Line(5) = {400, 402};
//+
Line(6) = {401, 403};
//+
Line(7) = {402, 404};
//+
Line(8) = {403, 405};
//+
Line(9) = {404, 406};
//+
Line(40) = {405 , 412};
//+
Line(41) = {N+1 , 412};
//+
Line(42) = {406 , 412};

//+
Line(11) = {400, 10};
//+
Line(12) = {401, 2*N-10};
//+
Line(13) = {402, N};
//+
Line(14) = {403, N+1};
//+
Line(15) = {N, 406};
//+
Line(16) = {N, N+1};




//+
Curve Loop(1) = {12, 3, -11, -4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {12, -2, -14, -6};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {1, -13, -5, 11};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {15, -13, 7, 9};
//+
Plane Surface(4) = {4};
//+
Curve Loop(6) = {14, 41, -40, -8};
//+
Plane Surface(5) = {6};
//+
Curve Loop(7) = {41, -42, -15, 16};
//+
Plane Surface(6) = {7};


//+
Physical Curve("Inlet", 41) = {4};
//+
Physical Curve("FreeStream", 42) = {5, 6, 7, 8};
//+
Physical Curve("Outlet", 43) = {9, 40, 42};
//+
Physical Curve("Airfoil", 44) = {3, 1, 2, 16};
//+
Physical Surface("Fluid", 45) = {1, 3, 2, 4, 5, 6};



//+
Transfinite Curve {11, 13, 12, 14, 9, 40} = N_fact * 25 Using Progression 0.94;
//+
Transfinite Curve {5, 6, 1, 2} = N_fact * 100 Using Progression 1;
//+
Transfinite Curve {7, 8, 15, 41} = N_fact * 20 Using Progression 1.10;
//+
Transfinite Curve {4, 3} = N_fact * 20 Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Transfinite Surface {2};
//+
Recombine Surface {1, 3, 2, 5, 4};
//+
Transfinite Curve {16, 42} = 1 Using Progression 1;
//+
Transfinite Surface {6};
//+
Recombine Surface {6};






