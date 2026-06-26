// Figure B: the transition map tau_AB = phi_B^{-1} o phi_A between two 3D bodies.
// Body A (upper) and body B (lower) in near contact, both star-shaped; a boundary
// point x = phi_A(theta_A) on A, a ray from c_B through the foot phi_B(psi_B) on B to x,
// the outward normal n at the foot, and the signed gap g_N from foot to x along n.
settings.outformat = "pdf";
settings.prc      = false;
settings.render   = 8;        // smooth rasterized surfaces; vector labels/lines
settings.tex      = "latex";

import three;
import graph3;

usepackage("amsmath");
usepackage("amssymb");
usepackage("bm");

size(12.5cm, 0);
currentlight = light(gray(0.80), specular=gray(0.16),
                     (3,-4,6), (-4,1,5));
currentprojection = orthographic(camera=(2.2, -4.6, 1.9),
                                 up=Z, target=(0,0,0.0), zoom=1.16);

// ----- palette --------------------------------------------------------------
pen fillA    = gray(0.64) + opacity(0.40);
pen fillB    = rgb(0.20,0.50,0.53) + opacity(0.42);
pen rayPen   = gray(0.30) + linewidth(0.7) + dashed;
pen normPen  = rgb(0.80,0.13,0.10) + linewidth(1.2);
pen gapPen   = rgb(0.10,0.34,0.62) + linewidth(1.25);
pen dotPen   = black;
pen lblpen   = black + fontsize(9pt);
pen mathpen  = black + fontsize(10.5pt);

// ----- a clearly star-shaped radius (pronounced rounded lobes) --------------
real Rstar(real theta, real phi, real seed) {
  real m = 1.0
         + 0.30*cos(3*(phi+seed))*sin(theta)*sin(theta)
         + 0.14*cos(2*theta);
  return 0.76*m;
}

triple bodyPoint(real theta, real phi, real seed, triple c, real s) {
  real r = s*Rstar(theta, phi, seed);
  return c + r*(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
}

surface bodySurf(real seed, triple c, real s, int nu=48, int nv=52) {
  return surface(new triple(pair t){ return bodyPoint(t.x,t.y,seed,c,s); },
                 (0,0),(pi,2pi),nu,nv,Spline);
}

real seedB = 1.2, sB = 1.10;
real seedA = 0.4, sA = 1.02;

// --- B is the obstacle below; centre fixed -----------------------------------
triple cB = (0.95, 0, -1.05);

// foot on B's upper lobe (facing up toward A)
real thF = pi*0.30, phF = 2.9;
triple foot = bodyPoint(thF, phF, seedB, cB, sB);

// outward normal n at the foot
real h = 1e-3;
triple Ft = bodyPoint(thF+h, phF, seedB, cB, sB) - foot;
triple Fp = bodyPoint(thF, phF+h, seedB, cB, sB) - foot;
triple nrm = unit(cross(Ft, Fp));
if (dot(nrm, foot - cB) < 0) nrm = -nrm;          // outward from B

// --- x sits a small gap above the foot, along n; A is placed so x is on dA ---
real gapLen = 0.55;
triple x = foot + gapLen*nrm;
// choose A's surface parameters and back out its centre so phi_A(thX,phX)=x.
real thX = pi*0.66, phX = -0.55;        // a lower-front lobe of A meets x
triple unitDirA = (sin(thX)*cos(phX), sin(thX)*sin(phX), cos(thX));
real rA = sA*Rstar(thX, phX, seedA);
triple cA = x - rA*unitDirA;            // guarantees bodyPoint(thX,phX,..)=x

// ----- draw the two bodies (A after B so its near lobe reads in front) ------
draw(bodySurf(seedB, cB, sB), surfacepen=fillB, meshpen=nullpen);
draw(bodySurf(seedA, cA, sA), surfacepen=fillA, meshpen=nullpen);

// centres
dot(cA, dotPen+linewidth(3.0));
dot(cB, dotPen+linewidth(3.0));
label("$c_A$", cA + (-0.20,0,0.10), align=W, lblpen);
label("$c_B$", cB + ( 0.28,0,-0.10), align=E, lblpen);

// ----- ray from c_B through the foot to x (thin dashed guide) ----------------
draw(cB -- x, rayPen);

// ----- outward normal n at the foot (reference arrow) -----------------------
triple noff = unit(cross(nrm, currentprojection.camera - foot));
draw(foot -- foot + 0.74*nrm, normPen, Arrow3(HookHead3, size=5.5));
label("$\mathbf{n}$", foot + 0.55*nrm - 0.18*noff, align=W, normPen);

// ----- signed gap g_N: a double-headed measure offset to the side -----------
triple offs = unit(cross(nrm, currentprojection.camera - foot));
triple goff = 0.22*offs;
draw((foot+goff) -- (x+goff), gapPen, Arrows3(HookHead3, size=4.2));
draw((foot+goff-0.06*nrm) -- (foot+goff+0.06*nrm), gapPen+linewidth(0.7));
draw((x+goff-0.06*nrm)    -- (x+goff+0.06*nrm),    gapPen+linewidth(0.7));
label("$g_N$", 0.5*(foot+x)+goff+0.30*offs, align=E, gapPen+fontsize(9pt));

// ----- markers and labels for x and the foot --------------------------------
dot(x, dotPen+linewidth(3.4));
dot(foot, dotPen+linewidth(3.4));
label("$\mathbf{x}=\varphi_A(\theta_A)$", x + (-0.10,0,0.34), align=N, lblpen);
label("$\varphi_B(\psi_B)$", foot + (0.34,0,-0.10), align=E, lblpen);

// ----- boxed formula caption inside the figure ------------------------------
// place in the empty upper-right area
label("$\boxed{\,\tau_{AB}=\varphi_B^{-1}\circ\varphi_A\,}$",
      (1.05, 0, 1.55), align=W, mathpen);

// invisible padding so titles/labels are not clipped
dot((cA.x-0.4, 0, 1.95), invisible);
dot((cB.x+0.6, 0, -2.10), invisible);
dot((3.05, 0, 1.55), invisible);   // right headroom for the formula box
